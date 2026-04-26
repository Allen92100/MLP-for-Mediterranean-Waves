import math
import pickle
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

# funzione per la riproducibilità degli esperimenti
def set_seed(seed=42):
    """Imposta il seed per rendere ripetibili i risultati."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Crea una cartella se non esiste gia' e restituisce il relativo percorso.
def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Rimuove in modo semplice le cartelle di output da ricostruire.
def clean_directories(paths):
    import shutil

    for path in paths:
        shutil.rmtree(Path(path), ignore_errors=True)


# Legge un file CSV provando prima il separatore automatico.
def read_csv(csv_path):
    csv_path = Path(csv_path)
    try:
        return pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(csv_path)



# Cerca il nome della coordinata temporale nel dataset NetCDF. (nei dataset precedenti era "Hm", mentre in quelli aggiornati "Hs")
def detect_time_coordinate(ds):
    for name in ("time", "valid_time"):
        if name in ds.coords:
            return name
    raise KeyError("Coordinata temporale non trovata nel NetCDF")


# Cerca il nome della coordinata di latitudine.
def detect_lat_coordinate(ds):
    for name in ("lat", "latitude"):
        if name in ds.coords:
            return name
    raise KeyError("Coordinata di latitudine non trovata nel NetCDF")


# Cerca il nome della coordinata di longitudine.
def detect_lon_coordinate(ds):
    for name in ("lon", "longitude"):
        if name in ds.coords:
            return name
    raise KeyError("Coordinata di longitudine non trovata nel NetCDF")


# Uniforma nomi di coordinate e variabili dei dataset ERA5.
def standardize_wind_dataset(ds, source_name=None):
    rename_map = {}

    time_name = detect_time_coordinate(ds)
    lat_name = detect_lat_coordinate(ds)
    lon_name = detect_lon_coordinate(ds)

    if time_name != "time":
        rename_map[time_name] = "time"
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if "u10" in ds.data_vars:
        rename_map["u10"] = "uwnd"
    if "v10" in ds.data_vars:
        rename_map["v10"] = "vwnd"

    ds = ds.rename(rename_map)

    available_vars = []
    for var in ("uwnd", "vwnd"):
        if var in ds.data_vars:
            available_vars.append(var)

    if not available_vars:
        if source_name is None:
            raise KeyError("Nel file NetCDF non ci sono le variabili u10/v10")
        raise KeyError(f"Nel file NetCDF {source_name} non ci sono le variabili u10/v10")

    ds = ds[available_vars]
    ds = ds.sortby("time")
    ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values).round("h"))
    return ds


# Legge le informazioni temporali essenziali di un file NetCDF del vento.
def inspect_wind_file(nc_path):
    nc_path = Path(nc_path)
    with xr.open_dataset(nc_path) as ds:
        ds_std = standardize_wind_dataset(ds, source_name=nc_path.name)
        times = pd.DatetimeIndex(pd.to_datetime(ds_std["time"].values)).sort_values()

        if len(times) == 0:
            raise ValueError(f"Il file {nc_path.name} non contiene timestamp validi")

        return {
            "path": nc_path,
            "name": nc_path.name,
            "start_time": times.min(),
            "end_time": times.max(),
            "n_times": len(times),
            "variables": ",".join(sorted(ds_std.data_vars)),
        }


# Cerca il file successivo che estende temporalmente la serie gia' costruita.
def find_next_dataset_by_time(current_end_time, remaining_file_infos):
    candidates = []
    for info in remaining_file_infos:
        if pd.Timestamp(info["end_time"]) > pd.Timestamp(current_end_time):
            candidates.append(info)

    if not candidates:
        return None

    candidates.sort(key=lambda info: (pd.Timestamp(info["start_time"]), pd.Timestamp(info["end_time"]), info["name"]))
    return candidates[0]


# Costruisce la sequenza cronologica corretta dei file NetCDF del vento.
def build_chronological_wind_sequence(file_infos):
    file_infos = list(file_infos)
    if not file_infos:
        return []

    remaining = sorted(
        file_infos,
        key=lambda info: (pd.Timestamp(info["start_time"]), pd.Timestamp(info["end_time"]), info["name"]),
    )

    selected = []
    current = remaining.pop(0)
    selected.append(current)
    current_end = pd.Timestamp(current["end_time"])

    while remaining:
        next_info = find_next_dataset_by_time(current_end, remaining)
        if next_info is None:
            break
        remaining.remove(next_info)
        selected.append(next_info)
        current_end = max(current_end, pd.Timestamp(next_info["end_time"]))

    return selected


# Carica e ordina i timestamp del dataset mare sincronizzato.
def load_target_times(sync_csv_path, time_column="time"):
    df = read_csv(sync_csv_path)
    if time_column not in df.columns:
        raise KeyError(f"Nel file {sync_csv_path} manca la colonna '{time_column}'")

    times = pd.to_datetime(df[time_column], errors="coerce").round("h")
    times = times.dropna().drop_duplicates().sort_values()
    if len(times) == 0:
        raise ValueError(f"Nessun timestamp valido trovato in {sync_csv_path}")
    return pd.DatetimeIndex(times)


# Calcola la distanza approssimata tra due punti geografici in km.
def haversine(lon1, lat1, lon2, lat2):
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    return float(2.0 * earth_radius_km * np.arcsin(np.sqrt(a)))


# Costruisce il bounding box che contiene il raggio di interesse attorno alla boa.
def bounding_box_for_radius(lat0, lon0, max_km):
    dlat = max_km / 111.0
    dlon = max_km / (111.0 * max(np.cos(np.radians(lat0)), 1e-6))
    dlat *= 1.05
    dlon *= 1.05
    return lat0 - dlat, lat0 + dlat, lon0 - dlon, lon0 + dlon


# Seleziona i punti di griglia entro la distanza massima dalla boa.
def get_points(ds, lat0, lon0, max_km):
    lats = ds["lat"].values
    lons = ds["lon"].values
    points = []

    for lat in lats:
        for lon in lons:
            dist_km = haversine(lon0, lat0, float(lon), float(lat))
            if dist_km <= max_km:
                points.append((float(lat), float(lon), float(dist_km)))

    points.sort(key=lambda item: item[2])
    return points


# Associa i lag temporali ai punti di griglia in funzione della distanza dalla boa.
def distance_class_lags(distance_km):
    if distance_km <= 60.0:
        return [0, 1, 2, 3]
    if distance_km <= 120.0:
        return [2, 3, 4, 5]
    return [4, 5, 6, 7]


# Calcola le metriche principali usate per valutare il modello.
def performance_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    den = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / den) if den != 0.0 else np.nan
    mae = float(np.mean(np.abs(y_true - y_pred)))
    max_error = float(np.max(np.abs(residuals)))

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "nse": r2,
        "media_residuo": float(np.mean(residuals)),
        "varianza_residuo": float(np.var(residuals)),
        "errore_massimo": max_error,
        "residui": residuals,
    }


# Salva una figura nei formati png, eps e fig.
def save_figure_bundle(fig, output_dir, stem, dpi=300):
    output_dir = ensure_directory(output_dir)

    png_path = output_dir / f"{stem}.png"
    eps_path = output_dir / f"{stem}.eps"
    fig_path = output_dir / f"{stem}.fig"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")

    with open(fig_path, "wb") as file_out:
        pickle.dump(fig, file_out, protocol=pickle.HIGHEST_PROTOCOL)

    return {"png": png_path, "eps": eps_path, "fig": fig_path}


# Riapre una figura matplotlib salvata in formato .fig serializzato.
def load_pickled_figure(fig_path):
    with open(fig_path, "rb") as file_in:
        fig = pickle.load(file_in)
    return fig


# Genera la figura con i punti della griglia selezionati attorno alla boa.
def build_grid_points_figure(points_df, buoy_lat, buoy_lon, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        points_df["lon"],
        points_df["lat"],
        c=points_df["dist_km"],
        cmap="viridis",
        s=60,
        edgecolor="k",
        label="Punti griglia",
    )
    fig.colorbar(scatter, ax=ax, label="Distanza dalla boa (km)")
    ax.scatter(buoy_lon, buoy_lat, color="red", marker="*", s=200, label="Boa")
    ax.set_title(title)
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


# Genera il grafico dell'andamento delle loss di training e validazione.
def build_loss_figure(train_loss, val_loss):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss, label="Training loss", color="blue")
    if val_loss:
        ax.plot(val_loss, label="Validation loss", color="red")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


# Genera lo scatter plot tra valori reali e valori predetti.
def build_scatter_figure(y_true, y_pred, title):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=30, alpha=0.9, color="red", edgecolors="black")
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black")
    ax.set_xlabel("Hs reale (m)")
    ax.set_ylabel("Hs predetto (m)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


# Genera il confronto tra serie reale e serie predetta.
def build_series_figure(y_true, y_pred, title, real_label, pred_label):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label=real_label, color="blue", linewidth=1)
    ax.plot(y_pred, label=pred_label, color="red", linewidth=1, linestyle="dashed", alpha=0.85)
    ax.set_xlabel("Campione")
    ax.set_ylabel("Hs (m)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


# Genera l'istogramma dei residui del modello.
def build_residual_histogram_figure(residuals, title, color):
    residuals = np.asarray(residuals).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.75, color=color)
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Campioni")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig
