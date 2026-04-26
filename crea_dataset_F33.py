from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import MinMaxScaler

from utils import bounding_box_for_radius, distance_class_lags, ensure_directory, get_points


def load_target_segmented_csv(path_ispra):
    """Legge il dataset mare sincronizzato e controlla le colonne minime."""
    df = pd.read_csv(path_ispra)

    colonne_richieste = {"time", "Hm", "segment_id"}
    mancanti = colonne_richieste.difference(df.columns)
    if mancanti:
        raise KeyError(f"Nel file sincronizzato mancano le colonne: {sorted(mancanti)}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.round("h")
    df = df.dropna(subset=["time", "Hm", "segment_id"]).copy()
    df["segment_id"] = df["segment_id"].astype(int)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df



def subset_wind_dataset(ds, buoy_lat, buoy_lon, max_km):
    """Riduce il dataset del vento al riquadro utile attorno alla boa."""
    lat_min, lat_max, lon_min, lon_max = bounding_box_for_radius(buoy_lat, buoy_lon, max_km)

    if ds["lat"][0] > ds["lat"][-1]:
        return ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))



def create_dataset_f33(
    uwnd_nc_path,
    vwnd_nc_path,
    synced_wave_csv,
    buoy_lat,
    buoy_lon,
    max_km,
    output_csv_path,
    npy_dir,
    split_train=0.70,
    split_val=0.15,
    split_test=0.15,
    study_start=None,
    study_end=None,
):
    """Costruisce il dataset F33 e i file .npy per train, validation e test."""
    if not np.isclose(split_train + split_val + split_test, 1.0):
        raise ValueError("Le percentuali di split devono sommare a 1.0")

    uwnd_nc_path = Path(uwnd_nc_path)
    vwnd_nc_path = Path(vwnd_nc_path)
    output_csv_path = Path(output_csv_path)
    npy_dir = ensure_directory(npy_dir)

    if not uwnd_nc_path.exists():
        raise FileNotFoundError(f"File uwnd non trovato: {uwnd_nc_path}")
    if not vwnd_nc_path.exists():
        raise FileNotFoundError(f"File vwnd non trovato: {vwnd_nc_path}")

    with xr.open_dataset(uwnd_nc_path) as ds_u, xr.open_dataset(vwnd_nc_path) as ds_v:
        ds_u_sub = subset_wind_dataset(ds_u, buoy_lat, buoy_lon, max_km)
        ds_v_sub = subset_wind_dataset(ds_v, buoy_lat, buoy_lon, max_km)

        if "uwnd" not in ds_u_sub.data_vars:
            raise KeyError("Variabile 'uwnd' non trovata nel file del vento zonale")
        if "vwnd" not in ds_v_sub.data_vars:
            raise KeyError("Variabile 'vwnd' non trovata nel file del vento meridionale")

        points = get_points(ds_u_sub, buoy_lat, buoy_lon, max_km)
        if not points:
            raise ValueError(f"Nessun punto di griglia trovato entro {max_km} km dalla boa")

        target_df = load_target_segmented_csv(synced_wave_csv)
        if study_start is not None:
            target_df = target_df[target_df["time"] >= pd.Timestamp(study_start)]
        if study_end is not None:
            target_df = target_df[target_df["time"] <= pd.Timestamp(study_end)]
        target_df = target_df.reset_index(drop=True)

        lats = [p[0] for p in points]
        lons = [p[1] for p in points]

        interp_u = ds_u_sub["uwnd"].interp(lat=xr.DataArray(lats, dims="points"), lon=xr.DataArray(lons, dims="points"))
        interp_v = ds_v_sub["vwnd"].interp(lat=xr.DataArray(lats, dims="points"), lon=xr.DataArray(lons, dims="points"))

        time_wind = pd.DatetimeIndex(pd.to_datetime(interp_u["time"].values)).round("h")

        colonne_vento = {}
        for i in range(len(points)):
            nome = f"p{i + 1:03d}"
            colonne_vento[f"uwnd_{nome}"] = interp_u.isel(points=i).values.astype(np.float32)
            colonne_vento[f"vwnd_{nome}"] = interp_v.isel(points=i).values.astype(np.float32)

        wind_df = pd.DataFrame(colonne_vento, index=time_wind).reset_index().rename(columns={"index": "time"})
        wind_df["time"] = pd.to_datetime(wind_df["time"], errors="coerce").dt.round("h")
        wind_df = wind_df.dropna(subset=["time"])
        wind_df = wind_df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    merged_df = pd.merge(target_df, wind_df, on="time", how="inner")
    merged_df = merged_df.dropna(subset=["time", "Hm", "segment_id"])
    merged_df = merged_df.sort_values("time").reset_index(drop=True)

    if merged_df.empty:
        raise ValueError("La fusione tra mare e vento ha prodotto un dataset vuoto")

    wind_feature_columns = []
    for colonna in merged_df.columns:
        if colonna.startswith("uwnd_") or colonna.startswith("vwnd_"):
            wind_feature_columns.append(colonna)

    point_distances = {}
    for i in range(len(points)):
        point_distances[f"p{i + 1:03d}"] = points[i][2]

    grouped_df = merged_df.groupby("segment_id", group_keys=False)
    max_lag = 7

    for colonna in wind_feature_columns:
        for lag in range(max_lag + 1):
            merged_df[f"{colonna}_t-{lag}"] = grouped_df[colonna].shift(lag)

    selected_feature_columns = []
    for colonna in wind_feature_columns:
        point_name = colonna.split("_")[-1]
        distanza = point_distances[point_name]
        lags = distance_class_lags(distanza)
        for lag in lags:
            selected_feature_columns.append(f"{colonna}_t-{lag}")

    output_columns = ["time", "Hm", "segment_id"] + selected_feature_columns
    dataset_f33 = merged_df[output_columns].dropna().reset_index(drop=True)

    if dataset_f33.empty:
        raise ValueError("Il dataset finale F33 e' vuoto dopo l'applicazione dei lag")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_f33.to_csv(output_csv_path, index=False)

    n_rows = len(dataset_f33)
    train_end = int(n_rows * split_train)
    val_end = int(n_rows * (split_train + split_val))

    train_df = dataset_f33.iloc[:train_end].copy()
    val_df = dataset_f33.iloc[train_end:val_end].copy()
    test_df = dataset_f33.iloc[val_end:].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Uno degli split train/val/test e' vuoto")

    X_train = train_df[selected_feature_columns].to_numpy(dtype=np.float32)
    y_train = train_df[["Hm"]].to_numpy(dtype=np.float32)
    X_val = val_df[selected_feature_columns].to_numpy(dtype=np.float32)
    y_val = val_df[["Hm"]].to_numpy(dtype=np.float32)
    X_test = test_df[selected_feature_columns].to_numpy(dtype=np.float32)
    y_test = test_df[["Hm"]].to_numpy(dtype=np.float32)

    scaler_X = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train)

    joblib.dump(scaler_X, npy_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, npy_dir / "scaler_y.pkl")

    np.save(npy_dir / "X_train.npy", scaler_X.transform(X_train).astype(np.float32))
    np.save(npy_dir / "y_train.npy", scaler_y.transform(y_train).astype(np.float32))
    np.save(npy_dir / "X_val.npy", scaler_X.transform(X_val).astype(np.float32))
    np.save(npy_dir / "y_val.npy", scaler_y.transform(y_val).astype(np.float32))
    np.save(npy_dir / "X_test.npy", scaler_X.transform(X_test).astype(np.float32))
    np.save(npy_dir / "y_test.npy", scaler_y.transform(y_test).astype(np.float32))

    np.save(npy_dir / "train_times.npy", train_df["time"].astype("datetime64[ns]").to_numpy())
    np.save(npy_dir / "val_times.npy", val_df["time"].astype("datetime64[ns]").to_numpy())
    np.save(npy_dir / "test_times.npy", test_df["time"].astype("datetime64[ns]").to_numpy())

    pd.DataFrame({"feature_name": selected_feature_columns}).to_csv(npy_dir / "feature_columns.csv", index=False)

    return {
        "dataset_path": output_csv_path,
        "n_rows": n_rows,
        "n_features": len(selected_feature_columns),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "points": points,
        "feature_columns": selected_feature_columns,
        "train_time_min": train_df["time"].min(),
        "train_time_max": train_df["time"].max(),
        "val_time_min": val_df["time"].min(),
        "val_time_max": val_df["time"].max(),
        "test_time_min": test_df["time"].min(),
        "test_time_max": test_df["time"].max(),
    }
