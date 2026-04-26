import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crea_dataset_F33 import create_dataset_f33
from sincmare import sync_seawaves_dataset
from sincvento import sync_wind_datasets
from utils import *
from wave_model import WaveMLPTrainer


# Directories
# Sottocartelle dei dataset prelevati da RON e ERA5 
SEAWAVES_DIR = Path("seawaves")
WIND_DIR = Path("wind")
UWND_OUTPUT_NC = Path("uwnd/uwnd_merged.nc")
VWND_OUTPUT_NC = Path("vwnd/vwnd_merged.nc")

# Sottocartelle contenenti i dataset uniti, il dataset finale e gli array già allenati 
SYNC_WAVE_CSV = Path("DatasetSync.csv")
DATASET_F33_CSV = Path("dataset_F33.csv")
NPY_DIR = Path("npy")

# Sottocartelle contenenti i risultati, il modello e gli errori
MODELS_DIR = Path("models")
ERRORS_DIR = Path("errors")
RESULTS_DIR = Path("results")
RESULTS_MAPS_DIR = Path("results/maps")
FIGURES_DIR = Path("figures")
FIGURES_MAPS_DIR = Path("figures/maps")

# Iperparametri
BUOY_LAT = 37.51 #latitudine
BUOY_LON = 12.53 #longitudine
MAX_KM = 180.0

MODEL_NAME = "F33_main"
SEED = 42
EPOCHS = 100
BATCH_SIZE = 256
PATIENCE = 100
LEARNING_RATE = 1e-5
DEVICE = None



# Carica i file npy per training validazione e test
def load_training_arrays(npy_dir):
    npy_dir = Path(npy_dir)
    return {
        "X_train": np.load(npy_dir / "X_train.npy"),
        "y_train": np.load(npy_dir / "y_train.npy"),
        "X_val": np.load(npy_dir / "X_val.npy"),
        "y_val": np.load(npy_dir / "y_val.npy"),
        "X_test": np.load(npy_dir / "X_test.npy"),
        "y_test": np.load(npy_dir / "y_test.npy"),
    }


# Salva le metriche finali di train e test in formato CSV e JSON, così da mantenere una traccia leggibile dei risultati ottenuti nelle diverse esecuzioni
def save_metrics(outputs_dir, metrics_train, metrics_test):
    outputs_dir = ensure_directory(outputs_dir)

    rows = [
        {"split": "train", **{k: v for k, v in metrics_train.items() if k != "residui"}},
        {"split": "test", **{k: v for k, v in metrics_test.items() if k != "residui"}},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(outputs_dir / "metrics_summary.csv", index=False)
    table.to_json(outputs_dir / "metrics_summary.json", orient="records", indent=2)


# Salva i contronti tra valori reali e predetti per il training e il test
def save_prediction_tables(outputs_dir, y_true_train, y_pred_train, y_true_test, y_pred_test):
    outputs_dir = ensure_directory(outputs_dir)

    pd.DataFrame({
        "Real_Hs": np.asarray(y_true_train).reshape(-1),
        "Predicted_Hs": np.asarray(y_pred_train).reshape(-1),
    }).to_csv(outputs_dir / "confronto_train.csv", index=False)

    pd.DataFrame({
        "Real_Hs": np.asarray(y_true_test).reshape(-1),
        "Predicted_Hs": np.asarray(y_pred_test).reshape(-1),
    }).to_csv(outputs_dir / "confronto_test.csv", index=False)


# Salva l'elenco dei punti di griglia attorno la boa.
def save_points_inventory(points, output_dir):
    output_dir = ensure_directory(output_dir)
    points_df = pd.DataFrame(points, columns=["lat", "lon", "dist_km"])
    csv_path = output_dir / "grid_points.csv"
    points_df.to_csv(csv_path, index=False)
    return csv_path




# Main
def main():
    set_seed(SEED) # riproducibilità

    # Crea le cartelle di output necessarie alla pipeline.
    ensure_directory(RESULTS_DIR)
    ensure_directory(RESULTS_MAPS_DIR)
    ensure_directory(FIGURES_DIR)
    ensure_directory(FIGURES_MAPS_DIR)

    # Sincronizza il dataset mare e costruisce i segmenti temporali continui.
    print("Sincronizzazione dataset mare...")
    sync_seawaves_dataset(
        seawaves_dir=SEAWAVES_DIR,
        output_csv_path=SYNC_WAVE_CSV,
        input_csv_name="DatasetBuoy.csv",
        force_rebuild=False
    )

    # Sincronizza i file del vento e li allinea temporalmente ai dati mare.
    print("Sincronizzazione dataset vento...")
    sync_wind_datasets(
        wind_dir=WIND_DIR,
        target_sync_csv=SYNC_WAVE_CSV,
        uwnd_output_path=UWND_OUTPUT_NC,
        vwnd_output_path=VWND_OUTPUT_NC,
        inventory_output_dir=RESULTS_DIR,
        force_rebuild=False
    )
    # Controlla se il dataset finale o i file .npy devono essere ricostruiti.
    print("Costruzione dataset F33...")
    npy_names = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "X_test.npy", "y_test.npy", "scaler_y.pkl"]
    npy_missing = not all((NPY_DIR / name).exists() for name in npy_names)
    dataset_needs_rebuild = (not DATASET_F33_CSV.exists()) or npy_missing

    # Costruisce il dataset finale con le feature spaziali e temporali del vento.
    if dataset_needs_rebuild or npy_missing:
        dataset_info = create_dataset_f33(
            uwnd_nc_path=UWND_OUTPUT_NC,
            vwnd_nc_path=VWND_OUTPUT_NC,
            synced_wave_csv=SYNC_WAVE_CSV,
            buoy_lat=BUOY_LAT,
            buoy_lon=BUOY_LON,
            max_km=MAX_KM,
            output_csv_path=DATASET_F33_CSV,
            npy_dir=NPY_DIR,
            split_train=0.70,
            split_val=0.15,
            split_test=0.15,
        )

        to_save = {}
        for key, value in dataset_info.items():
            if key in ["points", "feature_columns"]:
                continue
            if isinstance(value, (Path, pd.Timestamp)):
                to_save[key] = str(value)
            else:
                to_save[key] = value

        with open(RESULTS_DIR / "dataset_info.json", "w", encoding="utf-8") as file_out:
            json.dump(to_save, file_out, ensure_ascii=False, indent=2)
    else:
        dataset_info = None

    # Carica gli array numerici e lo scaler del target gia' salvati su disco.
    arrays = load_training_arrays(NPY_DIR)
    scaler_y = joblib.load(NPY_DIR / "scaler_y.pkl")

    # Inizializza il trainer del modello MLP.
    trainer = WaveMLPTrainer(
        n_inputs=arrays["X_train"].shape[1],
        model_name=MODEL_NAME,
        models_dir=MODELS_DIR,
        errors_dir=ERRORS_DIR,
        lr=LEARNING_RATE,
        device=DEVICE,
    )

    # Se il modello non esiste, viene addestrato; altrimenti viene caricato da disco.
    if not trainer.model_path.exists():
        print(" Training del modello...")
        trainer.fit(
            X_train=arrays["X_train"],
            y_train=arrays["y_train"],
            X_val=arrays["X_val"],
            y_val=arrays["y_val"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            patience=PATIENCE,
            resume_if_available=False,
        )
    else:
        print(f" Modello gia' presente: {trainer.model_path}")
        trainer.load_model()

    # Esegue le predizioni sui set di train e test.
    print("Predizioni...")
    y_pred_test = trainer.predict(arrays["X_test"])
    y_pred_train = trainer.predict(arrays["X_train"])

    # Riporta valori reali e predetti alla scala fisica originale.
    y_true_test = scaler_y.inverse_transform(arrays["y_test"])
    y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)
    y_true_train = scaler_y.inverse_transform(arrays["y_train"])
    y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)

    # Calcola le metriche di prestazione del modello.
    metrics_test = performance_metrics(y_true_test, y_pred_test_inv)
    metrics_train = performance_metrics(y_true_train, y_pred_train_inv)

    print("Metriche TEST")
    for key, value in metrics_test.items():
        if key != "residui":
            print(f"  {key}: {value}")

    print("Metriche TRAIN")
    for key, value in metrics_train.items():
        if key != "residui":
            print(f"  {key}: {value}")

    # Salva metriche e tabelle di confronto tra valori reali e predetti.
    save_metrics(RESULTS_DIR, metrics_train, metrics_test)
    save_prediction_tables(RESULTS_DIR, y_true_train, y_pred_train_inv, y_true_test, y_pred_test_inv)

    # Recupera o salva l'inventario dei punti di griglia usati nel dataset.
    if dataset_info is not None:
        points = dataset_info["points"]
    else:
        points = []
        old_points_path = RESULTS_MAPS_DIR / "grid_points.csv"
        if old_points_path.exists():
            old_points_df = pd.read_csv(old_points_path)
            points = list(old_points_df[["lat", "lon", "dist_km"]].itertuples(index=False, name=None))

    # Genera e salva le figure finali dell'esperimento. (.FIG)
    if points:
        save_points_inventory(points, RESULTS_MAPS_DIR)
        points_df = pd.DataFrame(points, columns=["lat", "lon", "dist_km"])
        fig = build_grid_points_figure(points_df, BUOY_LAT, BUOY_LON, "Punti della griglia entro 180 km dalla boa")
        save_figure_bundle(fig, FIGURES_MAPS_DIR, "grid_points_map")
        plt.close(fig)

    fig = build_loss_figure(trainer.loss_history, trainer.val_loss_history)
    save_figure_bundle(fig, FIGURES_DIR, "loss")
    plt.close(fig)

    fig = build_scatter_figure(y_true_test, y_pred_test_inv, "Confronto Hs reale vs predetto - Test")
    save_figure_bundle(fig, FIGURES_DIR, "scatter_test")
    plt.close(fig)

    fig = build_scatter_figure(y_true_train, y_pred_train_inv, "Confronto Hs reale vs predetto - Train")
    save_figure_bundle(fig, FIGURES_DIR, "scatter_train")
    plt.close(fig)

    fig = build_series_figure(y_true_test, y_pred_test_inv, "Confronto Hs reale vs predetto - Test", "Reale test", "Predetto test")
    save_figure_bundle(fig, FIGURES_DIR, "reale_vs_predetta_test")
    plt.close(fig)

    fig = build_series_figure(y_true_train, y_pred_train_inv, "Confronto Hs reale vs predetto - Train", "Reale train", "Predetto train")
    save_figure_bundle(fig, FIGURES_DIR, "reale_vs_predetta_train")
    plt.close(fig)

    fig = build_residual_histogram_figure(metrics_test["residui"], "Istogramma dei residui - Test set", "violet")
    save_figure_bundle(fig, FIGURES_DIR, "istogramma_residui_test")
    plt.close(fig)

    fig = build_residual_histogram_figure(metrics_train["residui"], "Istogramma dei residui - Train set", "orange")
    save_figure_bundle(fig, FIGURES_DIR, "istogramma_residui_train")
    plt.close(fig)

    print(f" Fine. Figure in {FIGURES_DIR}")
    print(f" Risultati in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
