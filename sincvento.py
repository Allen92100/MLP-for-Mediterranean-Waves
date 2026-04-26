from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import (
    build_chronological_wind_sequence,
    ensure_directory,
    inspect_wind_file,
    load_target_times,
    standardize_wind_dataset,
)

# Tiene solo la parte del dataset che estende davvero la serie temporale.
# In questo modo non vengono riaggiunti intervalli gia' presenti.
def append_only_new_times(ds, last_time):
    
    # Se non esiste ancora un tempo finale di riferimento, il dataset viene preso interamente.
    if last_time is None:
        return ds
    
    # Seleziona solo i timestamp successivi a quelli gia' acquisiti.
    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).round("h")
    mask = times > pd.Timestamp(last_time)
    idx = np.where(mask)[0]
    return ds.isel(time=idx)


# Legge i file NetCDF del vento, li ordina temporalmente,
# li concatena senza sovrapposizioni e li allinea al dataset mare.
def sync_wind_datasets(wind_dir, target_sync_csv, uwnd_output_path, vwnd_output_path, inventory_output_dir=None, force_rebuild=False):
    # Converte i percorsi in oggetti Path per gestire file e cartelle in modo uniforme.
    wind_dir = Path(wind_dir)
    uwnd_output_path = Path(uwnd_output_path)
    vwnd_output_path = Path(vwnd_output_path)

    # Se i file finali del vento esistono gia', non rifà la sincronizzazione.
    if uwnd_output_path.exists() and vwnd_output_path.exists() and not force_rebuild:
        print(" File del vento gia' presenti.")
        return uwnd_output_path, vwnd_output_path

    # Verifica che la cartella contenente i NetCDF del vento sia presente.
    if not wind_dir.exists():
        raise FileNotFoundError(f"Cartella wind non trovata: {wind_dir}")

    # Cerca tutti i file NetCDF disponibili nella cartella wind.
    nc_files = sorted(wind_dir.glob("*.nc"))

    # Se non trova file NetCDF, interrompe l'esecuzione con errore esplicito.
    if not nc_files:
        raise FileNotFoundError(f"Nessun file .nc trovato in {wind_dir}")
    
    # Legge per ogni file le informazioni temporali necessarie alla ricostruzione della sequenza.
    file_infos = []
    for path in nc_files:
        file_infos.append(inspect_wind_file(path))

    # Ricostruisce l'ordine corretto dei file usando le date contenute nei dataset.
    ordered_infos = build_chronological_wind_sequence(file_infos)
    # Se non riesce a costruire una sequenza coerente, interrompe la pipeline.
    if not ordered_infos:
        raise RuntimeError("Non sono riuscito a costruire la sequenza temporale del vento.")
    
    # Stampa a video la sequenza temporale selezionata per controllo rapido.
    print(" Sequenza selezionata:")
    for info in ordered_infos:
        print(
            f"  - {info['name']} | {info['start_time']} -> {info['end_time']} | "
            f"n_times={info['n_times']}"
        )

    # Se richiesto, salva su disco l'inventario completo dei file e la sequenza finale scelta.
    if inventory_output_dir is not None:
        inventory_output_dir = ensure_directory(inventory_output_dir)
        pd.DataFrame(file_infos).to_csv(inventory_output_dir / "wind_inventory.csv", index=False)
        pd.DataFrame(ordered_infos).to_csv(inventory_output_dir / "wind_sequence.csv", index=False)

    # Prepara la lista dei dataset da concatenare e il riferimento temporale progressivo.
    datasets = []
    last_time = None
    
    # Apre ciascun file, uniforma la struttura e tiene solo la parte temporale utile.
    for info in ordered_infos:
        # Uniforma nomi di coordinate e variabili per trattare tutti i file nello stesso modo.
        ds = xr.open_dataset(info["path"])
        ds = standardize_wind_dataset(ds, source_name=info["name"])
        ds = ds.sortby("time")
        ds = append_only_new_times(ds, last_time)  # Elimina eventuali intervalli che non estendono la serie temporale gia' costruita.

        # Se dopo il taglio non resta nessun istante utile, il file viene ignorato.
        if ds.sizes.get("time", 0) == 0:
            ds.close()
            continue
        # Aggiorna il tempo finale della serie e aggiunge il dataset alla lista finale.
        times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).round("h")
        last_time = times.max()
        datasets.append(ds)

    # Verifica che almeno un dataset utile sia stato selezionato.
    if not datasets:
        raise RuntimeError("Nessun dataset utile dopo il controllo cronologico.")
    
    # Unisce tutti i dataset selezionati lungo la dimensione temporale.
    ds_all = xr.concat(datasets, dim="time")
    ds_all = ds_all.sortby("time")

    # Rimuove eventuali timestamp duplicati rimasti dopo la concatenazione.
    time_values = pd.DatetimeIndex(pd.to_datetime(ds_all["time"].values)).round("h")
    _, unique_indices = np.unique(time_values.values, return_index=True)
    ds_all = ds_all.isel(time=np.sort(unique_indices))

    # Mantiene solo gli istanti temporali comuni tra vento e dataset mare sincronizzato.
    target_times = load_target_times(target_sync_csv)
    available_times = pd.DatetimeIndex(pd.to_datetime(ds_all["time"].values)).round("h")
    common_times = target_times.intersection(available_times)

    # Se non ci sono tempi in comune con il dataset mare, il dataset vento non e' utilizzabile.
    if len(common_times) == 0:
        raise ValueError("Nessun timestamp in comune tra vento e dataset mare.")

    # Seleziona dal dataset concatenato solo l'intervallo temporale utile alla pipeline.
    ds_aligned = ds_all.sel(time=common_times)

    # Verifica che nel dataset finale siano presenti entrambe le componenti del vento.
    if "uwnd" not in ds_aligned.data_vars or "vwnd" not in ds_aligned.data_vars:
        raise KeyError("Nel dataset finale non risultano entrambe le componenti uwnd e vwnd.")

    # Crea le cartelle di output dei file finali del vento.
    ensure_directory(uwnd_output_path.parent)
    ensure_directory(vwnd_output_path.parent)

    # Salva separatamente le due componenti del vento nei file NetCDF finali.
    ds_aligned[["uwnd"]].to_netcdf(
        uwnd_output_path,
        encoding={"uwnd": {"zlib": True, "complevel": 4}},
    )
    ds_aligned[["vwnd"]].to_netcdf(
        vwnd_output_path,
        encoding={"vwnd": {"zlib": True, "complevel": 4}},
    )
    
    # Stampa un riepilogo del periodo comune e dei file generati.
    print(
        f" Salvati:\n"
        f"  - {uwnd_output_path}\n"
        f"  - {vwnd_output_path}\n"
        f"  - periodo comune: {common_times.min()} -> {common_times.max()}\n"
        f"  - istanti comuni: {len(common_times)}"
    )
    # Chiude tutti i dataset aperti per liberare memoria.
    for ds in datasets:
        ds.close()
    ds_all.close()
    ds_aligned.close()

    return uwnd_output_path, vwnd_output_path
