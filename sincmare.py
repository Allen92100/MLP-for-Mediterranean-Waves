from pathlib import Path
import pandas as pd
from utils import read_csv

# Legge il dataset mare, sistema la colonna temporale
# Costruisce i segmenti continui da usare nella pipeline.
def sync_seawaves_dataset(seawaves_dir, output_csv_path, input_csv_name, time_column="time", time_step_hours=1, force_rebuild=False):
    # Converte i percorsi in oggetti Path per gestire i file in modo semplice.
    output_csv_path = Path(output_csv_path)
    seawaves_dir = Path(seawaves_dir)

    # Se il file sincronizzato esiste gia', non lo ricostruisce.
    if output_csv_path.exists() and not force_rebuild:
        print(f"File gia' presente: {output_csv_path}")
        return output_csv_path
    
    # Costruisce il percorso del file mare da leggere.
    input_csv_path = seawaves_dir / input_csv_name
    if not input_csv_path.exists():
        raise FileNotFoundError(f"File mare non trovato: {input_csv_path}")
    
    # Legge il file CSV con gestione semplice del separatore.
    df = read_csv(input_csv_path)

    # Uniforma il nome della variabile target alla convenzione usata nella pipeline.
    if "Hs" in df.columns and "Hm" not in df.columns:
        df = df.rename(columns={"Hs": "Hm"})

    # Verifica che nel file siano presenti le colonne minime necessarie.
    if time_column not in df.columns:
        raise KeyError(f"Nel file mare manca la colonna '{time_column}'.")
    if "Hm" not in df.columns:
        raise KeyError("Nel file mare manca la colonna 'Hm' oppure 'Hs'.")

    # Converte il tempo in datetime, rimuove righe non valide e ordina cronologicamente.
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce").dt.round("h")
    df = df.dropna(subset=[time_column, "Hm"]).copy()
    df = df.sort_values(time_column).drop_duplicates(subset=[time_column]).reset_index(drop=True)
    
    # Individua i segmenti continui del dataset per evitare salti temporali nella fase dei lag.
    passo = pd.Timedelta(hours=time_step_hours)
    delta_t = df[time_column].diff()
    nuovo_segmento = (delta_t != passo).fillna(True)

    # Assegna un identificativo a ogni segmento e ne calcola la lunghezza.
    df["segment_id"] = nuovo_segmento.cumsum().astype(int)
    lunghezze = df.groupby("segment_id").size().rename("segment_length")
    df = df.merge(lunghezze, on="segment_id", how="left")

    # Salva il dataset sincronizzato su disco.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"Salvato {output_csv_path}")
    return output_csv_path