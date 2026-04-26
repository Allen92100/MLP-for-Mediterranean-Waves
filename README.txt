Struttura del progetto

File principali
- syncwaves.py / sincmare.py   sincronizzazione del dataset mare letto da seawaves/
- syncwind.py / sincvento.py   concatenazione dei NetCDF ERA5 letti da wind/
- crea_dataset_F33.py          costruzione del dataset finale e dei file .npy
- utils.py                     funzioni di supporto comuni
- wave_model.py                modello MLP e trainer PyTorch
- mainMLP.py                   file principale
- main_wave_NN.py              alias compatibile con il nome riportato in tesi

Cartelle richieste
- seawaves/    contiene il dataset mare in CSV
- wind/        contiene i file NetCDF ERA5

Cartelle prodotte
- uwnd/        NetCDF finale della componente zonale
- vwnd/        NetCDF finale della componente meridionale
- npy/         dataset scalato e scaler
- models/      pesi del modello
- errors/      storico delle loss
- results/     metriche e tabelle CSV
- figures/     grafici png, eps e fig

Esecuzione
1. Copiare il dataset mare in seawaves/
2. Copiare i file ERA5 in wind/
3. Eseguire:
   python mainMLP.py

Opzioni utili
- python mainMLP.py --clean-output
- python mainMLP.py --force-sync-wave --force-sync-wind --force-dataset --force-train

Nota su .fig
I file .fig vengono salvati come figure matplotlib serializzate con pickle.
Si possono riaprire in Python, ma non sono .fig nativi di MATLAB.
