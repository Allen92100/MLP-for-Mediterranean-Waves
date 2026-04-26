from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Modello MLP semplice per la stima dell'altezza significativa dell'onda.
class WaveMLPModel(nn.Module):
    """MLP semplice con un solo hidden layer."""

    def __init__(self, n_inputs):
        super().__init__()
        n_hidden = n_inputs + 2 # Numero di neuroni nel layer nascosto.
        self.hidden = nn.Linear(n_inputs, n_hidden)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x): # Propagazione in avanti del modello.
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# Gestisce training, validazione, salvataggio del modello e predizione.
class WaveMLPTrainer:
    """Gestisce addestramento, salvataggio e predizione."""

    def __init__(self, n_inputs, model_name, models_dir, errors_dir, lr=1e-5, device=None):
        if device is None:
            if torch.cuda.is_available(): # Seleziona automaticamente GPU o CPU in base alla disponibilita'.
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = WaveMLPModel(n_inputs).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # Ottimizzatore usato per aggiornare i pesi della rete.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=1e-6,
        ) # Scheduler che riduce il learning rate se la validation loss non migliora.
        self.criterion = nn.MSELoss() # Funzione di perdita usata durante il training.
        self.model_name = model_name

        self.models_dir = Path(models_dir)
        self.errors_dir = Path(errors_dir) / model_name
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.models_dir / f"{model_name}.pth"
        self.loss_csv_path = self.errors_dir / "loss.csv"
        self.loss_history = []
        self.val_loss_history = []

    # Salva su disco i pesi del modello corrente.
    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        path = Path(path)
        torch.save(self.model.state_dict(), path)
        return path
    # Carica da disco i pesi del modello salvato.
    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        path = Path(path)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    # Addestra il modello usando il set di train e controlla il miglioramento sul validation set.
    def fit(self, X_train, y_train, X_val, y_val, batch_size, epochs, patience, resume_if_available=False):
        if resume_if_available and self.model_path.exists(): # Se richiesto, riprende il training da un checkpoint gia' esistente.
            self.load_model(self.model_path)
            print(f"[wave_model] Ripreso checkpoint: {self.model_path}")

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Variabili usate per controllare il miglior modello e l'early stopping.
        best_val_loss = float("inf") 
        wait = 0

        for _ in tqdm(range(epochs), desc="Training MLP", colour="cyan"):
            # Modalita' training e azzeramento della loss accumulata nell'epoca.
            self.model.train()
            running_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Azzera i gradienti prima del nuovo passo di ottimizzazione.
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred.view(-1), y_batch.view(-1))
                loss.backward()
                self.optimizer.step()
                running_loss += float(loss.item())

            # Forward, calcolo della loss, backward e aggiornamento dei pesi.
            train_loss = running_loss / max(len(train_loader), 1)
            self.loss_history.append(train_loss)

            self.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                val_pred = self.model(X_val_tensor)
                val_loss = float(self.criterion(val_pred.view(-1), y_val_tensor.view(-1)).item())

            self.val_loss_history.append(val_loss)
            self.scheduler.step(val_loss) # Aggiorna il learning rate in base all'andamento della validation loss.

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                wait = 0
                self.save_model()
            else:
                wait += 1
                if wait >= patience:
                    print(f"[wave_model] Early stopping dopo {patience} epoche senza miglioramenti.")
                    break

        # Ricarica il miglior modello salvato durante il training.
        self.load_model(self.model_path)
        pd.DataFrame(
            {
                "train_loss": pd.Series(self.loss_history, dtype=float),
                "val_loss": pd.Series(self.val_loss_history, dtype=float),
            }
        ).to_csv(self.loss_csv_path, index=False)
    # Esegue la predizione del modello sul dataset fornito.
    def predict(self, X):
        self.model.eval() # Usa il modello in sola inferenza.
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            pred = self.model(X_tensor)
        return pred.cpu().numpy()
