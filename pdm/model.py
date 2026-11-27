"""The autoencoder (PyTorch): squeezes a sensor reading through a small bottleneck and rebuilds it.

Trained only on HEALTHY data, it learns the normal relationships between sensors. When a machine starts to wear out,
the readings no longer fit what it learned, so the rebuild is poor and the reconstruction error rises."""
import copy

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, n_in, hidden=(32, 16), latent=3):
        super().__init__()
        sizes = [n_in, *hidden, latent]
        enc, dec = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        enc = enc[:-1]                                         # no activation on the bottleneck
        rsizes = sizes[::-1]
        for a, b in zip(rsizes[:-1], rsizes[1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec = dec[:-1]                                         # linear output layer
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)
        self.n_in, self.hidden, self.latent = n_in, tuple(hidden), latent

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X_train, X_val, hidden=(32, 16), latent=3, epochs=200, batch_size=128, lr=2e-3,
                      noise=0.05, patience=20, seed=0, verbose=False):
    """Train with mean squared error. Small input noise makes the model more robust. Stops early when validation loss stops improving."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Autoencoder(X_train.shape[1], hidden, latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    best, best_state, wait = float("inf"), None, 0
    history = {"train": [], "val": []}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.tensor(rng.permutation(len(Xt)))
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            xb = Xt[order[i:i + batch_size]]
            noisy = xb + noise * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(noisy), xb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), Xv).item()
        history["train"].append(total / len(Xt))
        history["val"].append(val)
        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  train loss {history['train'][-1]:.4f}  validation loss {val:.4f}")
        if val < best - 1e-5:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    history["best_val"], history["epochs_run"] = best, len(history["val"])
    return model, history


def reconstruct_fn(model):
    """A function that maps a NumPy array to the model's reconstruction of it."""
    def run(X):
        with torch.no_grad():
            return model(torch.tensor(np.asarray(X), dtype=torch.float32)).numpy()
    return run
