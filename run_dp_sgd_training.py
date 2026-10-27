"""
FedAvg trained with DP-SGD on every client (reviewer minor correction #3).

Each RSU trains its local model with Opacus DP-SGD (per-sample clipping C = 1.0, Gaussian
noise multiplier sigma in {0.5, 1.0, 2.0}, Poisson sampling with expected batch size 128,
Adam lr 0.002) for 1 local epoch per round over 12 rounds, the same budget as the
non-private FedAvg benchmark. The server aggregates with sample-weighted FedAvg.

Each client's privacy loss is tracked by its own Opacus RDP accountant (delta = 1e-5),
so the reported epsilon is the one spent by the run whose ADE is reported.
Results are written to results/dp_sgd_training_results.csv.
"""

import os
import re
import copy
import argparse
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.layers import DPLSTM

from config import OUTPUT_DIR, DEVICE, LEARNING_RATE, WEIGHT_DECAY, FEET_TO_METERS, HIDDEN_DIM, NUM_LAYERS, DROPOUT
from data_preprocessing import NGSIMPreprocessor, get_dataloaders
from models import Seq2SeqLSTM
from metrics import evaluate_model_on_loader, compute_ade

warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")

SIGMAS = [0.5, 1.0, 2.0]
SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 12
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
ZONES = {0: 'US-101 (Highway)', 1: 'Lankershim (Arterial)', 2: 'Peachtree (Downtown)'}


class DPSeq2SeqLSTM(Seq2SeqLSTM):
    """Seq2SeqLSTM with Opacus DPLSTM layers (same parameters and state_dict keys as nn.LSTM)."""
    def __init__(self):
        super().__init__()
        self.encoder.lstm = DPLSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.decoder_lstm = DPLSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)


def to_lstm_state_dict(dp_state):
    """Renames DPLSTM keys (lstm.l0.ih.weight) to nn.LSTM keys (lstm.weight_ih_l0)."""
    out = {}
    for k, v in dp_state.items():
        m = re.match(r"(.*)\.l(\d+)\.(ih|hh)\.(weight|bias)$", k)
        out[f"{m.group(1)}.{m.group(4)}_{m.group(3)}_l{m.group(2)}" if m else k] = v
    return out


def run_dp_fedavg(sigma, seed, client_loaders, scalers):
    torch.manual_seed(seed)
    np.random.seed(seed)
    global_model = DPSeq2SeqLSTM().to(DEVICE)
    criterion = nn.MSELoss()

    clients = {}
    for cid, loaders in client_loaders.items():
        engine = PrivacyEngine(accountant='rdp')
        model = DPSeq2SeqLSTM().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model, optimizer, loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loaders['train'],
            noise_multiplier=sigma, max_grad_norm=MAX_GRAD_NORM, poisson_sampling=True)
        clients[cid] = {'engine': engine, 'model': model, 'optimizer': optimizer, 'loader': loader,
                        'n': loaders['train_size'], 'steps': 0}

    for r in range(1, ROUNDS + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        states, counts = [], []
        for cid, c in clients.items():
            c['model']._module.load_state_dict(global_state)
            # A fresh Adam state every round, as in the non-private FedAvg clients
            c['optimizer'].original_optimizer.state = defaultdict(dict)
            c['model'].train()
            for batch in c['loader']:
                inputs, targets = batch[0].to(DEVICE), batch[1].to(DEVICE)
                if inputs.shape[0] == 0:
                    continue
                c['optimizer'].zero_grad()
                loss = criterion(c['model'](inputs), targets)
                loss.backward()
                c['optimizer'].step()
                c['steps'] += 1
            states.append(copy.deepcopy(c['model']._module.state_dict()))
            counts.append(c['n'])
        weights = np.array(counts, dtype=np.float64) / sum(counts)
        global_model.load_state_dict({k: sum(w * s[k].float() for w, s in zip(weights, states)) for k in states[0]})
        print(f"[DP-SGD sigma={sigma} seed={seed}] round {r:02d}/{ROUNDS}", flush=True)

    eval_model = Seq2SeqLSTM().to(DEVICE)
    eval_model.load_state_dict(to_lstm_state_dict(global_model.state_dict()))
    rows, preds, gts = [], [], []
    eps = {cid: c['engine'].get_epsilon(DELTA) for cid, c in clients.items()}
    for cid, loaders in client_loaders.items():
        res = evaluate_model_on_loader(eval_model, loaders['test'], scalers, device=DEVICE)
        preds.append(res['preds_raw_meters'])
        gts.append(res['gts_raw_meters'])
        rows.append({'Seed': seed, 'Noise Multiplier (sigma)': sigma, 'Domain': ZONES[cid],
                     'ADE (m)': res['ADE_m'], 'FDE (m)': res['FDE_m'],
                     'Epsilon (delta=1e-5)': eps[cid], 'Optimizer Steps': clients[cid]['steps'],
                     'Sample Rate q': 1.0 / len(client_loaders[cid]['train'])})
    pm, gm = np.concatenate(preds), np.concatenate(gts)
    rows.append({'Seed': seed, 'Noise Multiplier (sigma)': sigma, 'Domain': 'Global Pooled',
                 'ADE (m)': compute_ade(pm, gm),
                 'FDE (m)': float(np.mean(np.linalg.norm(pm[:, -1] - gm[:, -1], axis=-1))),
                 'Epsilon (delta=1e-5)': max(eps.values()), 'Optimizer Steps': None, 'Sample Rate q': None})
    print(f"[DP-SGD sigma={sigma} seed={seed}] pooled ADE {rows[-1]['ADE (m)']:.3f} m, eps per client "
          + ", ".join(f"{e:.2f}" for e in eps.values()), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--sigmas', type=float, nargs='+', default=SIGMAS)
    parser.add_argument('--out', default="dp_sgd_training_results.csv",
                        help='Output file in results/ (use one file per process when sigmas run in parallel)')
    args = parser.parse_args()

    preprocessor = NGSIMPreprocessor()
    normalized_data, scalers = preprocessor.prepare_dataset(force_recompute=False)
    client_loaders, _ = get_dataloaders(normalized_data, batch_size=128)

    rows = []
    for sigma in args.sigmas:
        for seed in args.seeds:
            rows.extend(run_dp_fedavg(sigma, seed, client_loaders, scalers))
            pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, args.out), index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(['Noise Multiplier (sigma)', 'Domain']).agg(
        ADE_mean=('ADE (m)', 'mean'), ADE_sd=('ADE (m)', 'std'), epsilon=('Epsilon (delta=1e-5)', 'max'))
    print(summary.to_string())


if __name__ == '__main__':
    main()
