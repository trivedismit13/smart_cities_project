"""
Validation grid search for the FedProx and pFedMe baselines (reviewer minor correction #5).

- FedProx: proximal coefficient mu in {0.001, 0.01, 0.1}
- pFedMe: 3 x 3 grid over Moreau coefficient lamda in {0.1, 1.0, 15.0}
          and inner (personal) learning rate in {0.001, 0.005, 0.01}; K = 3, beta = 0.2

Every configuration is trained with the benchmark budget (12 rounds, 1 local epoch, seed 42)
and scored by pooled validation ADE. The test split is never used. The selected values are
written to results/tuning_selected.json, which run_full_journal_suite.py reads.
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
import torch

from config import OUTPUT_DIR, DEVICE
from data_preprocessing import NGSIMPreprocessor, get_dataloaders
from models import Seq2SeqLSTM
from metrics import evaluate_model_on_loader
from fl_framework import FederatedClient, FederatedServer, PFedMeClient, PFedMeServer

SEED = 42
ROUNDS = 12
FEDPROX_MUS = [0.001, 0.01, 0.1]
PFEDME_LAMDAS = [0.1, 1.0, 15.0]
PFEDME_LRS = [0.001, 0.005, 0.01]


def pooled_val_ade(models_per_client, client_loaders, scalers):
    """Pooled validation ADE = window-weighted mean of the per-client validation ADEs."""
    ades, sizes = [], []
    for cid, loaders in client_loaders.items():
        ades.append(evaluate_model_on_loader(models_per_client[cid], loaders['val'], scalers, device=DEVICE)['ADE_m'])
        sizes.append(loaders['val_size'])
    return float(np.average(ades, weights=sizes)), ades


def main():
    preprocessor = NGSIMPreprocessor()
    normalized_data, scalers = preprocessor.prepare_dataset(force_recompute=False)
    client_loaders, cent_loaders = get_dataloaders(normalized_data, batch_size=128)
    rows = []

    for mu in FEDPROX_MUS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clients = [FederatedClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=DEVICE)
                   for cid, l in client_loaders.items()]
        server = FederatedServer(Seq2SeqLSTM, clients, cent_loaders['test'], scalers, device=DEVICE)
        server.run_federated_training(num_rounds=ROUNDS, local_epochs=1, fedprox_mu=mu)
        pooled, per_client = pooled_val_ade({cid: server.global_model for cid in client_loaders}, client_loaders, scalers)
        rows.append({'Method': 'FedProx', 'mu': mu, 'lamda': None, 'lr_inner': None,
                     'Val ADE pooled (m)': pooled, 'Val ADE US-101 (m)': per_client[0],
                     'Val ADE Lankershim (m)': per_client[1], 'Val ADE Peachtree (m)': per_client[2]})
        print(f"[Tuning] FedProx mu={mu}: pooled val ADE {pooled:.4f} m", flush=True)

    for lamda, lr_inner in itertools.product(PFEDME_LAMDAS, PFEDME_LRS):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        clients = [PFedMeClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=DEVICE,
                                lamda=lamda, K_inner=3, lr_inner=lr_inner)
                   for cid, l in client_loaders.items()]
        server = PFedMeServer(Seq2SeqLSTM, clients, cent_loaders['test'], scalers, device=DEVICE)
        server.run_pfedme_training(num_rounds=ROUNDS, local_epochs=1, lamda=lamda, K_inner=3, lr_inner=lr_inner, beta=0.2)
        pooled, per_client = pooled_val_ade({c.client_id: c.personalized_model for c in server.clients}, client_loaders, scalers)
        rows.append({'Method': 'pFedMe', 'mu': None, 'lamda': lamda, 'lr_inner': lr_inner,
                     'Val ADE pooled (m)': pooled, 'Val ADE US-101 (m)': per_client[0],
                     'Val ADE Lankershim (m)': per_client[1], 'Val ADE Peachtree (m)': per_client[2]})
        print(f"[Tuning] pFedMe lamda={lamda}, lr_inner={lr_inner}: pooled val ADE {pooled:.4f} m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "tuning_grid_validation.csv"), index=False)
    best_prox = df[df['Method'] == 'FedProx'].sort_values('Val ADE pooled (m)').iloc[0]
    best_pfedme = df[df['Method'] == 'pFedMe'].sort_values('Val ADE pooled (m)').iloc[0]
    selected = {
        'fedprox_mu': float(best_prox['mu']),
        'pfedme_lamda': float(best_pfedme['lamda']),
        'pfedme_lr_inner': float(best_pfedme['lr_inner']),
        'fedprox_val_ade_m': float(best_prox['Val ADE pooled (m)']),
        'pfedme_val_ade_m': float(best_pfedme['Val ADE pooled (m)']),
    }
    with open(os.path.join(OUTPUT_DIR, "tuning_selected.json"), "w") as f:
        json.dump(selected, f, indent=2)
    print(df.to_string(index=False))
    print(f"Selected: {selected}")


if __name__ == '__main__':
    main()
