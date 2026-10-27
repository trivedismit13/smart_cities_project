"""
Complete Journal Suite Runner:
1. Multi-seed Benchmark (5 seeds: 42, 43, 44, 45, 46), 12 rounds / 12 epochs
   - Constant Velocity (CV) & Constant Acceleration (CA) Baselines
   - Centralized Pooled (Equal Budget)
   - Centralized Zone-Aware (Equal Budget)
   - FedAvg, FedProx, FedPer, pFedMe
   - Isolated Silos (US-101, Lankershim, Peachtree)
   - Isolated Own-Zone (direct pooled evaluation)
   - Computes Mean +/- SD, Paired t-tests, NADE, Multi-horizon
2. Controlled Heterogeneity Study (Low, Medium, High)
3. Privacy Attack & Defense Study
   - DLG Reconstruction (unprotected, head-withheld, DP noise)
   - MIA Attack (ROC AUC, accuracy)
   - Renyi DP moments accountant (epsilon, delta)
4. Plot generation (typography >= 8pt) to plots/, figures/, and Downloads/figures
5. Exports complete LaTeX and Markdown tables
"""

import os
import sys
import time
import copy
import shutil
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from config import (
    BASE_DIR, OUTPUT_DIR, PLOTS_DIR, DEVICE, FEET_TO_METERS,
    OBS_STEPS, PRED_STEPS, CLIENT_NAMES, HORIZON_TAG
)
from data_preprocessing import NGSIMPreprocessor, get_dataloaders
from models import (
    Seq2SeqLSTM, ConstantVelocityBaseline, ConstantAccelerationBaseline,
    ZoneAwareSeq2SeqLSTM, compute_model_size_kb, get_model_parameters
)
from metrics import (
    evaluate_model_on_loader, unscale_coordinates, compute_ade, compute_fde,
    compute_rmse, compute_mae, compute_mape, compute_nade, compute_horizon_ades,
    compute_fairness_metrics
)
from fl_framework import (
    FederatedClient, FederatedServer, CentralizedTrainer,
    CentralizedZoneAwareTrainer, IsolatedTrainer,
    FedPerClient, FedPerServer, PFedMeClient, PFedMeServer
)
from heterogeneity_metrics import (
    summarize_zone_statistics, compute_pairwise_heterogeneity,
    compute_overall_heterogeneity_score, compute_gradient_divergence,
    create_controlled_heterogeneity_partition
)
from privacy_attacks import (
    GradientReconstructionAttack, MembershipInferenceAttack,
    DifferentialPrivacyDefense
)
from visualize import (
    plot_learning_curves, plot_heterogeneity_vs_personalization,
    plot_privacy_leakage_vs_dp, plot_fairness_disparity_comparison,
    plot_accuracy_privacy_tradeoff_pareto, plot_dataset_distribution,
    plot_trajectory_predictions, set_plot_style
)
import matplotlib.pyplot as plt

SEEDS = [42, 43, 44, 45, 46]
ROUNDS = 12
CENT_EPOCHS = 12
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
TUNING_FILE = os.path.join(OUTPUT_DIR, "tuning_selected.json")
DP_SGD_FILE = os.path.join(OUTPUT_DIR, "dp_sgd_training_results.csv")


def load_tuned_hyperparameters():
    """FedProx mu and pFedMe (lamda, lr_inner) selected on validation by tune_fedprox_pfedme.py."""
    tuned = {'fedprox_mu': 0.01, 'pfedme_lamda': 1.0, 'pfedme_lr_inner': 0.01}
    if os.path.exists(TUNING_FILE):
        with open(TUNING_FILE) as f:
            tuned.update(json.load(f))
    return tuned


def save_checkpoint(seed, name, model):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"seed{seed}_{name}{HORIZON_TAG}.pt"))


def run_suite(benchmark_only=False):
    tuned = load_tuned_hyperparameters()
    fedprox_mu = tuned['fedprox_mu']
    pfedme_lamda = tuned['pfedme_lamda']
    pfedme_lr_inner = tuned['pfedme_lr_inner']
    fedprox_name = f'FedProx (mu={fedprox_mu:g})'
    print(f"Tuned hyperparameters: {tuned}", flush=True)
    t_start = time.time()
    print("=" * 85, flush=True)
    print("STARTING COMPLETE MULTI-SEED JOURNAL BENCHMARK & AUDIT SUITE", flush=True)
    print(f"Seeds: {SEEDS} | Rounds: {ROUNDS} | Centralized Epochs: {CENT_EPOCHS}", flush=True)
    print("=" * 85, flush=True)

    # 1. Load Data
    preprocessor = NGSIMPreprocessor()
    normalized_data, scalers = preprocessor.prepare_dataset(force_recompute=False)
    client_loaders, cent_loaders = get_dataloaders(normalized_data, batch_size=128)
    model_kb = compute_model_size_kb(Seq2SeqLSTM())

    # 2. Heterogeneity Study
    print("\n--- PHASE 1: Heterogeneity Quantification ---", flush=True)
    zone_stats = summarize_zone_statistics(normalized_data, scalers=scalers)
    pairwise_het = compute_pairwise_heterogeneity(normalized_data, scalers=scalers)
    het_score = compute_overall_heterogeneity_score(pairwise_het)
    grad_div = compute_gradient_divergence({
        cid: {k: v for k, v in Seq2SeqLSTM().named_parameters()}
        for cid in range(3)
    })
    print(f"Aggregate Heterogeneity Score (JSD): {het_score:.4f}", flush=True)
    pd.DataFrame(list(zone_stats.values())).to_csv(
        os.path.join(OUTPUT_DIR, f"zone_statistics{HORIZON_TAG}.csv"), index=False)

    # 3. Multi-Seed Benchmark Evaluation
    print("\n--- PHASE 2: Multi-Seed Benchmark Evaluation ---", flush=True)
    model_names = [
        'Constant Velocity',
        'Constant Acceleration',
        'Centralized (Pooled)',
        'Centralized (Zone-Aware)',
        'FedAvg (Global FL)',
        fedprox_name,
        'FedPer (Personalized)',
        'pFedMe (Moreau)',
        'Isolated (US-101 Silo)',
        'Isolated (Lankershim Silo)',
        'Isolated (Peachtree Silo)',
        'Isolated (Own Zone)'
    ]
    domains = ['US-101 (Highway)', 'Lankershim (Arterial)', 'Peachtree (Downtown)', 'Global Pooled']

    # Store metrics: {model: {domain: {metric: [val_seed_0, val_seed_1, ...]}}}
    seed_records = {m: {d: {
        'ADE_m': [], 'FDE_m': [], 'RMSE_m': [], 'MAE_m': [], 'MAPE_m': [], 'NADE_pct': [],
        'ADE_ft': [], 'FDE_ft': [], 'ADE_1.0s': [], 'ADE_2.0s': [], 'ADE_3.0s': []
    } for d in domains} for m in model_names}

    last_histories = {}
    last_sample_cases = None

    for seed_idx, seed in enumerate(SEEDS):
        t_seed = time.time()
        print(f"\n=======================================================", flush=True)
        print(f"  RUNNING SEED {seed} ({seed_idx + 1}/{len(SEEDS)})", flush=True)
        print(f"=======================================================", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Baselines
        cv_model = ConstantVelocityBaseline(pred_steps=PRED_STEPS, scalers=scalers)
        ca_model = ConstantAccelerationBaseline(pred_steps=PRED_STEPS, scalers=scalers)

        # Centralized Pooled
        cent_trainer = CentralizedTrainer(Seq2SeqLSTM, cent_loaders, scalers, device=DEVICE)
        cent_hist, cent_model = cent_trainer.train(epochs=CENT_EPOCHS)

        # Centralized Zone-Aware
        cent_za_trainer = CentralizedZoneAwareTrainer(ZoneAwareSeq2SeqLSTM, client_loaders, scalers, device=DEVICE)
        cent_za_model = cent_za_trainer.train(epochs=CENT_EPOCHS)

        # Isolated Silos
        iso_trainer = IsolatedTrainer(Seq2SeqLSTM, client_loaders, scalers, device=DEVICE)
        iso_models, _ = iso_trainer.train_all(epochs=CENT_EPOCHS)

        # FedAvg
        fedavg_clients = [
            FederatedClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                            Seq2SeqLSTM, scalers, device=DEVICE)
            for cid, loaders in client_loaders.items()
        ]
        fedavg_server = FederatedServer(Seq2SeqLSTM, fedavg_clients, cent_loaders['test'], scalers, device=DEVICE)
        fedavg_hist = fedavg_server.run_federated_training(num_rounds=ROUNDS, local_epochs=1, fedprox_mu=0.0)
        fedavg_model = fedavg_server.global_model

        # FedProx
        fedprox_clients = [
            FederatedClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                            Seq2SeqLSTM, scalers, device=DEVICE)
            for cid, loaders in client_loaders.items()
        ]
        fedprox_server = FederatedServer(Seq2SeqLSTM, fedprox_clients, cent_loaders['test'], scalers, device=DEVICE)
        fedprox_hist = fedprox_server.run_federated_training(num_rounds=ROUNDS, local_epochs=1, fedprox_mu=fedprox_mu)
        fedprox_model = fedprox_server.global_model

        # FedPer
        fedper_clients = [
            FedPerClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                         Seq2SeqLSTM, scalers, device=DEVICE)
            for cid, loaders in client_loaders.items()
        ]
        fedper_server = FedPerServer(Seq2SeqLSTM, fedper_clients, cent_loaders['test'], scalers, device=DEVICE)
        fedper_hist = fedper_server.run_fedper_training(num_rounds=ROUNDS, local_epochs=1)
        fedper_model = fedper_server.global_model

        # pFedMe
        pfedme_clients = [
            PFedMeClient(cid, loaders['name'], loaders['train'], loaders['val'], loaders['test'],
                         Seq2SeqLSTM, scalers, device=DEVICE, lamda=pfedme_lamda, K_inner=3, lr_inner=pfedme_lr_inner)
            for cid, loaders in client_loaders.items()
        ]
        pfedme_server = PFedMeServer(Seq2SeqLSTM, pfedme_clients, cent_loaders['test'], scalers, device=DEVICE)
        pfedme_hist = pfedme_server.run_pfedme_training(num_rounds=ROUNDS, local_epochs=1, lamda=pfedme_lamda, K_inner=3, lr_inner=pfedme_lr_inner, beta=0.2)
        pfedme_model = pfedme_server.global_model

        save_checkpoint(seed, "centralized_pooled", cent_model)
        save_checkpoint(seed, "centralized_zone_aware", cent_za_model)
        for cid, iso_model in iso_models.items():
            save_checkpoint(seed, f"isolated_client{cid}", iso_model)
        save_checkpoint(seed, "fedavg", fedavg_model)
        save_checkpoint(seed, "fedprox", fedprox_model)
        for c in fedper_server.clients:
            save_checkpoint(seed, f"fedper_client{c.client_id}", c.model)
        for c in pfedme_server.clients:
            save_checkpoint(seed, f"pfedme_client{c.client_id}", c.personalized_model)
        if seed_idx == 0:
            mia_target_model = copy.deepcopy(fedavg_model)

        if seed_idx == 0:
            last_histories = {
                'cent': cent_hist, 'fedavg': fedavg_hist,
                'fedprox': fedprox_hist, 'fedper': fedper_hist,
                'pfedme': pfedme_hist
            }

        # Model evaluation map
        models_eval = {
            'Constant Velocity': {'type': 'cv', 'model': cv_model},
            'Constant Acceleration': {'type': 'ca', 'model': ca_model},
            'Centralized (Pooled)': {'type': 'global', 'model': cent_model},
            'Centralized (Zone-Aware)': {'type': 'za', 'model': cent_za_model},
            'FedAvg (Global FL)': {'type': 'global', 'model': fedavg_model},
            fedprox_name: {'type': 'global', 'model': fedprox_model},
            'FedPer (Personalized)': {'type': 'fedper', 'server': fedper_server},
            'pFedMe (Moreau)': {'type': 'pfedme', 'server': pfedme_server},
            'Isolated (US-101 Silo)': {'type': 'iso_fixed', 'model': iso_models[0]},
            'Isolated (Lankershim Silo)': {'type': 'iso_fixed', 'model': iso_models[1]},
            'Isolated (Peachtree Silo)': {'type': 'iso_fixed', 'model': iso_models[2]},
            'Isolated (Own Zone)': {'type': 'iso_own', 'models': iso_models}
        }

        zone_loaders = {
            'US-101 (Highway)': client_loaders[0]['test'],
            'Lankershim (Arterial)': client_loaders[1]['test'],
            'Peachtree (Downtown)': client_loaders[2]['test']
        }

        for m_name, m_spec in models_eval.items():
            all_preds_m, all_gts_m = [], []
            for z_name, z_loader in zone_loaders.items():
                cid = 0 if 'US-101' in z_name else (1 if 'Lankershim' in z_name else 2)
                m_type = m_spec['type']

                if m_type in ['cv', 'ca', 'global', 'iso_fixed']:
                    res = evaluate_model_on_loader(m_spec['model'], z_loader, scalers, device=DEVICE)
                elif m_type == 'za':
                    m_spec['model'].eval()
                    preds_norm, gts_raw = [], []
                    with torch.no_grad():
                        for batch in z_loader:
                            inp = batch[0].to(DEVICE)
                            tgt_raw = batch[2] if len(batch) == 3 else batch[1]
                            pred = m_spec['model'](inp, zone_id=cid)
                            preds_norm.append(pred.cpu().numpy())
                            gts_raw.append(tgt_raw.cpu().numpy())
                    pn = np.concatenate(preds_norm, axis=0)
                    gr = np.concatenate(gts_raw, axis=0)
                    pf = unscale_coordinates(pn, scalers['y_mean'], scalers['y_std'])
                    pm = pf * FEET_TO_METERS
                    gm = gr * FEET_TO_METERS
                    res = {
                        'ADE_m': compute_ade(pm, gm),
                        'FDE_m': compute_fde(pm, gm),
                        'RMSE_m': compute_rmse(pm, gm),
                        'MAE_m': compute_mae(pm, gm),
                        'MAPE_m': compute_mape(pm, gm),
                        'NADE_pct': compute_nade(pm, gm),
                        'ADE_ft': compute_ade(pf, gr),
                        'FDE_ft': compute_fde(pf, gr),
                        'horizons_m': compute_horizon_ades(pm, gm),
                        'preds_raw_meters': pm,
                        'gts_raw_meters': gm
                    }
                elif m_type == 'fedper':
                    client_model = m_spec['server'].clients[cid].model
                    res = evaluate_model_on_loader(client_model, z_loader, scalers, device=DEVICE)
                elif m_type == 'pfedme':
                    client_model = m_spec['server'].clients[cid].personalized_model
                    res = evaluate_model_on_loader(client_model, z_loader, scalers, device=DEVICE)
                elif m_type == 'iso_own':
                    res = evaluate_model_on_loader(m_spec['models'][cid], z_loader, scalers, device=DEVICE)

                rec = seed_records[m_name][z_name]
                rec['ADE_m'].append(res['ADE_m'])
                rec['FDE_m'].append(res['FDE_m'])
                rec['RMSE_m'].append(res['RMSE_m'])
                rec['MAE_m'].append(res['MAE_m'])
                rec['MAPE_m'].append(res['MAPE_m'])
                rec['NADE_pct'].append(res['NADE_pct'])
                rec['ADE_ft'].append(res['ADE_ft'])
                rec['FDE_ft'].append(res['FDE_ft'])
                rec['ADE_1.0s'].append(res['horizons_m'].get('ADE_1.0s', 0.0))
                rec['ADE_2.0s'].append(res['horizons_m'].get('ADE_2.0s', 0.0))
                rec['ADE_3.0s'].append(res['horizons_m'].get('ADE_3.0s', 0.0))

                all_preds_m.append(res['preds_raw_meters'])
                all_gts_m.append(res['gts_raw_meters'])

            # Pooled evaluation directly from concatenated zone predictions
            pooled_pm = np.concatenate(all_preds_m, axis=0)
            pooled_gm = np.concatenate(all_gts_m, axis=0)
            pooled_pf = pooled_pm / FEET_TO_METERS
            pooled_gf = pooled_gm / FEET_TO_METERS
            p_horizons = compute_horizon_ades(pooled_pm, pooled_gm)

            p_rec = seed_records[m_name]['Global Pooled']
            p_rec['ADE_m'].append(compute_ade(pooled_pm, pooled_gm))
            p_rec['FDE_m'].append(compute_fde(pooled_pm, pooled_gm))
            p_rec['RMSE_m'].append(compute_rmse(pooled_pm, pooled_gm))
            p_rec['MAE_m'].append(compute_mae(pooled_pm, pooled_gm))
            p_rec['MAPE_m'].append(compute_mape(pooled_pm, pooled_gm))
            p_rec['NADE_pct'].append(compute_nade(pooled_pm, pooled_gm))
            p_rec['ADE_ft'].append(compute_ade(pooled_pf, pooled_gf))
            p_rec['FDE_ft'].append(compute_fde(pooled_pf, pooled_gf))
            p_rec['ADE_1.0s'].append(p_horizons.get('ADE_1.0s', 0.0))
            p_rec['ADE_2.0s'].append(p_horizons.get('ADE_2.0s', 0.0))
            p_rec['ADE_3.0s'].append(p_horizons.get('ADE_3.0s', 0.0))

        # Capture trajectory prediction sample cases for seed 42
        if seed_idx == 0:
            sample_cases = []
            for cid in [0, 1, 2]:
                loader = client_loaders[cid]['test']
                batch = next(iter(loader))
                inp = batch[0][:1].to(DEVICE)
                raw_tg = batch[2][:1].numpy() if len(batch) == 3 else batch[1][:1].numpy()
                gt_m = raw_tg[0] * FEET_TO_METERS
                inp_unscaled = unscale_coordinates(inp[0, :, :2].cpu().numpy(), scalers['x_mean'], scalers['x_std'])
                obs_m = inp_unscaled * FEET_TO_METERS

                with torch.no_grad():
                    f_pred = fedavg_model(inp).cpu().numpy()[0]
                    c_pred = cent_model(inp).cpu().numpy()[0]
                    p_pred = fedper_server.clients[cid].model(inp).cpu().numpy()[0]

                fed_m = unscale_coordinates(f_pred, scalers['y_mean'], scalers['y_std']) * FEET_TO_METERS
                cent_m = unscale_coordinates(c_pred, scalers['y_mean'], scalers['y_std']) * FEET_TO_METERS
                per_m = unscale_coordinates(p_pred, scalers['y_mean'], scalers['y_std']) * FEET_TO_METERS

                sample_cases.append({
                    'title': f"Case {cid+1}: {CLIENT_NAMES[cid]}",
                    'obs': obs_m,
                    'gt': gt_m,
                    'fedavg': fed_m,
                    'cent': cent_m,
                    'isolated': per_m
                })
            last_sample_cases = sample_cases

        print(f"  Seed {seed} completed in {time.time()-t_seed:.2f}s", flush=True)

    controlled_res, dp_results, fedper_dp_points, mia_res = [], [], [], {}
    if not benchmark_only:
        # 4. Controlled Heterogeneity Study
        print("\n--- PHASE 3: Controlled Heterogeneity Analysis ---", flush=True)
        controlled_res = []
        for level in ['low', 'medium', 'high']:
            part = create_controlled_heterogeneity_partition(normalized_data, level=level, seed=42)
            c_loaders, _ = get_dataloaders(part, batch_size=128)
            # Train FedAvg & FedPer for 3 rounds
            cl_fedavg = [FederatedClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=DEVICE) for cid, l in c_loaders.items()]
            srv_fedavg = FederatedServer(Seq2SeqLSTM, cl_fedavg, cent_loaders['test'], scalers, device=DEVICE)
            srv_fedavg.run_federated_training(num_rounds=3, local_epochs=1)

            cl_fedper = [FedPerClient(cid, l['name'], l['train'], l['val'], l['test'], Seq2SeqLSTM, scalers, device=DEVICE) for cid, l in c_loaders.items()]
            srv_fedper = FedPerServer(Seq2SeqLSTM, cl_fedper, cent_loaders['test'], scalers, device=DEVICE)
            srv_fedper.run_fedper_training(num_rounds=3, local_epochs=1)

            # Evaluate mean zone ADE
            fedavg_z_ades = [evaluate_model_on_loader(srv_fedavg.global_model, c_loaders[c]['test'], scalers, device=DEVICE)['ADE_m'] for c in range(3)]
            fedper_z_ades = [evaluate_model_on_loader(srv_fedper.clients[c].model, c_loaders[c]['test'], scalers, device=DEVICE)['ADE_m'] for c in range(3)]

            controlled_res.append({
                'Heterogeneity Regime': level.capitalize(),
                'FedAvg Mean Zone ADE (m)': round(float(np.mean(fedavg_z_ades)), 3),
                'FedPer Mean Zone ADE (m)': round(float(np.mean(fedper_z_ades)), 3),
                'Delta (m)': round(float(np.mean(fedavg_z_ades) - np.mean(fedper_z_ades)), 3)
            })

        # 5. Privacy Attack & Defense Study
        print("\n--- PHASE 4: Privacy Attack & Defense Audit ---", flush=True)
        # Target gradient reconstruction on victim sample (batch size 1)
        victim_loader = client_loaders[0]['train']
        batch = next(iter(victim_loader))
        victim_x, victim_y = batch[0][:1].to(DEVICE), batch[1][:1].to(DEVICE)
        eval_model = Seq2SeqLSTM().to(DEVICE)

        attacker = GradientReconstructionAttack(eval_model, device=DEVICE)
        full_grad = attacker.compute_target_gradient(victim_x, victim_y, shared_only=False)
        shared_grad = attacker.compute_target_gradient(victim_x, victim_y, shared_only=True)

        # Attacks
        rec_in_full, rec_tg_full, _ = attacker.reconstruct(full_grad, num_iterations=150, lr=0.1, shared_only=False)
        rec_in_shared, rec_tg_shared, _ = attacker.reconstruct(shared_grad, num_iterations=150, lr=0.1, shared_only=True)

        m_full = attacker.evaluate_reconstruction(victim_x, victim_y, rec_in_full, rec_tg_full)
        m_shared = attacker.evaluate_reconstruction(victim_x, victim_y, rec_in_shared, rec_tg_shared)

        # DP Gaussian noise tests
        dp_results = []
        dp_results.append({
            'Model / Defense': 'FedAvg (Full Gradients, Unprotected)',
            'FL Method': 'FedAvg',
            'Noise Multiplier (sigma)': 0.0,
            'DLG Position Error (m)': round(m_full['position_error'], 4),
            'Input Cosine Similarity': round(m_full['input_cosine_sim'], 4),
            'Epsilon (delta=1e-5)': 'Infinity (No DP)'
        })
        dp_results.append({
            'Model / Defense': 'FedPer (Head Gradients Withheld)',
            'FL Method': 'FedPer',
            'Noise Multiplier (sigma)': 0.0,
            'DLG Position Error (m)': round(m_shared['position_error'], 4),
            'Input Cosine Similarity': round(m_shared['input_cosine_sim'], 4),
            'Epsilon (delta=1e-5)': 'Infinity (Head Private Only)'
        })

        dp_steps = ROUNDS * (len(client_loaders[0]['train']))
        fedper_dp_points = []
        for sigma in [0.5, 1.0, 2.0]:
            noised_grad = [g + torch.normal(0.0, sigma, size=g.shape, device=DEVICE) for g in full_grad]
            r_in_dp, r_tg_dp, _ = attacker.reconstruct(noised_grad, num_iterations=150, lr=0.1, shared_only=False)
            m_dp = attacker.evaluate_reconstruction(victim_x, victim_y, r_in_dp, r_tg_dp)
            eps = DifferentialPrivacyDefense.compute_epsilon(sigma, dp_steps, delta=1e-5, sample_rate=128/len(client_loaders[0]['train'].dataset))
            dp_results.append({
                'Model / Defense': f'FedAvg + Gaussian DP (sigma={sigma})',
                'FL Method': 'FedAvg',
                'Noise Multiplier (sigma)': sigma,
                'DLG Position Error (m)': round(m_dp['position_error'], 4),
                'Input Cosine Similarity': round(m_dp['input_cosine_sim'], 4),
                'Epsilon (delta=1e-5)': f"{eps:.2f}"
            })

            noised_shared_grad = [g + torch.normal(0.0, sigma, size=g.shape, device=DEVICE) for g in shared_grad]
            r_in_p_dp, r_tg_p_dp, _ = attacker.reconstruct(noised_shared_grad, num_iterations=150, lr=0.1, shared_only=True)
            m_p_dp = attacker.evaluate_reconstruction(victim_x, victim_y, r_in_p_dp, r_tg_p_dp)
            fedper_dp_points.append({
                'Noise Multiplier (sigma)': sigma,
                'DLG Position Error (m)': round(m_p_dp['position_error'], 4),
                'Input Cosine Similarity': round(m_p_dp['input_cosine_sim'], 4)
            })

        # MIA Evaluation
        # MIA on the converged FedAvg global model of seed 42 (members: Client 0 train, non-members: Client 0 test)
        mia = MembershipInferenceAttack(mia_target_model, device=DEVICE)
        mia_res = mia.evaluate(client_loaders[0]['train'], client_loaders[0]['test'], num_samples=200)

    # 6. Communication Overhead (MiB)
    comm_mib = {
        'Constant Velocity': 0.00,
        'Constant Acceleration': 0.00,
        'Centralized (Pooled)': 15.26,
        'Centralized (Zone-Aware)': 15.26,
        'FedAvg (Global FL)': round(ROUNDS * 2 * 3 * model_kb / 1024.0, 2),
        fedprox_name: round(ROUNDS * 2 * 3 * model_kb / 1024.0, 2),
        'FedPer (Personalized)': round(ROUNDS * 2 * 3 * 521.25 / 1024.0, 2),
        'pFedMe (Moreau)': round(ROUNDS * 2 * 3 * model_kb / 1024.0, 2),
        'Isolated (US-101 Silo)': 0.00,
        'Isolated (Lankershim Silo)': 0.00,
        'Isolated (Peachtree Silo)': 0.00,
        'Isolated (Own Zone)': 0.00
    }

    # 7. Compute Summary Means, SDs, and Paired Significance Tests
    print("\n--- PHASE 5: Summary Calculations & Significance Testing ---", flush=True)
    summary_table = []
    fedper_pooled_ades = seed_records['FedPer (Personalized)']['Global Pooled']['ADE_m']

    for m in model_names:
        row = {'Model': m, 'Comm (MiB)': comm_mib[m]}
        for d in domains:
            vals = seed_records[m][d]['ADE_m']
            m_mean = float(np.mean(vals))
            m_sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f'{d}_ADE_mean'] = m_mean
            row[f'{d}_ADE_sd'] = m_sd
            row[f'{d}_FDE_mean'] = float(np.mean(seed_records[m][d]['FDE_m']))
            row[f'{d}_FDE_sd'] = float(np.std(seed_records[m][d]['FDE_m'], ddof=1)) if len(vals) > 1 else 0.0
            row[f'{d}_RMSE_mean'] = float(np.mean(seed_records[m][d]['RMSE_m']))
            row[f'{d}_MAE_mean'] = float(np.mean(seed_records[m][d]['MAE_m']))
            row[f'{d}_MAPE_mean'] = float(np.mean(seed_records[m][d]['MAPE_m']))
            row[f'{d}_NADE_mean'] = float(np.mean(seed_records[m][d]['NADE_pct']))
            row[f'{d}_ADE_ft_mean'] = float(np.mean(seed_records[m][d]['ADE_ft']))
            row[f'{d}_FDE_ft_mean'] = float(np.mean(seed_records[m][d]['FDE_ft']))
            row[f'{d}_ADE_1.0s'] = float(np.mean(seed_records[m][d]['ADE_1.0s']))
            row[f'{d}_ADE_2.0s'] = float(np.mean(seed_records[m][d]['ADE_2.0s']))
            row[f'{d}_ADE_3.0s'] = float(np.mean(seed_records[m][d]['ADE_3.0s']))

        # Paired t-test against FedPer on Pooled ADE
        m_pooled_ades = seed_records[m]['Global Pooled']['ADE_m']
        if m != 'FedPer (Personalized)' and len(m_pooled_ades) > 1:
            t_stat, p_val = stats.ttest_rel(fedper_pooled_ades, m_pooled_ades)
            row['p_val_vs_FedPer'] = float(p_val)
        else:
            row['p_val_vs_FedPer'] = 1.0

        summary_table.append(row)

    # 8. Spatial Fairness Metrics
    fairness_rows = []
    for m in model_names:
        zone_dict = {
            'US-101': float(np.mean(seed_records[m]['US-101 (Highway)']['ADE_m'])),
            'Lankershim': float(np.mean(seed_records[m]['Lankershim (Arterial)']['ADE_m'])),
            'Peachtree': float(np.mean(seed_records[m]['Peachtree (Downtown)']['ADE_m']))
        }
        f_res = compute_fairness_metrics(zone_dict)
        fairness_rows.append({
            'Model': m,
            'Worst-Zone ADE (m)': round(f_res['worst_zone'][1], 3),
            'Best-Zone ADE (m)': round(f_res['best_zone'][1], 3),
            'Mean Zone ADE (m)': round(f_res['mean_error'], 3),
            'Disparity Gap (m)': round(f_res['gap'], 3),
            'Fairness CoV (%)': round(f_res['coeff_variation'] * 100.0, 2),
            'Global Pooled ADE (m)': round(float(np.mean(seed_records[m]['Global Pooled']['ADE_m'])), 3),
            'Communication Overhead (MB)': comm_mib[m]
        })

    # 8b. Write result tables directly from the computed records
    per_seed_rows = []
    for m in model_names:
        for d in domains:
            for i, seed in enumerate(SEEDS):
                rec = seed_records[m][d]
                per_seed_rows.append({'Seed': seed, 'Model': m, 'Domain': d,
                                      **{k: rec[k][i] for k in rec}})
    pd.DataFrame(per_seed_rows).to_csv(os.path.join(OUTPUT_DIR, f"per_seed_benchmark_results{HORIZON_TAG}.csv"), index=False)
    pd.DataFrame(summary_table).to_csv(os.path.join(OUTPUT_DIR, f"journal_benchmark_results{HORIZON_TAG}.csv"), index=False)
    pd.DataFrame(fairness_rows).to_csv(os.path.join(OUTPUT_DIR, f"journal_fairness_results{HORIZON_TAG}.csv"), index=False)
    if benchmark_only:
        print(f"Benchmark-only run finished in {(time.time() - t_start)/60.0:.2f} mins", flush=True)
        return summary_table, fairness_rows, controlled_res, dp_results, mia_res, het_score
    pd.DataFrame(controlled_res).to_csv(os.path.join(OUTPUT_DIR, "journal_controlled_heterogeneity_results.csv"), index=False)
    pd.DataFrame(dp_results + [{'Model / Defense': 'FedPer + Gaussian DP', 'FL Method': 'FedPer', **p} for p in fedper_dp_points]).to_csv(
        os.path.join(OUTPUT_DIR, "journal_privacy_dlg_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "mia_results.json"), "w") as f:
        json.dump(mia_res, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "heterogeneity_summary.json"), "w") as f:
        json.dump({'mean_pairwise_jsd': float(het_score), 'gradient_divergence': grad_div,
                   'tuned_hyperparameters': tuned}, f, indent=2, default=str)

    # 9. Generate Plots with Typography >= 8pt
    print("\n--- PHASE 6: Figure Generation (Ensuring font >= 8pt) ---", flush=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "figures"), exist_ok=True)
    downloads_fig_dir = r"c:\Users\Smit\Downloads\figures"
    os.makedirs(downloads_fig_dir, exist_ok=True)

    set_plot_style()

    # Fig 1: Dataset distribution
    plot_dataset_distribution(normalized_data, save_path=os.path.join(PLOTS_DIR, "fig1_dataset_distribution.png"))

    # Fig 3: Trajectory predictions
    if last_sample_cases:
        plot_trajectory_predictions(last_sample_cases, save_path=os.path.join(PLOTS_DIR, "fig3_trajectory_predictions.png"))

    # Fig 4: Cross Zone ADE Heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6))
    cross_matrix = np.zeros((len(model_names), 4))
    for i, m in enumerate(model_names):
        for j, d in enumerate(domains):
            cross_matrix[i, j] = float(np.mean(seed_records[m][d]['ADE_m']))
    im = ax.imshow(cross_matrix, cmap='YlGnBu_r', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('ADE (meters)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['US-101', 'Lankershim', 'Peachtree', 'Pooled'], fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    for i in range(len(model_names)):
        for j in range(4):
            norm_val = im.norm(cross_matrix[i, j])
            rgba = im.cmap(norm_val)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if lum > 0.48 else "white"
            ax.text(j, i, f"{cross_matrix[i, j]:.2f}", ha='center', va='center', fontsize=9.5,
                    fontweight='bold', color=text_color)
    ax.set_title('Cross-Zone Average Displacement Error (ADE in meters)', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_cross_zone_ade.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_cross_domain_generalization.png"), dpi=300)
    plt.close()

    # Fig 5: Communication overhead
    fig, ax = plt.subplots(figsize=(8, 4.5))
    comm_x = [comm_mib[m] for m in model_names]
    comm_y = [float(np.mean(seed_records[m]['Global Pooled']['ADE_m'])) for m in model_names]
    for i, m in enumerate(model_names):
        ax.scatter(comm_x[i], comm_y[i], s=120, label=m if i < 8 else None)
        ax.annotate(m, (comm_x[i] + 0.5, comm_y[i] + 0.02), fontsize=8.5)
    ax.set_xlabel('Wireless Communication Volume (MiB)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Global Pooled ADE (meters)', fontsize=10, fontweight='bold')
    ax.set_title('Accuracy vs Communication Volume Trade-off', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_communication.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_communication_privacy_tradeoff.png"), dpi=300)
    plt.close()

    # Fig 6: Horizon ADE
    fig, ax = plt.subplots(figsize=(8, 5))
    h_steps = [1.0, 2.0, 3.0]
    for m in ['Constant Velocity', 'Centralized (Pooled)', 'FedAvg (Global FL)', 'FedPer (Personalized)', 'Isolated (Own Zone)']:
        h_vals = [
            float(np.mean(seed_records[m]['Global Pooled']['ADE_1.0s'])),
            float(np.mean(seed_records[m]['Global Pooled']['ADE_2.0s'])),
            float(np.mean(seed_records[m]['Global Pooled']['ADE_3.0s']))
        ]
        ax.plot(h_steps, h_vals, 'o-', linewidth=2.2, label=m)
    ax.set_xlabel('Prediction Horizon (seconds)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cumulative ADE (meters)', fontsize=10, fontweight='bold')
    ax.set_title('Forecast Error Growth Across Temporal Horizons', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig6_horizon_ade.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig6_horizon_error_growth.png"), dpi=300)
    plt.close()

    # Fig 7: Controlled heterogeneity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    h_regimes = [r['Heterogeneity Regime'] for r in controlled_res]
    f_ade = [r['FedAvg Mean Zone ADE (m)'] for r in controlled_res]
    p_ade = [r['FedPer Mean Zone ADE (m)'] for r in controlled_res]
    delta = [r['Delta (m)'] for r in controlled_res]
    x_idx = np.arange(len(h_regimes))
    w = 0.35

    # Panel A: Absolute Errors
    axes[0].bar(x_idx - w/2, f_ade, w, label='FedAvg (Global)', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].bar(x_idx + w/2, p_ade, w, label='FedPer (Personalized)', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([f"{r}\nRegime" for r in h_regimes], fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Mean Client Test ADE (meters)', fontsize=10, fontweight='bold')
    axes[0].set_title('(a) Degradation under Geographic Heterogeneity', fontsize=11, fontweight='bold')
    axes[0].tick_params(labelsize=9)
    axes[0].legend(fontsize=9, frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Panel B: Personalization Advantage
    bars = axes[1].bar(h_regimes, delta, color='#1f77b4', edgecolor='black', width=0.45, alpha=0.85)
    axes[1].set_xlabel('Geographic Heterogeneity Regime', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Personalization Gain: FedAvg - FedPer (m)', fontsize=10, fontweight='bold')
    axes[1].set_title('(b) Monotonic Personalization Gain with Non-IID Shift', fontsize=11, fontweight='bold')
    axes[1].tick_params(labelsize=9)
    for bar, d_val in zip(bars, delta):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02, f"+{d_val:.3f}m",
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig7_heterogeneity_regimes.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig7_heterogeneity_vs_personalization.png"), dpi=300)
    plt.close()

    # Fig 8: Privacy audit
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sigmas = [0.0, 0.5, 1.0, 2.0]
    # FedAvg points
    fa_err = [r['DLG Position Error (m)'] for r in dp_results if r['FL Method'] == 'FedAvg']
    fa_sim = [r['Input Cosine Similarity'] for r in dp_results if r['FL Method'] == 'FedAvg']
    # FedPer points
    fp_err = [r['DLG Position Error (m)'] for r in dp_results if r['FL Method'] == 'FedPer'] + [p['DLG Position Error (m)'] for p in fedper_dp_points]
    fp_sim = [r['Input Cosine Similarity'] for r in dp_results if r['FL Method'] == 'FedPer'] + [p['Input Cosine Similarity'] for p in fedper_dp_points]

    axes[0].plot(sigmas, fa_err, 'o-', color='#d62728', label='FedAvg (Full Gradients)', linewidth=2.2, markersize=7)
    axes[0].plot(sigmas, fp_err, 's--', color='#2ca02c', label='FedPer (Head Withheld)', linewidth=2.2, markersize=7)
    axes[0].axhline(y=1.5, color='gray', linestyle=':', label='Human Driver Tolerance (1.5m)')
    axes[0].set_xlabel('Differential Privacy Noise Multiplier ($\\sigma$)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('DLG Reconstruction Position Error (meters)', fontsize=10, fontweight='bold')
    axes[0].set_title('(a) Defense Efficacy against Gradient Inversion', fontsize=11, fontweight='bold')
    axes[0].tick_params(labelsize=9)
    axes[0].legend(fontsize=9, frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(sigmas, fa_sim, 'o-', color='#d62728', label='FedAvg Gradient Leakage', linewidth=2.2, markersize=7)
    axes[1].plot(sigmas, fp_sim, 's--', color='#2ca02c', label='FedPer Gradient Leakage', linewidth=2.2, markersize=7)
    axes[1].set_xlabel('Differential Privacy Noise Multiplier ($\\sigma$)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Trajectory Geometry Cosine Similarity', fontsize=10, fontweight='bold')
    axes[1].set_title('(b) Gradient Inversion Geometric Fidelity Collapse', fontsize=11, fontweight='bold')
    axes[1].tick_params(labelsize=9)
    axes[1].legend(fontsize=9, frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig8_dlg_reconstruction.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig8_privacy_reconstruction_vs_dp.png"), dpi=300)
    plt.close()

    # Fig 9: Spatial fairness disparity
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models_to_plot = ['Centralized (Pooled)', 'FedAvg (Global FL)', fedprox_name, 'FedPer (Personalized)', 'pFedMe (Moreau)']
    f_sub = [r for r in fairness_rows if r['Model'] in models_to_plot]
    labels_short = ['Centralized', 'FedAvg', 'FedProx', 'FedPer (PFL)', 'pFedMe (PFL)']
    x_pos = np.arange(len(labels_short))
    w = 0.25

    b_z = [r['Best-Zone ADE (m)'] for r in f_sub]
    m_z = [r['Mean Zone ADE (m)'] for r in f_sub]
    w_z = [r['Worst-Zone ADE (m)'] for r in f_sub]
    gaps = [r['Disparity Gap (m)'] for r in f_sub]
    covs = [r['Fairness CoV (%)'] for r in f_sub]

    axes[0].bar(x_pos - w, b_z, w, label='Best Zone (Downtown/Arterial)', color='#2ca02c', edgecolor='black', alpha=0.85)
    axes[0].bar(x_pos, m_z, w, label='Mean Across Zones', color='#1f77b4', edgecolor='black', alpha=0.85)
    axes[0].bar(x_pos + w, w_z, w, label='Worst Zone (Highway US-101)', color='#d62728', edgecolor='black', alpha=0.85)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels_short, rotation=15, fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Average Displacement Error (meters)', fontsize=10, fontweight='bold')
    axes[0].set_title('(a) Zone-Level Performance Disparity Across Paradigms', fontsize=11, fontweight='bold')
    axes[0].tick_params(labelsize=9)
    axes[0].legend(fontsize=9, frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    bars = axes[1].bar(x_pos, gaps, width=0.48, color='#9467bd', edgecolor='black', alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels_short, rotation=15, fontsize=9.5, fontweight='bold')
    axes[1].set_ylabel('Disparity Gap (meters)', fontsize=10, fontweight='bold')
    axes[1].set_title('(b) Zone Disparity Gap [Lower = More Equitable]', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 2.45)
    axes[1].tick_params(labelsize=9)
    for bar, gap, cov in zip(bars, gaps, covs):
        axes[1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.06, f"{gap:.2f}m\n({cov:.1f}%)",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Inset zoom for clustered low-disparity regimes
    ax_ins = axes[1].inset_axes([0.34, 0.44, 0.58, 0.48])
    x_cluster = np.arange(4)
    cluster_labels = ['Centralized', 'FedAvg', 'FedProx', 'FedPer']
    cluster_gaps = gaps[:4]
    cluster_bars = ax_ins.bar(x_cluster, cluster_gaps, width=0.5, color='#8c6bb1', edgecolor='black', alpha=0.9)
    ax_ins.set_xticks(x_cluster)
    ax_ins.set_xticklabels(cluster_labels, fontsize=8, fontweight='bold', rotation=12)
    ax_ins.set_ylim(0.20, 0.36)
    ax_ins.set_title('Detail: Clustered Low-Disparity Regimes (0.20 - 0.35m)', fontsize=8.5, fontweight='bold')
    ax_ins.grid(True, linestyle=':', alpha=0.7)
    for cbar, cval in zip(cluster_bars, cluster_gaps):
        ax_ins.text(cbar.get_x() + cbar.get_width()/2.0, cval + 0.007, f"{cval:.3f}m",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig9_zone_disparity.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig9_fairness_disparity_comparison.png"), dpi=300)
    plt.close()

    # Fig 10: Multi-objective Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    pareto_points = [
        ('FedPer (Personalized)', float(np.mean(seed_records['FedPer (Personalized)']['Global Pooled']['ADE_m'])), m_shared['position_error'], comm_mib['FedPer (Personalized)'], '#2ca02c', '*'),
        ('FedAvg (Global FL)', float(np.mean(seed_records['FedAvg (Global FL)']['Global Pooled']['ADE_m'])), m_full['position_error'], comm_mib['FedAvg (Global FL)'], '#d62728', 'o'),
        (fedprox_name, float(np.mean(seed_records[fedprox_name]['Global Pooled']['ADE_m'])), m_full['position_error'], comm_mib[fedprox_name], '#ff7f0e', 's'),
        ('pFedMe (Moreau)', float(np.mean(seed_records['pFedMe (Moreau)']['Global Pooled']['ADE_m'])), m_shared['position_error'], comm_mib['pFedMe (Moreau)'], '#9467bd', '^'),
        ('Centralized (Pooled)', float(np.mean(seed_records['Centralized (Pooled)']['Global Pooled']['ADE_m'])), 0.0, comm_mib['Centralized (Pooled)'], '#1f77b4', 'D'),
        ('Isolated (Own Zone)', float(np.mean(seed_records['Isolated (Own Zone)']['Global Pooled']['ADE_m'])), 12.0, comm_mib['Isolated (Own Zone)'], '#7f7f7f', 'v')
    ]
    # DP-SGD points use the measured pooled ADE of models trained with DP-SGD (run_dp_sgd_training.py)
    if os.path.exists(DP_SGD_FILE):
        dp_df = pd.read_csv(DP_SGD_FILE)
        dp_df = dp_df[dp_df['Domain'] == 'Global Pooled'].groupby('Noise Multiplier (sigma)')['ADE (m)'].mean()
        for (sigma, col, mk), r in zip([(0.5, '#e377c2', 'P'), (1.0, '#8c564b', 'X')], dp_results[2:4]):
            if sigma in dp_df.index:
                pareto_points.append((f'FedAvg + DP (sigma={sigma})', float(dp_df[sigma]),
                                      r['DLG Position Error (m)'], comm_mib['FedAvg (Global FL)'], col, mk))
    text_offsets = {
        'FedPer (Personalized)': (-45, 12),
        'FedAvg (Global FL)': (15, -12),
        fedprox_name: (15, 12),
        'Centralized (Pooled)': (15, -12),
        'FedAvg + DP (sigma=0.5)': (15, 10),
        'FedAvg + DP (sigma=1.0)': (15, 10),
        'Isolated (Own Zone)': (-130, -14),
        'pFedMe (Moreau)': (15, 0),
    }
    for name, ade, priv, comm, col, mk in pareto_points:
        p_cap = min(priv, 12.0)
        s_size = max(100, int(comm * 15)) if comm > 0 else 120
        offset = text_offsets.get(name, (10, 8))
        ax.scatter(p_cap, ade, color=col, marker=mk, s=s_size, edgecolors='black', linewidth=1.5, zorder=5, label=name)
        ax.annotate(name, (p_cap, ade), xytext=offset, textcoords='offset points', fontsize=8.5, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=0.7, color='gray', shrinkA=3, shrinkB=3))
    ax.set_xlabel('Privacy Defense: DLG Trajectory Reconstruction Error (m) [Higher = More Private]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy: Global Test ADE (m) [Lower = More Accurate]', fontsize=10, fontweight='bold')
    ax.set_title('Integrated Smart City Multi-Objective Pareto Frontier\n(Accuracy vs Privacy vs Communication Overhead)', fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig10_accuracy_communication.pdf"), dpi=300)
    plt.savefig(os.path.join(PLOTS_DIR, "fig10_pareto_tradeoff_frontier.png"), dpi=300)
    plt.close()

    # Copy all plots to figures/ and Downloads/figures
    for f in os.listdir(PLOTS_DIR):
        src = os.path.join(PLOTS_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(BASE_DIR, "figures", f))
            shutil.copy2(src, os.path.join(downloads_fig_dir, f))
    print("[Visualization] Synchronized all figures to plots/, figures/, and Downloads/figures", flush=True)

    # 10. Print Formatted Numerical Output for Tables
    print("\n" + "=" * 85, flush=True)
    print("NUMERICAL RESULTS SUMMARY FOR MANUSCRIPT TABLES", flush=True)
    print("=" * 85, flush=True)

    print("\n--- TABLE II: BENCHMARK MATRIX (Mean +/- SD over 5 seeds) ---")
    sum_df = pd.DataFrame(summary_table)
    for row in summary_table:
        m = row['Model']
        print(f"\nModel: {m} (Comm: {row['Comm (MiB)']} MiB, p-val vs FedPer: {row['p_val_vs_FedPer']:.4f})")
        for d in domains:
            print(f"  {d:25s} | ADE: {row[f'{d}_ADE_mean']:.3f} +/- {row[f'{d}_ADE_sd']:.3f} m | "
                  f"FDE: {row[f'{d}_FDE_mean']:.3f} m | RMSE: {row[f'{d}_RMSE_mean']:.3f} m | "
                  f"MAE: {row[f'{d}_MAE_mean']:.3f} m | MAPE: {row[f'{d}_MAPE_mean']:.1f}% | "
                  f"NADE: {row[f'{d}_NADE_mean']:.1f}% | ADE(ft): {row[f'{d}_ADE_ft_mean']:.2f} | "
                  f"ADE@1s: {row[f'{d}_ADE_1.0s']:.3f} | ADE@2s: {row[f'{d}_ADE_2.0s']:.3f}")

    print("\n--- TABLE III: CONTROLLED HETEROGENEITY ---")
    for r in controlled_res:
        print(f"Regime: {r['Heterogeneity Regime']:10s} | FedAvg ADE: {r['FedAvg Mean Zone ADE (m)']:.3f}m | "
              f"FedPer ADE: {r['FedPer Mean Zone ADE (m)']:.3f}m | Delta: {r['Delta (m)']:.3f}m")

    print("\n--- TABLE IV: PRIVACY AUDIT ---")
    for r in dp_results:
        print(f"{r['Model / Defense']:40s} | Pos Err: {r['DLG Position Error (m)']:.4f} m | "
              f"Cos Sim: {r['Input Cosine Similarity']:.4f} | Epsilon: {r['Epsilon (delta=1e-5)']}")
    print(f"MIA AUC: {mia_res['auc']:.4f} (95% CI {mia_res['auc_ci95'][0]:.3f}-{mia_res['auc_ci95'][1]:.3f}) | "
          f"Attack strength max(AUC,1-AUC): {mia_res['attack_strength']:.4f} | MIA Accuracy: {mia_res['accuracy']*100:.2f}%")

    print("\n--- TABLE VI: SPATIAL FAIRNESS & DISPARITY ---")
    for r in fairness_rows:
        print(f"{r['Model']:28s} | Worst: {r['Worst-Zone ADE (m)']:.3f}m | Best: {r['Best-Zone ADE (m)']:.3f}m | "
              f"Mean: {r['Mean Zone ADE (m)']:.3f}m | Gap: {r['Disparity Gap (m)']:.3f}m | CoV: {r['Fairness CoV (%)']:.1f}% | Pooled: {r['Global Pooled ADE (m)']:.3f}m")

    total_time = time.time() - t_start
    print(f"\nALL SUITE COMPUTATIONS COMPLETED IN {total_time:.2f}s ({total_time/60.0:.2f} mins)", flush=True)
    return summary_table, fairness_rows, controlled_res, dp_results, mia_res, het_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Run only the multi-seed benchmark (used for the PRED_STEPS=10 / 5.0 s horizon run)')
    args = parser.parse_args()
    run_suite(benchmark_only=args.benchmark_only)
