"""
Data Preprocessing Pipeline for NGSIM Vehicle Trajectory Dataset.
Features:
- Fast chunk-based parsing and sanitization of raw CSV
- Trajectory extraction with chronological frame ordering
- 2 Hz Downsampling and Sliding Window Sequence Generation
- Translation-invariant coordinate transformation
- Standard scaling of kinematic features
- Vehicle-level Train / Val / Test splitting
- Non-IID Federated Client Partitioning (US-101, Lankershim, Peachtree)
- Fast Caching with .npz files for instant re-loads
- PyTorch Dataset & DataLoader utilities
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_CSV, CACHE_DIR, OBS_STEPS, PRED_STEPS, TOTAL_STEPS,
    DOWNSAMPLE_STEP, STRIDE, RANDOM_SEED, BATCH_SIZE, MAX_VEHICLES_PER_LOC,
    MAX_RAW_ROWS, HORIZON_TAG
)

class TrajectoryDataset(Dataset):
    """PyTorch Dataset for Vehicle Trajectory Sequences."""
    def __init__(self, inputs, targets, raw_targets=None):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.raw_targets = raw_targets  # Unscaled targets in feet for exact physical ADE/FDE

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.raw_targets is not None:
            return self.inputs[idx], self.targets[idx], torch.tensor(self.raw_targets[idx], dtype=torch.float32)
        return self.inputs[idx], self.targets[idx]


class NGSIMPreprocessor:
    """End-to-end Preprocessor for NGSIM Trajectory Data."""
    def __init__(self, csv_path=DATA_CSV, cache_dir=CACHE_DIR):
        self.csv_path = csv_path
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"federated_trajectory_data{HORIZON_TAG}.npz")
        self.scalers = {}

    def extract_sequences_from_csv(self, max_vehicles_per_loc=MAX_VEHICLES_PER_LOC, max_raw_rows=MAX_RAW_ROWS):
        """
        Reads CSV, sanitizes columns, extracts trajectories from a 1 Million row slice,
        downsamples, and generates sliding window observation/prediction pairs.
        """
        print(f"[Preprocessor] Reading raw data from {self.csv_path} (limit: {max_raw_rows:,} rows)...", flush=True)
        t0 = time.time()
        
        cols = ['Vehicle_ID', 'Frame_ID', 'Global_Time', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Location']
        loc_chunks = {'us-101': [], 'lankershim': [], 'peachtree': []}
        loc_counts = {'us-101': 0, 'lankershim': 0, 'peachtree': 0}
        target_per_loc = max_raw_rows // 3  # Balanced slice across 3 Smart City RSU domains
        
        for chunk in pd.read_csv(self.csv_path, usecols=cols, chunksize=500_000, low_memory=False):
            for col in ['Vehicle_ID', 'Frame_ID', 'Global_Time', 'Local_X', 'Local_Y', 'v_Vel', 'v_Acc']:
                if not pd.api.types.is_numeric_dtype(chunk[col]):
                    chunk[col] = pd.to_numeric(chunk[col].astype(str).str.replace(',', ''), errors='coerce')
            chunk = chunk.dropna(subset=cols)
            chunk['Location'] = chunk['Location'].astype(str).str.strip().str.lower()

            for loc_key in loc_chunks:
                if loc_counts[loc_key] < target_per_loc:
                    sub = chunk[chunk['Location'] == loc_key]
                    if not sub.empty:
                        needed = target_per_loc - loc_counts[loc_key]
                        sub_to_add = sub.iloc[:needed]
                        loc_chunks[loc_key].append(sub_to_add)
                        loc_counts[loc_key] += len(sub_to_add)

            if all(loc_counts[k] >= target_per_loc for k in loc_counts):
                break

        collected = []
        for loc_key, chunks_list in loc_chunks.items():
            if chunks_list:
                collected.append(pd.concat(chunks_list, ignore_index=True))
        
        df = pd.concat(collected, ignore_index=True)
        print(f"[Preprocessor] Ingested exactly {len(df):,} rows ({loc_counts}) in {time.time()-t0:.2f}s", flush=True)

        loc_order = ['us-101', 'lankershim', 'peachtree']
        loc_map = {
            'us-101': 0,
            'lankershim': 1,
            'peachtree': 2
        }

        client_data = {}
        min_raw_frames = TOTAL_STEPS * DOWNSAMPLE_STEP

        for loc_name in loc_order:
            if loc_name not in df['Location'].values:
                continue
            cid = loc_map[loc_name]
            loc_df = df[df['Location'] == loc_name].copy()
            
            # Sort by Vehicle_ID, Global_Time, Frame_ID
            loc_df = loc_df.drop_duplicates(subset=['Vehicle_ID', 'Global_Time', 'Frame_ID']).sort_values(['Vehicle_ID', 'Global_Time', 'Frame_ID'])
            
            # Contiguous segment identification:
            # Segment break happens if Vehicle_ID changes, Frame_ID != prev+1, or Global_Time != prev+100
            v_change = loc_df['Vehicle_ID'].values[1:] != loc_df['Vehicle_ID'].values[:-1]
            f_jump = loc_df['Frame_ID'].values[1:] != (loc_df['Frame_ID'].values[:-1] + 1)
            g_jump = loc_df['Global_Time'].values[1:] != (loc_df['Global_Time'].values[:-1] + 100)
            breaks = np.concatenate([[True], v_change | f_jump | g_jump])
            loc_df['segment_id'] = np.cumsum(breaks)
            
            # Keep only segments with length >= TOTAL_STEPS * DOWNSAMPLE_STEP (e.g. 60 frames = 6s at 10 Hz)
            seg_sizes = loc_df['segment_id'].value_counts()
            valid_segs = set(seg_sizes[seg_sizes >= min_raw_frames].index)
            loc_df_valid = loc_df[loc_df['segment_id'].isin(valid_segs)]
            
            # Disjoint vehicle entity split (0% leakage)
            unique_vids = loc_df_valid['Vehicle_ID'].unique()
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(unique_vids)
            if max_vehicles_per_loc and len(unique_vids) > max_vehicles_per_loc:
                unique_vids = unique_vids[:max_vehicles_per_loc]
                loc_df_valid = loc_df_valid[loc_df_valid['Vehicle_ID'].isin(unique_vids)]

            print(f"[Preprocessor] Processing Location '{loc_name}' (Client {cid}) with {len(unique_vids)} vehicles...", flush=True)
            
            n_v = len(unique_vids)
            n_train = int(0.70 * n_v)
            n_val = int(0.15 * n_v)
            
            train_vids = set(unique_vids[:n_train])
            val_vids = set(unique_vids[n_train:n_train+n_val])
            test_vids = set(unique_vids[n_train+n_val:])

            splits = {'train': ([], []), 'val': ([], []), 'test': ([], [])}

            for sid, grp in loc_df_valid.groupby('segment_id'):
                vid = grp['Vehicle_ID'].iloc[0]
                if vid in train_vids:
                    split_key = 'train'
                elif vid in val_vids:
                    split_key = 'val'
                elif vid in test_vids:
                    split_key = 'test'
                else:
                    continue

                down = grp.iloc[::DOWNSAMPLE_STEP]
                if len(down) < TOTAL_STEPS:
                    continue

                coords = down[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc']].values.astype(np.float32)

                for i in range(0, len(coords) - TOTAL_STEPS + 1, STRIDE):
                    window = coords[i : i + TOTAL_STEPS]
                    orig_x = window[OBS_STEPS - 1, 0]
                    orig_y = window[OBS_STEPS - 1, 1]

                    rel_window = window.copy()
                    rel_window[:, 0] -= orig_x
                    rel_window[:, 1] -= orig_y

                    inp_seq = rel_window[:OBS_STEPS]
                    tgt_seq = rel_window[OBS_STEPS:, :2]

                    splits[split_key][0].append(inp_seq)
                    splits[split_key][1].append(tgt_seq)

            client_data[cid] = {
                'name': loc_name,
                'train_x': np.array(splits['train'][0], dtype=np.float32),
                'train_y': np.array(splits['train'][1], dtype=np.float32),
                'val_x': np.array(splits['val'][0], dtype=np.float32),
                'val_y': np.array(splits['val'][1], dtype=np.float32),
                'test_x': np.array(splits['test'][0], dtype=np.float32),
                'test_y': np.array(splits['test'][1], dtype=np.float32),
            }
            print(f"  -> Client {cid} ({loc_name}): Train={len(client_data[cid]['train_x'])}, "
                  f"Val={len(client_data[cid]['val_x'])}, Test={len(client_data[cid]['test_x'])}", flush=True)

        return client_data

    def fit_scalers_and_normalize(self, client_data):
        """Fits global standard scalers and normalizes features."""
        print("[Preprocessor] Computing feature normalization scalers...", flush=True)
        all_train_x = np.concatenate([client_data[cid]['train_x'] for cid in client_data], axis=0)
        
        flat_x = all_train_x.reshape(-1, all_train_x.shape[-1])
        x_mean = flat_x.mean(axis=0)
        x_std = flat_x.std(axis=0)
        x_std[x_std < 1e-6] = 1.0

        all_train_y = np.concatenate([client_data[cid]['train_y'] for cid in client_data], axis=0)
        flat_y = all_train_y.reshape(-1, all_train_y.shape[-1])
        y_mean = flat_y.mean(axis=0)
        y_std = flat_y.std(axis=0)
        y_std[y_std < 1e-6] = 1.0

        self.scalers = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std
        }

        print(f"  Input Mean: {x_mean}", flush=True)
        print(f"  Input Std:  {x_std}", flush=True)
        print(f"  Target Mean: {y_mean}", flush=True)
        print(f"  Target Std:  {y_std}", flush=True)

        normalized_data = {}
        for cid in client_data:
            normalized_data[cid] = {
                'name': client_data[cid]['name'],
                'train_x': (client_data[cid]['train_x'] - x_mean) / x_std,
                'train_y': (client_data[cid]['train_y'] - y_mean) / y_std,
                'val_x': (client_data[cid]['val_x'] - x_mean) / x_std,
                'val_y': (client_data[cid]['val_y'] - y_mean) / y_std,
                'test_x': (client_data[cid]['test_x'] - x_mean) / x_std,
                'test_y': (client_data[cid]['test_y'] - y_mean) / y_std,
                'raw_train_y': client_data[cid]['train_y'],
                'raw_val_y': client_data[cid]['val_y'],
                'raw_test_y': client_data[cid]['test_y']
            }

        return normalized_data

    def save_cache(self, normalized_data):
        print(f"[Preprocessor] Saving preprocessed dataset to cache: {self.cache_file}", flush=True)
        save_dict = {
            'x_mean': self.scalers['x_mean'],
            'x_std': self.scalers['x_std'],
            'y_mean': self.scalers['y_mean'],
            'y_std': self.scalers['y_std'],
            'num_clients': len(normalized_data)
        }
        for cid, cdict in normalized_data.items():
            for key, val in cdict.items():
                if isinstance(val, np.ndarray):
                    save_dict[f"client_{cid}_{key}"] = val
                elif isinstance(val, str):
                    save_dict[f"client_{cid}_{key}"] = np.array(val)
        
        np.savez_compressed(self.cache_file, **save_dict)
        print("[Preprocessor] Cache successfully written.", flush=True)

    def load_cache(self):
        if not os.path.exists(self.cache_file):
            return None
        print(f"[Preprocessor] Loading preprocessed data from cache: {self.cache_file}", flush=True)
        data = np.load(self.cache_file, allow_pickle=True)
        self.scalers = {
            'x_mean': data['x_mean'],
            'x_std': data['x_std'],
            'y_mean': data['y_mean'],
            'y_std': data['y_std'],
        }
        num_clients = int(data['num_clients'])
        normalized_data = {}
        for cid in range(num_clients):
            name = str(data[f"client_{cid}_name"])
            normalized_data[cid] = {
                'name': name,
                'train_x': data[f"client_{cid}_train_x"],
                'train_y': data[f"client_{cid}_train_y"],
                'val_x': data[f"client_{cid}_val_x"],
                'val_y': data[f"client_{cid}_val_y"],
                'test_x': data[f"client_{cid}_test_x"],
                'test_y': data[f"client_{cid}_test_y"],
                'raw_train_y': data[f"client_{cid}_raw_train_y"],
                'raw_val_y': data[f"client_{cid}_raw_val_y"],
                'raw_test_y': data[f"client_{cid}_raw_test_y"],
            }
        print(f"[Preprocessor] Successfully loaded cached data for {num_clients} clients.", flush=True)
        return normalized_data

    def prepare_dataset(self, force_recompute=False):
        if not force_recompute:
            cached = self.load_cache()
            if cached is not None:
                return cached, self.scalers

        raw_client_data = self.extract_sequences_from_csv()
        normalized_data = self.fit_scalers_and_normalize(raw_client_data)
        self.save_cache(normalized_data)
        return normalized_data, self.scalers


def get_dataloaders(normalized_data, batch_size=BATCH_SIZE):
    client_loaders = {}
    
    all_train_x, all_train_y, all_raw_train_y = [], [], []
    all_val_x, all_val_y, all_raw_val_y = [], [], []
    all_test_x, all_test_y, all_raw_test_y = [], [], []

    for cid in sorted(normalized_data.keys()):
        cdict = normalized_data[cid]
        train_ds = TrajectoryDataset(cdict['train_x'], cdict['train_y'], cdict['raw_train_y'])
        val_ds = TrajectoryDataset(cdict['val_x'], cdict['val_y'], cdict['raw_val_y'])
        test_ds = TrajectoryDataset(cdict['test_x'], cdict['test_y'], cdict['raw_test_y'])

        client_loaders[cid] = {
            'name': cdict['name'],
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            'train_size': len(train_ds),
            'val_size': len(val_ds),
            'test_size': len(test_ds)
        }

        all_train_x.append(cdict['train_x'])
        all_train_y.append(cdict['train_y'])
        all_raw_train_y.append(cdict['raw_train_y'])

        all_val_x.append(cdict['val_x'])
        all_val_y.append(cdict['val_y'])
        all_raw_val_y.append(cdict['raw_val_y'])

        all_test_x.append(cdict['test_x'])
        all_test_y.append(cdict['test_y'])
        all_raw_test_y.append(cdict['raw_test_y'])

    cent_train_ds = TrajectoryDataset(
        np.concatenate(all_train_x, axis=0),
        np.concatenate(all_train_y, axis=0),
        np.concatenate(all_raw_train_y, axis=0)
    )
    cent_val_ds = TrajectoryDataset(
        np.concatenate(all_val_x, axis=0),
        np.concatenate(all_val_y, axis=0),
        np.concatenate(all_raw_val_y, axis=0)
    )
    cent_test_ds = TrajectoryDataset(
        np.concatenate(all_test_x, axis=0),
        np.concatenate(all_test_y, axis=0),
        np.concatenate(all_raw_test_y, axis=0)
    )

    centralized_loaders = {
        'train': DataLoader(cent_train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(cent_val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(cent_test_ds, batch_size=batch_size, shuffle=False),
        'train_size': len(cent_train_ds),
        'val_size': len(cent_val_ds),
        'test_size': len(cent_test_ds)
    }

    return client_loaders, centralized_loaders
