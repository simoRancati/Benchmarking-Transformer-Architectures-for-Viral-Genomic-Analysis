#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding visualization with pipeline reader (LogReg_mi_all) + PCA/UMAP/centroids,
USING ONLY common_samples_{scenario}.csv per scenario.

Per ogni scenario:
- legge common_samples_{scenario_safe}.csv da OUT_DIR/splits
- carica embeddings ViBE/XVir solo per quei sample_id
- calcola:
    * PCA (2D) + rescaling a range fisso
    * UMAP (2D)
    * silhouette score (PCA & UMAP)
    * distanze tra centroidi in PCA space
- salva:
    * un PCA .jpg per scenario
    * un UMAP .jpg per scenario
    * un .txt con i risultati
"""

import os, gc
import numpy as np
import pandas as pd

# ===================== CONFIG =====================

OUT_DIR = "/blue/simone.marini/share/Revision_IEEE/Results_Paired"

VIBE_ROOT = "/blue/simone.marini/share/Embedder_Benchmarking_AMIA/Transformers/ViBE"
XVIR_ROOT = "/blue/simone.marini/share/Embedder_Benchmarking_AMIA/Transformers/XVir"

# Scegli qui quale embedder visualizzare
EMB_ROOT = XVIR_ROOT           # <-- metti XVIR_ROOT per XVir
EMB_TAG  = "XVir"              # solo un tag nei nomi file ("ViBE", "XVir", "Local", ...)

BAC_FOLDER = "bac_preds"
HUM_FOLDER = "human_preds"
VIR_FOLDER = "hpv_preds"
LIMIT_BAC  = 184
LIMIT_HUM  = 185
LIMIT_VIR  = 185

VIZ_DIR = os.path.join(OUT_DIR, f"viz_{EMB_TAG}")
os.makedirs(VIZ_DIR, exist_ok=True)

OUTLIER_STD_THRESHOLD = 3.0
PCA_RANGE_MIN, PCA_RANGE_MAX = -10, 10

LABEL_COLORS_FULL = {
    "bacteria": "#9467bd",   # purple
    "human":    "#ff7f0e",   # orange
    "virus":    "#1f77b4"    # blue
}

NUMERIC_DTYPE = np.float32
CSV_CHUNKSIZE = 100_000

# ===================== SAMPLING CONFIG =====================
N_PER_CLASS = 1000     # max punti per classe in ogni run
N_REPEATS   = 5        # numero di run ripetute

# ===================== IO HELPERS (copiati da LogReg_mi_all) =====================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _downcast_numeric_inplace(df: pd.DataFrame):
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].dtype != NUMERIC_DTYPE:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(NUMERIC_DTYPE)

def _read_header(csv_path, has_header=True):
    df_head = pd.read_csv(
        csv_path,
        nrows=1,
        header=0 if has_header else None,
        low_memory=True
    )
    return df_head.columns.tolist()

def read_embeddings_filtered(csv_path, label, sample_id_filter=None, has_header=True):
    """
    Esattamente come nello script LogReg:
    - prima colonna = sample_id
    - resto = embeddings
    - filtro opzionale su sample_id_filter
    - conversione numerica + drop righe con NaN
    - ritorna DataFrame: [features numeriche] + sample_id + label
    """
    if not os.path.isfile(csv_path):
        return pd.DataFrame()

    try:
        cols = _read_header(csv_path, has_header=has_header)
    except Exception:
        return pd.DataFrame()

    if not cols:
        return pd.DataFrame()

    first_col = cols[0]
    chunks = []

    for chunk in pd.read_csv(
        csv_path,
        header=0 if has_header else None,
        low_memory=True,
        chunksize=CSV_CHUNKSIZE,
        usecols=cols
    ):
        if not has_header:
            chunk.columns = cols

        sample_ids = chunk[first_col].astype("string")

        if sample_id_filter is not None:
            mask = sample_ids.isin(sample_id_filter)
            if not mask.any():
                continue
            chunk = chunk.loc[mask].copy()
            sample_ids = sample_ids.loc[mask]

        chunk.drop(columns=[first_col], inplace=True)

        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        chunk.dropna(axis=0, how='any', inplace=True)

        _downcast_numeric_inplace(chunk)

        newcols = pd.DataFrame(
            {
                "sample_id": sample_ids.to_numpy(),
                "label": np.repeat(label, len(chunk))
            },
            index=chunk.index
        ).astype({"sample_id": "category", "label": "category"})

        chunk = pd.concat([chunk, newcols], axis=1, copy=False)
        chunks.append(chunk)
        gc.collect()

    if not chunks:
        return pd.DataFrame()

    out = pd.concat(chunks, ignore_index=True)
    return out

def load_embeddings(folder_path, label, limit_files=None, has_header=True, sample_id_filter=None):
    """
    Loader a livello cartella (come nello script LogReg).
    """
    if not os.path.isdir(folder_path):
        return pd.DataFrame()

    csv_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".csv")]
    if not csv_files:
        return pd.DataFrame()

    if limit_files is not None and limit_files < len(csv_files):
        rng = np.random.default_rng(42)
        csv_files = list(rng.choice(csv_files, size=limit_files, replace=False))

    frames = []
    for fn in csv_files:
        df = read_embeddings_filtered(
            os.path.join(folder_path, fn),
            label=label,
            sample_id_filter=sample_id_filter,
            has_header=has_header
        )
        if not df.empty:
            frames.append(df)
            gc.collect()

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out

def merge_and_clean(dfs):
    """
    Merge con intersezione delle colonne + drop NaN sulle numeriche.
    """
    nonempty = [d for d in dfs if d is not None and not d.empty]
    if not nonempty:
        return pd.DataFrame()

    combined = nonempty[0]
    for df in nonempty[1:]:
        common_cols = combined.columns.intersection(df.columns)
        if 'label' in combined.columns and 'label' in df.columns:
            common_cols = common_cols.union(['label'])
        combined = combined.loc[:, list(common_cols)]
        df2 = df.loc[:, list(common_cols)]
        combined = pd.concat([combined, df2], ignore_index=True, copy=False)

    num_cols = combined.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        combined.dropna(subset=list(num_cols), how='any', inplace=True)

    return combined

def concat_pipeline_frames(frames, keep_labels):
    """
    Merge + filtra label e imposta dtypes.
    """
    df = merge_and_clean(frames)
    if df.empty:
        return df

    df = df[df["label"].isin(keep_labels)]
    df["label"] = df["label"].astype("category")
    df["sample_id"] = df["sample_id"].astype("category")
    return df

def _folder_for_label(label, limits):
    if label == "bacteria":
        return BAC_FOLDER, limits.get("bac", None)
    if label == "human":
        return HUM_FOLDER, limits.get("hum", None)
    return VIR_FOLDER, limits.get("vir", None)

def _load_pipeline_filtered(root_dir, keep_labels, sample_filter, limits):
    """
    Carica embeddings per le label richieste, filtrando sui sample_id in sample_filter.
    """
    frames = []
    for label in keep_labels:
        sub, lim = _folder_for_label(label, limits)
        folder_path = os.path.join(root_dir, sub)
        df = load_embeddings(
            folder_path,
            label,
            limit_files=lim,
            sample_id_filter=sample_filter
        )
        if not df.empty:
            frames.append(df)
            gc.collect()

    if not frames:
        return pd.DataFrame()

    out = concat_pipeline_frames(frames, keep_labels)
    frames.clear()
    gc.collect()
    return out

def _safe_name(s):
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in s)

# ===================== DIM REDUCTION / PLOTTING =====================

from sklearn.decomposition import PCA
import umap.umap_ as umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="notebook", font_scale=1.2)
plt.rcParams["font.family"] = "Times New Roman"

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def remove_outliers_2d(X_2d, labels, threshold=3.0):
    mean_0 = np.mean(X_2d[:, 0])
    std_0  = np.std(X_2d[:, 0])
    mean_1 = np.mean(X_2d[:, 1])
    std_1  = np.std(X_2d[:, 1])

    mask = (
        (np.abs(X_2d[:, 0] - mean_0) <= threshold * std_0) &
        (np.abs(X_2d[:, 1] - mean_1) <= threshold * std_1)
    )
    return X_2d[mask], labels[mask]

def rescale_2d_to_range(X_2d, x_min, x_max):
    data_min = np.min(X_2d, axis=0)
    data_max = np.max(X_2d, axis=0)
    scaled = X_2d.copy()
    for dim in range(2):
        denom = (data_max[dim] - data_min[dim])
        if denom == 0:
            scaled[:, dim] = 0.5 * (x_min + x_max)
        else:
            scaled[:, dim] = ((X_2d[:, dim] - data_min[dim]) / denom) \
                             * (x_max - x_min) + x_min
    return scaled

def compute_centroid_distances(X_2d, labels):
    centroids = {}
    uniq = np.unique(labels)
    for lab in uniq:
        points_lab = X_2d[labels == lab]
        centroids[lab] = np.mean(points_lab, axis=0)

    distances = {}
    labs = list(uniq)
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            lab1, lab2 = labs[i], labs[j]
            dist = np.linalg.norm(centroids[lab1] - centroids[lab2])
            distances[(lab1, lab2)] = dist
    return distances

def plot_2d(
    X_2d,
    labels,
    title,
    outfile,
    label_order,
    color_dict,
    remove_outliers_flag=True,
    outlier_threshold=3.0
):
    original_n = X_2d.shape[0]

    if remove_outliers_flag:
        X_2d_plot, labels_plot = remove_outliers_2d(X_2d, labels, threshold=outlier_threshold)
        print(f"    [Outlier removal] {original_n - X_2d_plot.shape[0]} points removed for plotting.")
    else:
        X_2d_plot, labels_plot = X_2d, labels

    plt.figure(figsize=(8, 6))
    palette = {lab: color_dict[lab] for lab in label_order if lab in color_dict}

    sns.scatterplot(
        x=X_2d_plot[:, 0],
        y=X_2d_plot[:, 1],
        hue=labels_plot,
        hue_order=label_order,
        palette=palette,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        s=40
    )
    plt.title(title)
    plt.legend(title="Label", loc="best")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"  --> Saved {outfile} [plotted {X_2d_plot.shape[0]} / {original_n} points]")

def run_analysis(
    scenario_name,
    df_merged,
    label_order,
    color_dict,
    pca_filename,
    umap_filename,
    results_filename
):
    print(f"\n========== SCENARIO: {scenario_name} ({EMB_TAG}) ==========")

    # Tieni solo le label di interesse
    df_merged = df_merged[df_merged["label"].isin(label_order)].copy()
    if df_merged.empty:
        msg = f"{scenario_name}: no data for labels {label_order}\n"
        print(msg)
        with open(results_filename, "w") as f:
            f.write(msg)
        return

    label_col = "label"
    feature_cols = [c for c in df_merged.columns if c not in (label_col, "sample_id")]

    # Statistiche sul dataset completo (prima del subsampling)
    y_full = df_merged[label_col].values
    labs_unique_full, counts_full = np.unique(y_full, return_counts=True)
    print(f"  --> Total data shape = {df_merged[feature_cols].shape}, unique labels: {labs_unique_full}")
    print("  --> Label counts (full):", dict(zip(labs_unique_full, counts_full)))

    if len(labs_unique_full) < 2 or df_merged.shape[0] < 3:
        msg = f"{scenario_name}: not enough data or only one label in full data.\n"
        print(msg)
        with open(results_filename, "w") as f:
            f.write(msg)
        return

    from math import floor

    # Accumulatori per le metriche sulle 5 run
    sil_pca_list = []
    sil_umap_list = []
    centroid_dists_list = []

    # Salvo l'ultima run per fare i plot
    X_pca_for_plot = None
    X_umap_for_plot = None
    y_for_plot = None

    rng = np.random.default_rng(42)

    for rep in range(N_REPEATS):
        # ===== SUBSAMPLING STRATIFICATO: max N_PER_CLASS per ciascuna classe =====
        frames_rep = []
        for lab in label_order:
            df_lab = df_merged[df_merged[label_col] == lab]
            if df_lab.empty:
                continue
            n_lab = len(df_lab)
            n_sample = min(N_PER_CLASS, n_lab)
            rs = int(rng.integers(0, 1_000_000))
            df_sample = df_lab.sample(n=n_sample, replace=False, random_state=rs)
            frames_rep.append(df_sample)

        if not frames_rep:
            msg = f"{scenario_name}: no data after stratified sampling (rep {rep+1}).\n"
            print(msg)
            with open(results_filename, "w") as f:
                f.write(msg)
            return

        df_rep = pd.concat(frames_rep, ignore_index=True)
        features_df = df_rep[feature_cols]
        labels_series = df_rep[label_col]

        try:
            X_raw = features_df.astype(np.float32).values
        except ValueError as e:
            msg = f"{scenario_name}: float32 conversion error (rep {rep+1}): {e}\n"
            print(msg)
            with open(results_filename, "w") as f:
                f.write(msg)
            return

        y_raw = labels_series.values
        labs_unique_rep, counts_rep = np.unique(y_raw, return_counts=True)
        print(f"  [Rep {rep+1}/{N_REPEATS}] data shape = {X_raw.shape}, label counts:",
              dict(zip(labs_unique_rep, counts_rep)))

        if len(labs_unique_rep) < 2 or X_raw.shape[0] < 3:
            msg = f"{scenario_name}: not enough data or only one label after sampling (rep {rep+1}).\n"
            print(msg)
            with open(results_filename, "w") as f:
                f.write(msg)
            return

        # ===== PCA su dati scalati =====
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        pca = PCA(n_components=2, random_state=42)
        X_pca_2d = pca.fit_transform(X_scaled)
        X_pca_2d = rescale_2d_to_range(X_pca_2d, PCA_RANGE_MIN, PCA_RANGE_MAX)
        print("    --> PCA shape:", X_pca_2d.shape)

        # ===== UMAP su X_raw (come in versione originale) =====
        n_neighbors = floor(min(15, X_raw.shape[0] - 1))
        if n_neighbors < 2:
            msg = f"{scenario_name}: too few points for UMAP (rep {rep+1}).\n"
            print(msg)
            with open(results_filename, "w") as f:
                f.write(msg)
            return

        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=2,
            random_state=42
        )
        X_umap_2d = umap_reducer.fit_transform(X_raw)
        print("    --> UMAP shape:", X_umap_2d.shape)

        # ===== Silhouette & centroid distances su questo sottocampione =====
        enc = LabelEncoder()
        y_enc = enc.fit_transform(y_raw)
        sil_pca = silhouette_score(X_pca_2d, y_enc)
        sil_umap = silhouette_score(X_umap_2d, y_enc)
        sil_pca_list.append(sil_pca)
        sil_umap_list.append(sil_umap)

        centroid_dists = compute_centroid_distances(X_pca_2d, y_raw)
        centroid_dists_list.append(centroid_dists)

        # Salvo l'ultima run per il plotting
        X_pca_for_plot = X_pca_2d
        X_umap_for_plot = X_umap_2d
        y_for_plot = y_raw

    # ===== MEDIA DELLE METRICHE SULLE 5 RUN =====
    mean_sil_pca = float(np.mean(sil_pca_list))
    mean_sil_umap = float(np.mean(sil_umap_list))

    # Media delle distanze tra centroidi (per ogni coppia di classi)
    all_keys = set()
    for d in centroid_dists_list:
        all_keys.update(d.keys())
    centroid_dists_mean = {}
    for key in sorted(all_keys):
        vals = [d[key] for d in centroid_dists_list if key in d]
        centroid_dists_mean[key] = float(np.mean(vals))

    # ===== Plot solo sull'ultima run =====
    if X_pca_for_plot is not None and y_for_plot is not None:
        plot_2d(
            X_2d=X_pca_for_plot,
            labels=y_for_plot,
            title=f"PCA (2D) - {scenario_name} - {EMB_TAG} (last run, subsampled)",
            outfile=pca_filename,
            label_order=label_order,
            color_dict=color_dict,
            remove_outliers_flag=True,
            outlier_threshold=OUTLIER_STD_THRESHOLD
        )
        plot_2d(
            X_2d=X_umap_for_plot,
            labels=y_for_plot,
            title=f"UMAP (2D) - {scenario_name} - {EMB_TAG} (last run, subsampled)",
            outfile=umap_filename,
            label_order=label_order,
            color_dict=color_dict,
            remove_outliers_flag=True,
            outlier_threshold=OUTLIER_STD_THRESHOLD
        )

    # ===== Output dei risultati medi =====
    label_counts_dict = dict(zip(labs_unique_full, counts_full))
    lines = []
    lines.append(f"Scenario: {scenario_name} ({EMB_TAG})")
    lines.append(f"n_samples_total={df_merged.shape[0]}, n_features={len(feature_cols)}")
    lines.append(
        f"Subsampling: up to {N_PER_CLASS} samples per class, "
        f"{N_REPEATS} repeated runs; metrics are means over runs."
    )
    lines.append("Label counts (full): " + ", ".join(f"{lab}={label_counts_dict[lab]}" for lab in labs_unique_full))
    lines.append(f"Silhouette (PCA 2D) [mean over {N_REPEATS} runs]: {mean_sil_pca:.4f}")
    lines.append(f"Silhouette (UMAP 2D) [mean over {N_REPEATS} runs]: {mean_sil_umap:.4f}")
    for (lab1, lab2), dist in sorted(centroid_dists_mean.items()):
        lines.append(
            f"Centroid distance {lab1} vs {lab2} (PCA space, mean over {N_REPEATS} runs): {dist:.4f}"
        )

    results_str = "\n".join(lines)
    print("\n---- RESULTS (mean over runs) ----\n" + results_str + "\n")
    with open(results_filename, "w") as f:
        f.write(results_str + "\n")

# ===================== MAIN =====================

def main():
    print(f">>> Using embeddings from: {EMB_ROOT} (tag={EMB_TAG})")
    limits = {"bac": LIMIT_BAC, "hum": LIMIT_HUM, "vir": LIMIT_VIR}
    splits_dir = os.path.join(OUT_DIR, "splits")

    # Gli scenari per cui ti aspetti i file common_samples_*.csv
    scenarios = [
        ("Human vs Virus", ["human", "virus"]),
        ("Bacteria vs Virus", ["bacteria", "virus"]),
        ("Bacteria vs Human", ["bacteria", "human"]),
        ("Bacteria vs Human vs Virus", ["bacteria", "human", "virus"]),
    ]

    for scenario_name, labs in scenarios:
        scenario_safe = _safe_name(scenario_name)
        common_path = os.path.join(splits_dir, f"common_samples_{scenario_safe}.csv")

        if not os.path.isfile(common_path):
            print(f"!!! Missing common samples for scenario '{scenario_name}': {common_path} -> SKIP")
            continue

        common_ids = set(
            pd.read_csv(common_path, usecols=["sample_id"])["sample_id"]
            .astype("string")
            .unique()
        )
        print(f"\n>>> Scenario '{scenario_name}': loaded {len(common_ids)} common sample_id")

        # Carico embeddings SOLO per questi sample_id e queste label
        df_scenario = _load_pipeline_filtered(
            root_dir=EMB_ROOT,
            keep_labels=labs,
            sample_filter=common_ids,
            limits=limits
        )

        if df_scenario.empty:
            print(f"!!! No embeddings loaded for '{scenario_name}' with common_samples filter.")
            continue

        safe = _safe_name(f"{EMB_TAG}_{scenario_name}")
        pca_file = os.path.join(VIZ_DIR, f"{safe}_pca.jpg")
        umap_file = os.path.join(VIZ_DIR, f"{safe}_umap.jpg")
        res_file = os.path.join(VIZ_DIR, f"{safe}_results.txt")

        color_dict = {lab: LABEL_COLORS_FULL[lab] for lab in labs}
        run_analysis(
            scenario_name=scenario_name,
            df_merged=df_scenario,
            label_order=labs,
            color_dict=color_dict,
            pca_filename=pca_file,
            umap_filename=umap_file,
            results_filename=res_file
        )

    print(">>> All scenarios complete.")

if __name__ == "__main__":
    main()
