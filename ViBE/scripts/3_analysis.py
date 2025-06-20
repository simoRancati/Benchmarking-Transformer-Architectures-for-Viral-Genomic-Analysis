#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# Dimensionality Reduction
from sklearn.decomposition import PCA
import umap.umap_ as umap

# Plotting (non-interactive)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Scoring & Classification
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

###############################################################################
# GLOBAL SETTINGS FOR PRETTIER PLOTS
###############################################################################
# Slightly larger text/markers and a “whitegrid” style
sns.set(style="whitegrid", context="notebook", font_scale=1.2)
# Use Times New Roman globally
plt.rcParams["font.family"] = "Times New Roman"

###############################################################################
# ADJUSTABLE PARAMETERS
###############################################################################
MAX_ROWS = 10            # How many rows to read from each CSV (use None for all)
OUTLIER_STD_THRESHOLD = 3.0  # Remove outliers > 3 stdev from mean in 2D plots
PCA_RANGE_MIN, PCA_RANGE_MAX = -10, 10  # Range for final PCA coords
# Colors for each label (pick any you like)
LABEL_COLORS_FULL = {
    "bacteria": "#9467bd",   # Purple
    "human":    "#ff7f0e",   # Orange
    "virus":    "#1f77b4"    # Blue
}

###############################################################################
# HELPERS: LOADING / CLEANING
###############################################################################
def read_numeric_embeddings(csv_path, has_header=True):
    """
    Reads up to MAX_ROWS from `csv_path`. 
      - If has_header=True, interpret the first row as column names.
      - Select only numeric columns, drop rows w/ any NaNs.
    Returns a numeric DataFrame or None if empty.
    """
    print(f"  --> Reading file: {csv_path}")
    if not os.path.isfile(csv_path):
        print(f"  ! File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(
            csv_path,
            header=0 if has_header else None,
            low_memory=False,
            nrows=MAX_ROWS
        )
    except Exception as e:
        print(f"  ! Failed to read {csv_path}: {e}")
        return None

    if df.empty:
        print(f"  ! File {csv_path} is empty.")
        return None

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()
    if df_numeric.empty:
        print(f"  ! File {csv_path} has no numeric columns.")
        return None

    # Drop rows that have any NaN
    before_drop = df_numeric.shape[0]
    df_numeric.dropna(axis=0, how='any', inplace=True)
    after_drop = df_numeric.shape[0]
    if df_numeric.empty:
        print(f"  ! File {csv_path} had all rows dropped due to NaNs.")
        return None

    print(f"  --> Loaded shape {df_numeric.shape}; dropped {before_drop - after_drop} rows w/ NaNs.")
    return df_numeric

def load_embeddings(folder_path, label, limit_files=None, has_header=True):
    """
    Finds CSV files in `folder_path`, reads each with `read_numeric_embeddings`,
    appends a 'label' column = label, and concatenates them.
    If limit_files is given, randomly samples that many from the folder.
    Returns a combined numeric DataFrame with shape (#rows, #cols+1).
    """
    print(f"\n=== Loading folder: {folder_path} with label: {label} ===")
    if not os.path.isdir(folder_path):
        print(f"  ! Folder not found: {folder_path}")
        return pd.DataFrame()

    csv_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".csv")]
    print(f"  --> Found {len(csv_files)} CSV files in {folder_path}.")
    if not csv_files:
        print(f"  ! No CSV files found in {folder_path}")
        return pd.DataFrame()

    if limit_files is not None and limit_files < len(csv_files):
        np.random.seed(42)
        chosen = np.random.choice(csv_files, size=limit_files, replace=False)
        csv_files = chosen
        print(f"  --> Randomly selected {limit_files} of these CSVs.")

    all_dfs = []
    for fn in csv_files:
        path = os.path.join(folder_path, fn)
        df_numeric = read_numeric_embeddings(path, has_header=has_header)
        if df_numeric is not None and not df_numeric.empty:
            df_numeric["label"] = label
            all_dfs.append(df_numeric)
        else:
            print(f"  ! Skipping {fn} (no valid numeric data).")

    if not all_dfs:
        print(f"  ! No valid embeddings found in {folder_path}")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"=== Finished loading {folder_path}. Combined shape: {combined_df.shape} ===\n")
    return combined_df

def merge_and_clean(dfs):
    """
    Merges multiple labeled DataFrames on common columns (besides 'label'),
    drops any rows w/ NaNs. Returns combined DataFrame.
    """
    print("  --> Merging dataframes...")
    nonempty = [df for df in dfs if not df.empty]
    if not nonempty:
        print("  ! All dataframes empty, nothing to merge.")
        return pd.DataFrame()

    combined = nonempty[0].copy()
    for df in nonempty[1:]:
        common_cols = combined.columns.intersection(df.columns)
        if 'label' in combined.columns and 'label' in df.columns:
            common_cols = common_cols.union(['label'])
        combined = combined[list(common_cols)]
        df2 = df[list(common_cols)]
        combined = pd.concat([combined, df2], ignore_index=True)

    before = combined.shape[0]
    combined.dropna(axis=0, how='any', inplace=True)
    after = combined.shape[0]
    print(f"  --> Merged shape: {combined.shape}, dropped {before - after} rows w/ NaNs.")
    return combined

###############################################################################
# OUTLIER REMOVAL (for plotting only)
###############################################################################
def remove_outliers_2d(X_2d, labels, threshold=3.0):
    """
    Removes points > threshold stdev from the mean along EITHER axis
    in X_2d. Returns (X_2d_filtered, labels_filtered).
    """
    mean_0 = np.mean(X_2d[:, 0])
    std_0  = np.std(X_2d[:, 0])
    mean_1 = np.mean(X_2d[:, 1])
    std_1  = np.std(X_2d[:, 1])

    mask = (
        (np.abs(X_2d[:, 0] - mean_0) <= threshold * std_0) &
        (np.abs(X_2d[:, 1] - mean_1) <= threshold * std_1)
    )
    return X_2d[mask], labels[mask]

###############################################################################
# POST-PCA SCALING FOR COMPARABLE CENTROIDS
###############################################################################
def rescale_2d_to_range(X_2d, x_min, x_max):
    """
    MinMax-scale each axis of X_2d to [x_min, x_max].
    (Applies the same transform to X and Y independently.)
    """
    # For each axis, find data min/max
    data_min = np.min(X_2d, axis=0)
    data_max = np.max(X_2d, axis=0)
    scaled = X_2d.copy()
    for dim in range(2):
        denom = (data_max[dim] - data_min[dim])
        if denom == 0:
            # Means everything is same value => just set them to midpoint
            scaled[:, dim] = 0.5 * (x_min + x_max)
        else:
            # Scale to [0,1], then to [x_min, x_max]
            scaled[:, dim] = ((X_2d[:, dim] - data_min[dim]) / denom) \
                             * (x_max - x_min) + x_min
    return scaled

###############################################################################
# CENTROID DISTANCE CALC
###############################################################################
def compute_centroid_distances(X_2d, labels):
    """
    For each unique label, compute its centroid in X_2d, 
    then compute pairwise Euclidean distances among them.
    Returns dict { (lab1,lab2): dist, ... }.
    """
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

###############################################################################
# PLOTTING
###############################################################################
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
    """
    Scatter-plot X_2d vs labels, optionally removing outliers 
    so they don't clutter the visualization. Then save to `outfile`.
    """
    original_n = X_2d.shape[0]
    if remove_outliers_flag:
        X_2d_plot, labels_plot = remove_outliers_2d(X_2d, labels, threshold=outlier_threshold)
        print(f"    [Outlier removal] {original_n - X_2d_plot.shape[0]} points removed for plotting.")
    else:
        X_2d_plot, labels_plot = X_2d, labels

    plt.figure(figsize=(8,6))
    # Make a list of colors in the same order as label_order
    used_labels = np.unique(labels_plot)
    pal = [color_dict[l] for l in label_order if l in used_labels]

    sns.scatterplot(
        x=X_2d_plot[:, 0],
        y=X_2d_plot[:, 1],
        hue=labels_plot,
        hue_order=label_order,  # ensures consistent color ordering
        palette=pal,
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

###############################################################################
# MAIN ANALYSIS
###############################################################################
def run_analysis(
    scenario_name,
    dataframes,          
    label_order,         
    color_dict,          
    pca_filename,
    umap_filename,
    results_filename
):
    """
    1) Merge DataFrames => X,y
    2) StandardScaler => PCA => forcibly rescale PCA to [PCA_RANGE_MIN, PCA_RANGE_MAX]
    3) UMAP on raw data
    4) Silhouette, centroid distances, outlier-removed plots
    5) Classification
    """
    print(f"\n========== SCENARIO: {scenario_name} ==========")
    df_merged = merge_and_clean([df for (df, _) in dataframes])
    if df_merged.empty:
        msg = f"{scenario_name}: All data empty or dropped.\n"
        print(msg)
        with open(results_filename, "w") as f:
            f.write(msg)
        return

    label_col = "label"
    features_df = df_merged.drop(columns=[label_col])
    labels_series = df_merged[label_col]
    try:
        X_raw = features_df.astype(np.float32).values
    except ValueError as e:
        msg = f"{scenario_name}: float32 conversion error: {e}\n"
        print(msg)
        with open(results_filename, "w") as f:
            f.write(msg)
        return
    y_raw = labels_series.values

    labs_unique, counts = np.unique(y_raw, return_counts=True)
    print(f"  --> Final data shape = {X_raw.shape}, unique labels: {labs_unique}")
    print("  --> Label counts:", dict(zip(labs_unique, counts)))
    if len(labs_unique) < 2 or X_raw.shape[0] < 3:
        msg = f"{scenario_name}: Not enough data or only one label.\n"
        print(msg)
        with open(results_filename, "w") as f:
            f.write(msg)
        return

    # (1) Standardize for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # (2) PCA => 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca_2d = pca.fit_transform(X_scaled)
    print("  --> PCA shape:", X_pca_2d.shape)

    # (3) Force the PCA coords to [PCA_RANGE_MIN, PCA_RANGE_MAX]
    X_pca_2d = rescale_2d_to_range(X_pca_2d, PCA_RANGE_MIN, PCA_RANGE_MAX)

    # (4) UMAP => 2D (on raw or scaled; your choice. We'll do raw.)
    from math import floor
    n_neighbors = floor(min(15, X_raw.shape[0] - 1))
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    X_umap_2d = umap_reducer.fit_transform(X_raw)

    # (5) Silhouette
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y_raw)
    sil_pca = silhouette_score(X_pca_2d, y_enc)
    sil_umap = silhouette_score(X_umap_2d, y_enc)

    # (6) Centroid Distances in the scaled-PCA space
    centroid_dists = compute_centroid_distances(X_pca_2d, y_raw)

    # (7) Plot PCA/UMAP with outlier removal
    plot_2d(
        X_2d=X_pca_2d, 
        labels=y_raw, 
        title=f"PCA (2D) - {scenario_name}", 
        outfile=pca_filename,
        label_order=label_order,
        color_dict=color_dict,
        remove_outliers_flag=True,
        outlier_threshold=OUTLIER_STD_THRESHOLD
    )
    plot_2d(
        X_2d=X_umap_2d, 
        labels=y_raw,
        title=f"UMAP (2D) - {scenario_name}",
        outfile=umap_filename,
        label_order=label_order,
        color_dict=color_dict,
        remove_outliers_flag=True,
        outlier_threshold=OUTLIER_STD_THRESHOLD
    )

    # (8) Classification
    if min(counts) < 3:
        msg = (f"{scenario_name}: Not enough samples per label for 3-fold CV.\n"
               f"Silhouette PCA:  {sil_pca:.4f}\n"
               f"Silhouette UMAP: {sil_umap:.4f}\n")
        print(msg)
        with open(results_filename, "w") as f:
            f.write(msg)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    print("  --> Running RandomForest grid search...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }
    rf_cv = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, scoring='accuracy')
    rf_cv.fit(X_train, y_train)
    rf_best = rf_cv.best_estimator_
    rf_preds = rf_best.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print("  --> Running LogisticRegression grid search...")
    lr = LogisticRegression(random_state=42, max_iter=10000)
    lr_params = {
        'penalty': ['l2'],
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'saga']
    }
    lr_cv = GridSearchCV(lr, lr_params, cv=3, n_jobs=-1, scoring='accuracy')
    lr_cv.fit(X_train, y_train)
    lr_best = lr_cv.best_estimator_
    lr_preds = lr_best.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    # (9) Save results
    lines = []
    lines.append(f"Scenario: {scenario_name}")
    lines.append(f"Data shape: X={X_raw.shape}, unique labels={set(y_raw)}")
    lines.append(f"Silhouette PCA:  {sil_pca:.4f}  (scaled PCA in range [{PCA_RANGE_MIN}, {PCA_RANGE_MAX}])")
    lines.append(f"Silhouette UMAP: {sil_umap:.4f}")
    lines.append("\nPairwise Centroid Distances (scaled PCA space):")
    for (labpair, distval) in centroid_dists.items():
        lines.append(f"  {labpair}: {distval:.4f}")
    lines.append("\nRandomForest best params: " + str(rf_cv.best_params_))
    lines.append(f"RandomForest best CV score: {rf_cv.best_score_:.4f}")
    lines.append(f"RandomForest test accuracy: {rf_acc:.4f}")
    lines.append("\nLogisticRegression best params: " + str(lr_cv.best_params_))
    lines.append(f"LogisticRegression best CV score: {lr_cv.best_score_:.4f}")
    lines.append(f"LogisticRegression test accuracy: {lr_acc:.4f}")

    results_str = "\n".join(lines)
    print("\n---- RESULTS ----\n" + results_str + "\n")
    with open(results_filename, "w") as f:
        f.write(results_str + "\n")

###############################################################################
# EXAMPLE MAIN
###############################################################################
def main():
    """
    Example main function that runs 4 analyses:
      1) Human vs Virus
      2) Bacteria vs Virus
      3) Bacteria vs Human
      4) Bacteria vs Human vs Virus

    Adjust the folder names, limits, and scenario calls as needed.
    """
    print(">>> Starting main function...")

    # Example folders for your CSV files
    bac_folder   = "bac_preds"
    human_folder = "human_preds"
    virus_folder = "hpv_preds"

    # Example: limit how many CSVs to read
    limit_bac   = 184
    limit_human = 185
    limit_virus = 185

    print(">>> Loading Bacteria embeddings...")
    df_bac = load_embeddings(bac_folder, "bacteria", limit_files=limit_bac)
    print(">>> Loading Human embeddings...")
    df_hum = load_embeddings(human_folder, "human",   limit_files=limit_human)
    print(">>> Loading Virus embeddings...")
    df_vir = load_embeddings(virus_folder, "virus",   limit_files=limit_virus)

    # SCENARIO 1: Human vs Virus
    run_analysis(
        scenario_name   = "Human vs Virus",
        dataframes      = [(df_hum, "human"), (df_vir, "virus")],
        label_order     = ["human", "virus"],
        color_dict      = {"human":"#ff7f0e", "virus":"#1f77b4"},
        pca_filename    = "pca_human_vs_virus.jpg",
        umap_filename   = "umap_human_vs_virus.jpg",
        results_filename= "results_human_vs_virus.txt"
    )

    # SCENARIO 2: Bacteria vs Virus
    run_analysis(
        scenario_name   = "Bacteria vs Virus",
        dataframes      = [(df_bac, "bacteria"), (df_vir, "virus")],
        label_order     = ["bacteria", "virus"],
        color_dict      = {"bacteria":"#9467bd","virus":"#1f77b4"},
        pca_filename    = "pca_bacteria_vs_virus.jpg",
        umap_filename   = "umap_bacteria_vs_virus.jpg",
        results_filename= "results_bacteria_vs_virus.txt"
    )

    # SCENARIO 3: Bacteria vs Human
    run_analysis(
        scenario_name   = "Bacteria vs Human",
        dataframes      = [(df_bac, "bacteria"), (df_hum, "human")],
        label_order     = ["bacteria","human"],
        color_dict      = {"bacteria":"#9467bd","human":"#ff7f0e"},
        pca_filename    = "pca_bacteria_vs_human.jpg",
        umap_filename   = "umap_bacteria_vs_human.jpg",
        results_filename= "results_bacteria_vs_human.txt"
    )

    # SCENARIO 4: Bacteria vs Human vs Virus
    run_analysis(
        scenario_name   = "Bacteria vs Human vs Virus",
        dataframes      = [(df_bac, "bacteria"), (df_hum, "human"), (df_vir, "virus")],
        label_order     = ["bacteria","human","virus"],
        color_dict      = {"bacteria":"#9467bd","human":"#ff7f0e","virus":"#1f77b4"},
        pca_filename    = "pca_bacteria_vs_human_vs_virus.jpg",
        umap_filename   = "umap_bacteria_vs_human_vs_virus.jpg",
        results_filename= "results_bacteria_vs_human_vs_virus.txt"
    )

    print(">>> All scenarios complete.")

if __name__ == "__main__":
    main()
