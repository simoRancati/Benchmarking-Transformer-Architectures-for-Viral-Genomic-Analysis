#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import umap.umap_ as umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Use Times New Roman globally
plt.rcParams['font.family'] = "Times New Roman"
sns.set(style="whitegrid")

# None if all should be used
MAX_ROWS = 10

# Define color mapping + order for these two labels
LABEL_COLORS = {
    "human": "#ff7f0e",   # Orange
    "virus": "#2ca02c"    # Green
}
HUE_ORDER = ["human", "virus"]

def read_numeric_embeddings(csv_path, has_header=True):
    if not os.path.isfile(csv_path):
        return None
    
    try:
        df = pd.read_csv(
            csv_path,
            header=0 if has_header else None,
            low_memory=False,
            nrows=MAX_ROWS
        )
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return None

    if df.empty:
        print(f"File {csv_path} is empty.")
        return None

    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].copy()

    if df_numeric.empty:
        print(f"File {csv_path} has no numeric columns after dropping non-numerics.")
        return None

    # Drop rows with NaNs
    df_numeric.dropna(axis=0, how='any', inplace=True)

    if df_numeric.empty:
        print(f"File {csv_path} has all rows dropped due to NaNs.")
        return None

    return df_numeric

def load_embeddings_from_folder(folder_path, label, has_header=True, limit_files=None):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return pd.DataFrame()

    csv_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return pd.DataFrame()

    if limit_files is not None and limit_files < len(csv_files):
        np.random.seed(42)
        csv_files = np.random.choice(csv_files, size=limit_files, replace=False)

    all_dfs = []
    for fn in csv_files:
        path = os.path.join(folder_path, fn)
        df_numeric = read_numeric_embeddings(path, has_header=has_header)
        if df_numeric is not None and not df_numeric.empty:
            print(f"Loaded {fn} with shape {df_numeric.shape}")
            df_numeric["label"] = label
            all_dfs.append(df_numeric)
        else:
            print(f"File {fn} did not produce valid numeric data.")
    
    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def main():
    # Folders
    human_folder = "human_preds"  
    virus_folder = "hpv_preds"

    # Limits
    limit_human = 185
    limit_virus = 185

    print("Loading human embeddings...")
    df_human = load_embeddings_from_folder(
        folder_path=human_folder,
        label="human",
        has_header=True,
        limit_files=limit_human
    )
    print(f"Human combined shape = {df_human.shape}")

    print("Loading virus embeddings...")
    df_virus = load_embeddings_from_folder(
        folder_path=virus_folder,
        label="virus",
        has_header=True,
        limit_files=limit_virus
    )
    print(f"Virus combined shape = {df_virus.shape}")

    # Combine
    if df_human.empty and df_virus.empty:
        print("No human or virus data loaded. Exiting.")
        return
    elif df_human.empty:
        df_combined = df_virus
    elif df_virus.empty:
        df_combined = df_human
    else:
        common_cols = df_human.columns.intersection(df_virus.columns)
        if 'label' in df_human.columns and 'label' in df_virus.columns:
            common_cols = common_cols.union(['label'])
        
        df_human = df_human[list(common_cols)]
        df_virus = df_virus[list(common_cols)]
        df_combined = pd.concat([df_human, df_virus], ignore_index=True)

    print(f"Combined dataset shape before dropping NaNs = {df_combined.shape}")

    df_combined.dropna(axis=0, how='any', inplace=True)
    print(f"Combined dataset shape after dropping NaNs = {df_combined.shape}")

    if df_combined.empty:
        msg = "All data was dropped after removing mismatched columns/NaNs. Exiting."
        print(msg)
        with open("results.txt", "w") as f:
            f.write(msg + "\n")
        return

    # Separate features / label
    label_col = 'label'
    features_df = df_combined.drop(columns=[label_col])
    labels_series = df_combined[label_col].copy()

    # Convert to float32
    try:
        X = features_df.astype(np.float32).values
    except ValueError as e:
        print(f"Error converting features to float32: {e}")
        with open("results.txt", "w") as f:
            f.write("Failed to convert features to float32.\n")
        return

    y = labels_series.values
    print(f"Final dataset: X.shape = {X.shape}, y.shape = {y.shape}")

    unique_labels, label_counts = np.unique(y, return_counts=True)
    print("Unique labels:", unique_labels)
    print("Label counts:", dict(zip(unique_labels, label_counts)))

    # If too few samples for a plot
    if X.shape[0] < 3 or len(unique_labels) < 2:
        msg = "Not enough data (fewer than 3 rows or only one label). Skipping analyses."
        print(msg)
        with open("results.txt", "w") as f:
            f.write(msg + "\n")
        return

    # PCA
    print("\nPerforming PCA (2D)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # UMAP
    print("Performing UMAP (2D)...")
    n_neighbors = min(15, X.shape[0] - 1)
    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                             n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X)

    # Silhouette
    label_enc = LabelEncoder()
    y_numeric = label_enc.fit_transform(y)
    sil_pca = silhouette_score(X_pca, y_numeric)
    sil_umap = silhouette_score(X_umap, y_numeric)

    def plot_2d(X_2d, labels, title, filename):
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            hue=labels,
            hue_order=HUE_ORDER,  
            palette=LABEL_COLORS,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.3,
            s=40
        )
        plt.title(title)
        plt.legend(title="Label")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    plot_2d(X_pca, y, "PCA (2D) - Human vs Virus", "pca_plot.jpg")
    plot_2d(X_umap, y, "UMAP (2D) - Human vs Virus", "umap_plot.jpg")

    # Classification
    min_samples_per_class = min(label_counts)
    if min_samples_per_class < 3:
        msg = ("Not enough samples per class to do 3-fold CV. "
               "Skipping classification.\n")
        print(msg)
        with open("results.txt", "w") as f:
            f.write(
                f"Final dataset shape: X={X.shape}, unique labels={set(y)}\n"
                f"Silhouette PCA:  {sil_pca:.4f}\n"
                f"Silhouette UMAP: {sil_umap:.4f}\n\n"
            )
            f.write(msg)
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
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

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=10000)
    lr_params = {
        'penalty': ['l2'],
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'saga']
    }
    lr_cv.fit(X_train, y_train)
    lr_best = lr_cv.best_estimator_
    lr_preds = lr_best.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    # Save results
    lines = []
    lines.append(f"Final dataset shape: X={X.shape}, unique labels={set(y)}")
    lines.append(f"Silhouette PCA:  {sil_pca:.4f}")
    lines.append(f"Silhouette UMAP: {sil_umap:.4f}")
    lines.append("\nRandomForest best params: " + str(rf_cv.best_params_))
    lines.append(f"RandomForest best CV score: {rf_cv.best_score_:.4f}")
    lines.append(f"RandomForest test accuracy: {rf_acc:.4f}")
    lines.append("\nLogisticRegression best params: " + str(lr_cv.best_params_))
    lines.append(f"LogisticRegression best CV score: {lr_cv.best_score_:.4f}")
    lines.append(f"LogisticRegression test accuracy: {lr_acc:.4f}")

    results = "\n".join(lines)
    print("\n---- RESULTS ----\n" + results + "\n")
    with open("results.txt", "w") as f:
        f.write(results + "\n")


if __name__ == "__main__":
    main()
