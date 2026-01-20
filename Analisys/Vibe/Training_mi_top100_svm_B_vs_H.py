#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SVM with SelectKBest(mutual_info_classif) — Bacteria vs Human — top100
# ViBE and XVir evaluated separately with saved splits. Predictions and summary CSV are saved.

import os, gc, logging, numpy as np, pandas as pd
from logging.handlers import RotatingFileHandler

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Final models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ===================== CONFIG =====================
OUT_DIR = "/blue/simone.marini/share/Revision_IEEE/Results_Paired"
VIBE_ROOT = "/blue/simone.marini/share/Embedder_Benchmarking_AMIA/Transformers/ViBE"
XVIR_ROOT = "/blue/simone.marini/share/Embedder_Benchmarking_AMIA/Transformers/XVir"

BAC_FOLDER = "bac_preds"; HUM_FOLDER = "human_preds"; VIR_FOLDER = "hpv_preds"
LIMIT_BAC = 184; LIMIT_HUM = 185; LIMIT_VIR = 185

SCENARIO_NAME = "Bacteria vs Human"
KEEP_LABELS = "bacteria", "human"

CSV_CHUNKSIZE = 100_000
NUMERIC_DTYPE = np.float32

TOPK = 100   # None => all (k='all')
MODEL_NAME = "SVM"
TOPK_TAG = "top100"

PRED_BASE_DIR = os.path.join(OUT_DIR, "predictions")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(PRED_BASE_DIR, exist_ok=True)

# ===================== Logging =====================
logger = logging.getLogger("SVM_mi_top100")
logger.setLevel(logging.INFO)

def setup_logger(log_path: str, level=logging.INFO):
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setLevel(level); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level); fh.setFormatter(fmt); logger.addHandler(fh)
    logger.info(f"Logger initialized. Writing to: {os.path.abspath(log_path)}")

# ===================== IO helpers =====================
def _ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def _downcast_numeric_inplace(df: pd.DataFrame):
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].dtype != NUMERIC_DTYPE:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(NUMERIC_DTYPE)

def _read_header(csv_path, has_header=True):
    df_head = pd.read_csv(csv_path, nrows=1, header=0 if has_header else None, low_memory=True)
    return df_head.columns.tolist()

def read_embeddings_filtered(csv_path, label, sample_id_filter=None, has_header=True):
    if not os.path.isfile(csv_path): return pd.DataFrame()
    try:
        cols = _read_header(csv_path, has_header=has_header)
    except Exception:
        return pd.DataFrame()
    if not cols: return pd.DataFrame()
    first_col = cols[0]
    chunks = []
    for chunk in pd.read_csv(csv_path, header=0 if has_header else None, low_memory=True,
                             chunksize=CSV_CHUNKSIZE, usecols=cols):
        if not has_header: chunk.columns = cols
        sample_ids = chunk[first_col].astype("string")
        if sample_id_filter is not None:
            mask = sample_ids.isin(sample_id_filter)
            if not mask.any(): continue
            chunk = chunk.loc[mask].copy()
            sample_ids = sample_ids.loc[mask]
        chunk.drop(columns=[first_col], inplace=True)
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        chunk.dropna(axis=0, how='any', inplace=True)
        _downcast_numeric_inplace(chunk)
        newcols = pd.DataFrame(
            {
                "sample_id": sample_ids.to_numpy(),  # già allineati per index
                "label": np.repeat(label, len(chunk))  # vettore ripetuto
            },
            index=chunk.index
        ).astype({"sample_id": "category", "label": "category"})
        # Concat una sola volta
        chunk = pd.concat([chunk, newcols], axis=1, copy=False)
        chunks.append(chunk); gc.collect()
    if not chunks: return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    return out

def load_embeddings(folder_path, label, limit_files=None, has_header=True, sample_id_filter=None):
    if not os.path.isdir(folder_path): return pd.DataFrame()
    csv_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".csv")]
    if not csv_files: return pd.DataFrame()
    if limit_files is not None and limit_files < len(csv_files):
        rng = np.random.default_rng(42)
        csv_files = list(rng.choice(csv_files, size=limit_files, replace=False))
    frames = []
    for fn in csv_files:
        df = read_embeddings_filtered(os.path.join(folder_path, fn), label=label,
                                      sample_id_filter=sample_id_filter, has_header=has_header)
        if not df.empty: frames.append(df); gc.collect()
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

def merge_and_clean(dfs):
    # Tieni solo DataFrame non vuoti
    nonempty = [d for d in dfs if d is not None and not d.empty]
    if not nonempty:
        return pd.DataFrame()

    combined = nonempty[0]
    for df in nonempty[1:]:
        # Intersezione colonne per evitare mismatch
        common_cols = combined.columns.intersection(df.columns)
        # preserva 'label' se presente
        if 'label' in combined.columns and 'label' in df.columns:
            common_cols = common_cols.union(['label'])
        combined = combined.loc[:, list(common_cols)]
        df2 = df.loc[:, list(common_cols)]
        combined = pd.concat([combined, df2], ignore_index=True, copy=False)

    # Drop righe con NaN nelle feature numeriche
    num_cols = combined.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        combined.dropna(subset=list(num_cols), how='any', inplace=True)

    return combined

def concat_pipeline_frames(frames, keep_labels):
    df = merge_and_clean(frames)
    if df.empty:
        return df
    # Filtra le etichette richieste
    df = df[df["label"].isin(keep_labels)]
    # Tipi categoriali consistenti
    df["label"] = df["label"].astype("category")
    df["sample_id"] = df["sample_id"].astype("category")
    return df

def _folder_for_label(label, limits):
    if label == "bacteria": return "bac_preds", limits.get("bac", None)
    if label == "human":    return "human_preds", limits.get("hum", None)
    return "hpv_preds",     limits.get("vir", None)

def _load_pipeline_filtered(root_dir, keep_label, sample_filter, limits):
    frames = []
    for label in keep_label:
        sub, lim = _folder_for_label(label, limits)
        folder_path = os.path.join(root_dir, sub)
        df = load_embeddings(folder_path, label, limit_files=lim, sample_id_filter=sample_filter)
        if not df.empty: frames.append(df); gc.collect()
    if not frames: return pd.DataFrame()
    out = concat_pipeline_frames(frames, keep_label)
    frames.clear(); gc.collect()
    return out

# ===================== ML utils =====================
def _get_sgkf(n_splits, seed):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except Exception:
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=n_splits)

def _to_xyg(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("label", "sample_id")]
    if not num_cols: raise ValueError("Nessuna feature numerica trovata!")
    X = df[num_cols].values.astype(NUMERIC_DTYPE, copy=False)
    y = df["label"].astype("category").values
    g = df["sample_id"].astype("category").values
    return X, y, g, num_cols

def _safe_name(s): return "".join(c if c.isalnum() or c in " _-" else "_" for c in s)

def _save_predictions_csv(base_dir, scenario_safe, run_no, embedding_tag, model_name,
                          topk_tag, sample_ids, y_true, y_pred, proba, classes):
    out_dir = os.path.join(base_dir, scenario_safe, f"run_{run_no:02d}")
    _ensure_dir(out_dir)
    fname = f"{embedding_tag}_{model_name}_{topk_tag}_run{run_no:02d}.csv"
    out_path = os.path.join(out_dir, fname)
    df = pd.DataFrame({
        'sample_id': pd.Series(sample_ids).astype(str).values,
        'true_label': pd.Series(y_true).astype(str).values,
        'pred_label': pd.Series(y_pred).astype(str).values,
    })
    if proba is not None and len(classes) > 0:
        for i, cls in enumerate(classes):
            df[f"proba_{str(cls)}"] = proba[:, i]
    df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions -> {out_path} ({len(df)} rows)")

def _fit_model_cv(Xtr, ytr, gtr, seed):
    pipe = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    )
    params = {"svc__C": [0.5, 1, 4], "svc__gamma": ["scale", 0.1, 0.01]}
    cvobj = _get_sgkf(n_splits=3, seed=seed)
    gs = GridSearchCV(pipe, params, cv=cvobj, scoring='balanced_accuracy',
                      verbose=1, n_jobs=1, refit=True)
    gs.fit(Xtr, ytr, groups=gtr)
    return gs.best_estimator_, gs.best_params_

def evaluate_with_mi_fs(df_pipeline, train_ids, test_ids, seed, scenario_safe, embedding_tag, run_no):
    # Split train/test
    train_df = df_pipeline[df_pipeline["sample_id"].isin(train_ids)]
    test_df = df_pipeline[df_pipeline["sample_id"].isin(test_ids)]
    Xtr, ytr, gtr, feat_cols = _to_xyg(train_df)
    Xte, yte, gte, _ = _to_xyg(test_df)

    # ======= FS per correlazione (NO sottocampionamento) =======
    if TOPK is None:
        # nessuna riduzione: tutte le feature
        Xtr_k, Xte_k = Xtr, Xte
        eff_k = Xtr.shape[1]
        topk_tag = "all"
        try:
            logger.info(f"[FS-CORR] skip (TOPK=None) -> using all features: {eff_k}")
        except Exception:
            pass
    else:
        selector, idx, sel_feat_names, eff_k, _scores = _corr_topk_from_train(Xtr, ytr, feat_cols, TOPK)
        Xtr_k = selector.transform(Xtr)
        Xte_k = selector.transform(Xte)
        topk_tag = f"top{eff_k}"
        try:
            logger.info(f"[FS-CORR] selected {eff_k}/{len(feat_cols)} features (k={TOPK})")
        except Exception:
            pass
    # ============================================================

    # CV + fit finale
    model, bestp = _fit_model_cv(Xtr_k, ytr, gtr, seed)

    # Test
    y_pred = model.predict(Xte_k)
    proba = model.predict_proba(Xte_k) if hasattr(model, "predict_proba") else None
    acc = balanced_accuracy_score(yte, y_pred, adjusted=True)

    # classi per colonne proba
    try:
        classes = model[-1].classes_ if hasattr(model[-1], 'classes_') else np.unique(ytr)
    except Exception:
        classes = np.unique(ytr)

    # Save predictions
    _save_predictions_csv(
        PRED_BASE_DIR, scenario_safe, run_no, embedding_tag, MODEL_NAME,
        topk_tag, gte, yte, y_pred, proba, classes
    )

    row = {
        "run": run_no, "seed": seed, "embedding": embedding_tag,
        "model": MODEL_NAME, "topk": topk_tag, "balanced_acc_adj": acc,
        "best_params": bestp, "n_features": eff_k
    }
    gc.collect()
    return row


### FEATURE SELELCTION
from time import perf_counter
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- helper: ColumnSelector (se già presente, puoi rimuovere questa classe) ---
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, idx):
        self.idx = np.array(idx, dtype=int)
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.idx]

def _y_to_binary(y):
    classes = []
    seen = set()
    for v in y:
        if v not in seen:
            classes.append(v); seen.add(v)
        if len(classes) == 2:
            break
    if len(classes) != 2:
        raise ValueError("Serve y binaria (2 classi).")
    m = {classes[0]: 0.0, classes[1]: 1.0}
    yb = np.array([m[v] for v in y], dtype=np.float64)
    return yb, np.array(classes, dtype=object)

# ===================== 1) CORRELAZIONE (SciPy point-biserial) =====================
def _corr_topk_from_train(Xtr, ytr, feat_names, k, log_topn: int = 10):
    """
    Ranking per |r| (point-biserial). Restituisce: selector, idx, sel_feat_names, eff_k, scores.
    Aggiunti log: inizio/fine, tempi, k richiesto/effettivo, top-10 per |r|.
    """
    t0 = perf_counter()
    yb, _ = _y_to_binary(ytr)
    n, p = Xtr.shape

    # log start
    try:
        logger.info(f"[FS-CORR] start | n={n} p={p} | k={k}")
    except Exception:
        print(f"[FS-CORR] start | n={n} p={p} | k={k}")

    scores = np.zeros(p, dtype=np.float64)
    for j in range(p):
        x = Xtr[:, j]
        # colonne costanti -> correlazione zero
        if np.all(x == x[0]) or np.std(x, ddof=1) == 0:
            scores[j] = 0.0
            continue
        r, _p = pointbiserialr(yb, x)  # Pearson tra binaria e continua
        scores[j] = 0.0 if not np.isfinite(r) else r

    order = np.argsort(np.abs(scores))[::-1]
    if (k is None) or (k >= p):
        idx = order; eff_k = p
    else:
        idx = order[:k]; eff_k = int(k)

    sel_feat_names = [feat_names[i] for i in idx]
    selector = ColumnSelector(idx)

    # log end + anteprima top feature
    dt = perf_counter() - t0
    try:
        logger.info(f"[FS-CORR] done  | eff_k={eff_k} | time={dt:.2f}s")
        topn = min(log_topn, eff_k)
        preview = ", ".join(f"{sel_feat_names[i]}(r={scores[idx[i]]:.4f})" for i in range(topn))
        logger.info(f"[FS-CORR] top{topn} by |r|: {preview}")
    except Exception:
        print(f"[FS-CORR] done  | eff_k={eff_k} | time={dt:.2f}s")

    return selector, idx, sel_feat_names, eff_k, scores

# ========== 2) LOGISTICA UNIVARIATA (ranking per |β| standardizzato) ==========
def _unilogit_topk_from_train(Xtr, ytr, feat_names, k, seed, log_topn: int = 10):
    """
    Fitta una logistica univariata per ciascuna feature (Pipeline: StandardScaler + LogisticRegression),
    rank per |β|. Restituisce: selector, idx, sel_feat_names, eff_k, betas.
    Aggiunti log: inizio/fine, tempi, k richiesto/effettivo, top-10 per |β|.
    """
    t0 = perf_counter()
    yb, _ = _y_to_binary(ytr)
    n, p = Xtr.shape

    try:
        logger.info(f"[FS-ULOG] start | n={n} p={p} | k={k} | seed={seed}")
    except Exception:
        print(f"[FS-ULOG] start | n={n} p={p} | k={k} | seed={seed}")

    betas = np.zeros(p, dtype=np.float64)
    for j in range(p):
        xj = Xtr[:, [j]]  # 2D
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=seed, max_iter=2000, solver='lbfgs', n_jobs=1)
        )
        try:
            pipe.fit(xj, yb)
            betas[j] = pipe[-1].coef_[0, 0]
        except Exception:
            betas[j] = 0.0

    order = np.argsort(np.abs(betas))[::-1]
    if (k is None) or (k >= p):
        idx = order; eff_k = p
    else:
        idx = order[:k]; eff_k = int(k)

    sel_feat_names = [feat_names[i] for i in idx]
    selector = ColumnSelector(idx)

    dt = perf_counter() - t0
    try:
        logger.info(f"[FS-ULOG] done  | eff_k={eff_k} | time={dt:.2f}s")
        topn = min(log_topn, eff_k)
        preview = ", ".join(f"{sel_feat_names[i]}(β={betas[idx[i]]:.4f})" for i in range(topn))
        logger.info(f"[FS-ULOG] top{topn} by |β|: {preview}")
    except Exception:
        print(f"[FS-ULOG] done  | eff_k={eff_k} | time={dt:.2f}s")

    return selector, idx, sel_feat_names, eff_k, betas


# Percentuale di training usata per imparare le feature (MI); None o 1.0 = tutto il train
FS_SUBSAMPLE_FRAC = 0.10   # 10%
# Se True, il sottocampione è scelto a livello di gruppo (sample_id) con stratifica per classe
FS_BY_GROUP = True

from sklearn.model_selection import train_test_split

def _make_fs_subsample_indices(y, g, frac: float, seed: int, by_group: bool = True):
    """
    Restituisce un boolean mask sugli esempi di training da usare per fittare il FS.
    Stratifica per classe; se by_group=True, stratifica a livello di sample_id.
    """
    if (frac is None) or (frac >= 1.0):
        return np.ones(len(y), dtype=bool)

    if by_group:
        # Stratifica a livello di sample_id: ricava una etichetta per ogni gruppo
        g = np.asarray(g)
        y = np.asarray(y)
        uniq_groups, inv = np.unique(g, return_inverse=True)
        # Assunzione: ogni sample_id ha una sola classe (coerente col tuo pipeline)
        # Se non fosse vero, prende la prima occorrenza.
        group_labels = np.zeros(len(uniq_groups), dtype=object)
        for i, gg in enumerate(uniq_groups):
            # prendi la classe del primo elemento di quel gruppo
            group_labels[i] = y[np.argmax(inv == i)]
        # pick una frazione di gruppi, stratificata per classe
        sel_groups, _ = train_test_split(
            uniq_groups,
            train_size=frac,
            random_state=seed,
            stratify=group_labels,
            shuffle=True
        )
        sel_groups = set(sel_groups.tolist())
        mask = np.array([gi in sel_groups for gi in g], dtype=bool)
        return mask
    else:
        # Stratifica direttamente sulle istanze
        y = np.asarray(y)
        idx = np.arange(len(y))
        sel_idx, _ = train_test_split(
            idx,
            train_size=frac,
            random_state=seed,
            stratify=y,
            shuffle=True
        )
        mask = np.zeros(len(y), dtype=bool)
        mask[sel_idx] = True
        return mask

def _mi_topk_from_train(Xtr, ytr, feat_names, k, seed):
    k_eff = 'all' if (k is None) else k
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, n_neighbors=3, random_state=seed),
        k=k_eff
    )
    selector.fit(Xtr, ytr)
    idx = selector.get_support(indices=True)
    sel_names = [feat_names[i] for i in idx]
    return selector, idx, sel_names, len(idx)

# ===================== Runner =====================
def train_on_saved_splits():
    splits_dir = os.path.join(OUT_DIR, "splits")
    scenario_safe = _safe_name(SCENARIO_NAME)
    common_path = os.path.join(splits_dir, f"common_samples_{scenario_safe}.csv")
    if not os.path.isfile(common_path):
        logger.error(f"Manca common samples: {common_path}"); return
    common_ids = set(pd.read_csv(common_path, usecols=["sample_id"])["sample_id"].astype("string").unique())
    limits = {"bac": LIMIT_BAC, "hum": LIMIT_HUM, "vir": LIMIT_VIR}

    df_vibe = _load_pipeline_filtered(VIBE_ROOT, KEEP_LABELS, common_ids, limits)
    #df_xvir = _load_pipeline_filtered(XVIR_ROOT, KEEP_LABELS, common_ids, limits)
    logger.info(f"[{SCENARIO_NAME}] df_vibe={df_vibe.shape}")

    per_run_files = sorted([fn for fn in os.listdir(splits_dir)
                            if fn.startswith(f"split_{scenario_safe}_run") and fn.endswith(".csv")])
    if not per_run_files:
        logger.error(f"Nessun file split per scenario '{SCENARIO_NAME}' in {splits_dir}"); return

    rows = []
    for fn in per_run_files:
        path = os.path.join(splits_dir, fn)
        split_df = pd.read_csv(path)
        run_no = int(split_df["run"].iloc[0]); seed = int(split_df["seed"].iloc[0])
        train_ids = set(split_df.loc[split_df["split"]=="train","sample_id"].astype("string"))
        test_ids  = set(split_df.loc[split_df["split"]=="test","sample_id"].astype("string"))

        logger.info(f"[{SCENARIO_NAME}] RUN {run_no} | seed={seed} | train={len(train_ids)} test={len(test_ids)}")

        for embedding_tag, df_pipe in (("ViBE", df_vibe),):
            if df_pipe.empty:
                logger.warning(f"Nessun dato per {embedding_tag}"); continue
            row = evaluate_with_mi_fs(df_pipe, train_ids, test_ids, seed, scenario_safe, embedding_tag, run_no)
            rows.append(row)
            gc.collect()

    if not rows:
        logger.error("Nessun risultato da salvare."); return

    res = pd.DataFrame(rows).sort_values(["embedding","run"])
    out_csv = os.path.join(OUT_DIR, f"mi_{MODEL_NAME}_{TOPK_TAG}_results_{scenario_safe}.csv")
    res.to_csv(out_csv, index=False)
    logger.info(f"[{SCENARIO_NAME}] Saved results: {out_csv}")

def main():
    setup_logger(os.path.join(OUT_DIR, f"{MODEL_NAME}_mi_{TOPK_TAG}.log"))
    train_on_saved_splits()
    logger.info("Done.")

if __name__ == "__main__":
    main()