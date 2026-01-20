# Viral Read Classification with Transformer Embeddings (ViBE, XVir) and Kraken-2

This repository accompanies a read-level **classification** study on detecting viral sequences in realistic metagenomic settings with overwhelming human and bacterial background. We evaluate how two viral sequence transformer encoders—**ViBE** (hierarchical BERT-style encoder trained on diverse eukaryotic viruses) and **XVir** (HPV-focused encoder)—behave on **150-bp Illumina/NovaSeq-like paired-end reads**, and we compare them to the k-mer classifier **Kraken-2**.

## Contents

- [Overview](#overview)
- [Classification tasks](#classification-tasks)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contacts](#contacts)
- [License](#license)

## Overview

Alignment-free k-mer methods (e.g., Kraken-2) remain widely used for viral screening, but exact k-mer matching against static reference panels can reduce sensitivity to novel or highly divergent viruses. Transformer language models can provide contextual sequence representations that may mitigate these limitations. Here, we study whether compact transformer embeddings can support accurate and scalable **viral read classification** under realistic Illumina error profiles and strong background contamination.

## Classification tasks

We report results for four virus-centric tasks:

1. **Virus vs Human**
2. **Virus vs Bacterial**
3. **Human vs Bacterial** (background control)
4. **Virus vs Human vs Bacterial** (three-way setting)

## Dataset

We simulate **paired-end 150-bp** reads using an Illumina **NovaSeq** error model. The reference panel includes:

- **184 HPV genomes**
- **147 RefSeq bacterial genomes**
- **62 human assemblies**

The full dataset contains **~1.2M read pairs**.

## Models

### ViBE
- Hierarchical BERT-style encoder trained on diverse eukaryotic viruses
- Produces compact per-read embeddings (e.g., **768-D** CLS-style representation, depending on checkpoint/config)

### XVir
- HPV-focused encoder
- Produces higher-dimensional read representations (dimension depends on export strategy/checkpoint)

### Kraken-2
- k-mer based classifier used as a strong alignment-free baseline

## Evaluation

We evaluate representations and classifiers in two complementary ways:

### 1) Embedding geometry
- **PCA / UMAP** visualization
- Quantitative structure/separation metrics:
  - **Silhouette score**
  - **Rescaled centroid distances** between classes

### 2) Lightweight downstream classifiers
We train small classifiers on top of fixed embeddings:
- **Logistic Regression**
- **Random Forest**

To assess compressibility/scalability, we repeat classification using:
- Full embeddings
- **Top-500 features**
- **Top-100 features**

Feature selection is performed **only on training data** (e.g., via univariate correlation / relevance scoring), then applied to the held-out test set.

## Repository structure

## Installation

### Python
Create an environment (conda or venv) and install core dependencies:

- Python **>= 3.10**
- `numpy`, `pandas`, `scikit-learn`
- `umap-learn`
- `matplotlib` (and optionally `seaborn`)
- `torch` (CUDA recommended for embedding extraction)
- `tqdm`, `pyyaml` (optional, if used by your scripts)

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn umap-learn matplotlib torch tqdm pyyaml
```

### External tool 
-InSilicoSeq (read simulation)

-Kraken-2 (k-mer classification)
### Quickstart
-Prepare reference FASTA files

-Place HPV, bacterial, and human FASTAs

-Simulate reads (paired-end 150 bp, NovaSeq profile)

-Output to data/reads/

-Create train/test splits

-Save split manifests to data/splits/ (e.g., per-species stratified 80/20)

-Extract embeddings

-Export per-read embeddings for ViBE and XVir to embeddings/vibe/ and embeddings/xvir/

-Run geometry analysis

-Produce PCA/UMAP plots and compute silhouette/centroid metrics under results/geometry/

-Train and evaluate classifiers

-Fit Logistic Regression / Random Forest models for each task and export metrics to results/classification/

-Run Kraken-2

-Classify held-out reads with Kraken-2 and map predictions to the same four tasks for comparison
