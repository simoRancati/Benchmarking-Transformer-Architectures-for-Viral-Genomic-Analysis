# Single-read viral transformers,exploring transformer architectures for viral metagenomic analysis at the single-read level

# 🔬 Short-read Viral Embedding Evaluation

This repository contains the **code** and **data** to reproduce the **read-level evaluation (2 × 150 bp)** for evaluating transformer-based models in metagenomic virome analysis.

---

## 📚 Dataset

5,520 paired-end Illumina reads simulated from:
- 184 **HPV genomes**
- 147 **RefSeq bacterial genomes**
- 62 **human genome assemblies**

All reads have uniform coverage and fragment length distributions.

---

## 🧠 Models Evaluated

- **ViBE** – 12-layer BERT model trained for broad eukaryotic virus detection, outputs 768-dimensional embeddings.
- **XVir** – HPV-optimized encoder, outputs 18,560-dimensional embeddings.

---

## 📊 Key Results

1. **ViBE** outperformed XVir:
   - Higher **UMAP silhouette scores** (+0.18)
   - Greater **centroid separations** (3 out of 4 pairwise tests)
   - **0.97–0.98 accuracy** in all classification tasks (Human vs Viral, Viral vs Bacterial, Human vs Bacterial)
   - Efficient: only **1.6 GB GPU RAM** per 128 reads

2. **XVir**:
   - Matched ViBE on **Human vs Viral** (0.98)
   - Dropped up to **14 percentage points** in other tasks
   - Required **6.7 GB GPU RAM** per 128 reads

---

## ❗ Why This Matters

The provided **code** enables transparent and reproducible evaluation of embedding models for **short-read viral metagenomics**.  
It reveals:
- **ViBE** as a general-purpose, resource-efficient model
- **XVir** as highly specialized for HPV
- The need for realistic, multi-platform benchmarks and alignment-free baselines (e.g., Kraken 2)

---
