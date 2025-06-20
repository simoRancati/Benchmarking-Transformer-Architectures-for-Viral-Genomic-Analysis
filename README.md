# Benchmarking-Transformer-Architectures-for-Viral-Genomic-Analysis

# Introduction
This repository contains the code and data to reproduce the first read-level benchmark (2 × 150 bp) for evaluating transformer-based models in metagenomic virome analysis.

-Dataset – 5,520 paired-end Illumina reads simulated from 184 HPV genomes, 184 RefSeq bacterial genomes, and 184 human assemblies, with uniform coverage and fragment length distributions.
-Models evaluated – ViBE (12-layer BERT, 768-dim embeddings) and XVir (HPV-optimized encoder, 18,560-dim embeddings).
-Key results:
  1)ViBE outperformed XVir with higher UMAP silhouette scores (+0.18), greater centroid separations (3 out of 4 tests), and 0.97–0.98 accuracy across all classification tasks (Human vs Viral, Viral vs Bacterial, Human vs Bacterial), using just 1.6 GB GPU RAM per 128 reads.
  2)XVir matched ViBE on Human vs Viral (0.98) but dropped by up to 14 percentage points in other comparisons, with 6.7 GB GPU RAM usage.
-Why this matters – The provided code enables transparent, reproducible evaluation of embedding models on short-read viral metagenomics, revealing ViBE’s general versatility and XVir’s niche strengths
