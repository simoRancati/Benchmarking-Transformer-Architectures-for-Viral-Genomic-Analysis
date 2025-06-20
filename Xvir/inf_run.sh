#!/bin/bash
#SBATCH --account=boucher 
#SBATCH --qos=boucher
#SBATCH --partition=gpu
#SBATCH --job-name=hpv_preds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sakshi.pandey@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128gb
#SBATCH --time=40:00:00
#SBATCH --output=logs/%j_hpv.log
#SBATCH --error=logs/%j_hpv.log
#SBATCH --gpus=a100:1
##----------------------------------------------------------

for file in /blue/simone.marini/share/Embedder_Benchmarking/Data/Illumina_Fasta/per_genome_reads/merged_fastas_hpv/*_merged.fasta; do
    echo "Processing $file"
    python inference.py \
        --model_path ./logs/XVir_150bp_model.pt \
        --input "$file" \
        --read_len 150 \
        --ngram 6 \
        --model_dim 128 \
        --num_layers 1 \
        --batch_size 100 \
        --cuda
done

