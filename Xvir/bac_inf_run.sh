#!/bin/bash
#SBATCH --account=simone.marini  
#SBATCH --qos=simone.marini 
#SBATCH --partition=gpu
#SBATCH --job-name=xvir_bac_preds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sakshi.pandey@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128gb
#SBATCH --time=40:00:00
#SBATCH --output=logs/%j_xvirbac.log
#SBATCH --error=logs/%j_xvirbac.log
#SBATCH --gpus=a100:1
##----------------------------------------------------------

DATA_DIR="/blue/simone.marini/share/Embedder_Benchmarking/Data/Illumina_Fasta/illumina_fasta/Bacteria/Illumina/merged_fastas_bacteria"

for file in "$DATA_DIR"/*_merged.fasta; do
    base=$(basename "$file")                
    prefix="${base%.fasta}"                
    embed_file="$DATA_DIR/${prefix}.fasta_embeddings.csv"

    echo "-----------------------------------------"
    echo "Checking for embeddings:"
    echo "Fasta file:   $file"
    echo "Embed file:   $embed_file"

    if [ -f "$embed_file" ]; then
        echo "Embeddings exist => skipping inference."
        continue
    else
        echo "No embeddings found => running inference."
    fi

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
