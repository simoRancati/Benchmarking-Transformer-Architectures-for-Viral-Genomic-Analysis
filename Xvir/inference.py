# inference.py
import csv
from model import XVir
from argparse import ArgumentParser
from utils.dataset import inferenceDataset
import torch
import numpy as np
from tqdm import tqdm

def parse_fasta(filename, read_len=150):
    base2int = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    reads = []
    labels = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:  # EOF
                break
            elif line.startswith('>'):  # Parse sequence data
                read_header = line[1:].rstrip()
                read_seq = f.readline().rstrip().upper()
                if 'N' in read_seq:
                    continue
                read_seq = read_seq[:read_len]  # Trim to 150 bp
                if len(read_seq) != read_len:
                    continue
                try:
                    read_seq = np.array([base2int[base] for base in read_seq])
                except KeyError:
                    continue  # Skip if an unexpected base is encountered
                labels.append(read_header)
                reads.append(read_seq)
    print([len(seq) for seq in reads])
    return np.array(reads), np.array(labels)

def main(args):
    # Load model
    model = XVir(args.read_len, args.ngram, args.model_dim, args.num_layers, 0.1)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    print("Model loaded")

    if args.cuda:
        model = model.to('cuda')

    # Parse FASTA
    reads, labels = parse_fasta(args.input, read_len=args.read_len)
    dataset = inferenceDataset(args, reads, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Fasta loaded, Predicting...")
    print(f"Loaded {len(reads)} reads from {args.input}")

    # Prepare CSV for writing embeddings
    csv_filename = args.input + '_embeddings.csv'
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row: read_id + each embedding dimension
        # Because the penultimate-layer is (batch_size, input_dim, model_dim),
        # flattened it becomes input_dim * model_dim dimensions.
        num_embed_dims = (args.read_len - args.ngram + 1) * args.model_dim
        header = ['read_id'] + [f'embedding_{i}' for i in range(num_embed_dims)]
        writer.writerow(header)

        # Inference loop
        preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                if args.cuda:
                    x = x.to('cuda')

                # Get embeddings and logits
                x_enc, outputs = model(x)
                # outputs has shape (batch_size, 1); let's get probabilities
                preds_batch = torch.sigmoid(outputs.detach().cpu()).squeeze(-1).numpy()

                # Move embeddings to CPU to flatten them
                x_enc_cpu = x_enc.detach().cpu().numpy()  # shape (batch_size, input_dim, model_dim)

                # For each read in this batch, flatten and write to CSV
                for i, label_id in enumerate(y):
                    # Flatten from (input_dim, model_dim) to a 1D array
                    embedding_flat = x_enc_cpu[i].reshape(-1)
                    row_data = [label_id] + embedding_flat.tolist()
                    writer.writerow(row_data)

                all_labels.extend(y)
                preds.extend(preds_batch)

    # Also store final predictions in a text file if desired
    output_txt = args.input + '.output.txt'
    with open(output_txt, 'w') as f:
        for read_header, score in zip(all_labels, preds):
            f.write(f'{read_header}\t{score}\n')

    print(f"Embeddings saved to {csv_filename}")
    print(f"Predictions saved to {output_txt}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--read_len', type=int, default=150)
    parser.add_argument('--ngram', type=int, default=6)
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
