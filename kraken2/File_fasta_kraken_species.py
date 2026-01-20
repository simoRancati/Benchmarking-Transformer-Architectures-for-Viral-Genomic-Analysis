#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import sys
from collections import defaultdict
import pandas as pd

###############################################################################
# PERCORSI (modifica se necessario)
###############################################################################
SPLITS_DIR = "/blue/simone.marini/share/Revision_IEEE/Results_Paired/splits"
FASTA_DIR  = "/blue/simone.marini/share/kraken2/Data"

FASTA_BY_LABEL = {
    "virus":    os.path.join(FASTA_DIR, "virus_kraken.fasta"),
    "human":    os.path.join(FASTA_DIR, "human_kraken.fasta"),
    "bacteria": os.path.join(FASTA_DIR, "bacteria_kraken.fasta"),
}

###############################################################################
# UTILS
###############################################################################
def wrap_seq(seq: str, width: int = 80):
    """Ritorna la sequenza in righe di larghezza fissa (per estetica FASTA)."""
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def read_needed_from_fasta(fasta_path: str, needed_ids: set):
    """
    Scansiona il FASTA e ritorna un dict {header -> sequence} SOLO per gli ID richiesti.
    L'header è tutto il testo dopo '>' (spazi inclusi), senza newline finale.
    """
    found = {}
    if not needed_ids:
        return found

    if not os.path.exists(fasta_path):
        print(f"[ERRORE] FASTA non trovato: {fasta_path}", file=sys.stderr)
        return found

    with open(fasta_path, "r", encoding="utf-8") as f:
        current_header = None
        seq_chunks = []

        def maybe_store():
            nonlocal current_header, seq_chunks
            if current_header is not None and current_header in needed_ids:
                found[current_header] = "".join(seq_chunks).replace("\n", "")
            # reset
            current_header = None
            seq_chunks = []

        for line in f:
            if line.startswith(">"):
                # chiudi record precedente
                maybe_store()
                current_header = line[1:].rstrip("\n").rstrip("\r")
            else:
                seq_chunks.append(line.strip())

        # ultimo record
        maybe_store()

    return found

def find_split_csvs(splits_dir: str):
    """
    Trova i CSV che finiscono con 'seed<number>.csv'.
    Esempi: *_seed2025.csv, something_seed7.csv, ecc.
    """
    pattern = os.path.join(splits_dir, "*seed*.csv")
    files = [p for p in glob.glob(pattern) if p.lower().endswith(".csv")]
    # Filtra solo quelli che terminano con 'seed<numero>.csv'
    out = []
    for p in files:
        base = os.path.basename(p)
        name_no_ext = os.path.splitext(base)[0]
        idx = name_no_ext.rfind("seed")
        if idx != -1:
            suffix = name_no_ext[idx+4:]  # dopo 'seed'
            if suffix.isdigit():
                out.append(p)
    return sorted(out)

###############################################################################
# MAIN
###############################################################################
def process_csv(csv_path: str):
    base = os.path.basename(csv_path)
    out_dir = os.path.dirname(csv_path)
    out_prefix = os.path.join(out_dir, os.path.splitext(base)[0])  # senza estensione

    # Leggi CSV
    try:
        df = pd.read_csv(csv_path, usecols=["sample_id", "label", "run", "split"])
    except ValueError:
        # In caso di colonne extra/ordine diverso, leggo tutto e poi seleziono
        df = pd.read_csv(csv_path)
        missing = [c for c in ["sample_id", "label", "run", "split"] if c not in df.columns]
        if missing:
            print(f"[ERRORE] {csv_path}: colonne mancanti {missing}", file=sys.stderr)
            return

    # Filtra TEST
    df_test = df[df["split"].astype(str).str.lower() == "test"].copy()
    if df_test.empty:
        print(f"[INFO] Nessun record 'test' in {base}. Salto.")
        return

    # Normalizza label
    df_test["label"] = df_test["label"].astype(str).str.lower().str.strip()

    # Salva CSV ridotto (sample_id, label) del test (utile per traccia)
    out_csv = f"{out_prefix}_test_only.csv"
    df_test[["sample_id", "label"]].to_csv(out_csv, index=False)

    # Prepara insiemi di ID per label
    ids_by_label = defaultdict(set)
    valid_labels = set(FASTA_BY_LABEL.keys())
    unknown_labels = set()

    for _, row in df_test.iterrows():
        lab = row["label"]
        sid = str(row["sample_id"])
        if lab in valid_labels:
            ids_by_label[lab].add(sid)
        else:
            unknown_labels.add(lab)

    if unknown_labels:
        print(f"[ATTENZIONE] {base}: trovate label sconosciute: {sorted(unknown_labels)} (ignorate).", file=sys.stderr)

    # Leggi dai FASTA solo ciò che serve, per ogni label
    seqs_by_label = {}
    for lab, wanted in ids_by_label.items():
        fasta_path = FASTA_BY_LABEL[lab]
        print(f"[INFO] {base}: leggo {len(wanted)} ID da {os.path.basename(fasta_path)} …")
        seqs_by_label[lab] = read_needed_from_fasta(fasta_path, wanted)

    # Scrittura FASTA separati per ciascuna label, rispettando l'ordine del CSV test
    for lab in sorted(ids_by_label.keys()):
        out_fasta_lab = f"{out_prefix}_{lab}.fasta"
        not_found = []

        with open(out_fasta_lab, "w", encoding="utf-8") as out_f:
            # scrivo solo le righe del test con quella label, in ordine
            subset = df_test[df_test["label"] == lab]
            for _, row in subset.iterrows():
                sid = str(row["sample_id"])
                seq = seqs_by_label.get(lab, {}).get(sid)
                if seq:
                    out_f.write(f">{sid}\n{wrap_seq(seq)}\n")
                else:
                    not_found.append(sid)

        if not_found:
            print(f"[ATTENZIONE] {base} [{lab}]: {len(not_found)} sample_id non trovati nei FASTA (es. {not_found[:5]}).", file=sys.stderr)
        print(f"[OK] {base}: scritto FASTA label-specifico -> {out_fasta_lab}")

    print(f"[OK] {base}: CSV test ridotto -> {out_csv}")

def main():
    csv_paths = find_split_csvs(SPLITS_DIR)
    if not csv_paths:
        print(f"[ERRORE] Nessun file trovato in {SPLITS_DIR} con pattern 'seed<number>.csv'", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Trovati {len(csv_paths)} file da processare.")
    for p in csv_paths:
        try:
            process_csv(p)
        except Exception as e:
            print(f"[ERRORE] Elaborando {p}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()