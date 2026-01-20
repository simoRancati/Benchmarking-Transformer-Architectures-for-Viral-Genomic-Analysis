#!/usr/bin/env python3
import os
import shlex
import subprocess
from pathlib import Path

# === Configurazione ===
INPUT_DIR = Path("/blue/simone.marini/share/Revision_IEEE/Results_Paired/splits")
RESULTS_DIR = Path("/blue/simone.marini/share/kraken2/Results")
DB_PATH = Path("/data/reference/kraken2/standard_20220926")

# Opzioni kraken2 come da esempio
KRAKEN2_OPTS = "--use-names --confidence 0.1"

def main():
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Cartella input non trovata: {INPUT_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(INPUT_DIR.glob("*.fasta"))
    if not fasta_files:
        raise SystemExit(f"Nessun file .fasta trovato in {INPUT_DIR}")

    print(f"Trovati {len(fasta_files)} FASTA in {INPUT_DIR}")
    print(f"I risultati saranno salvati in {RESULTS_DIR}\n")

    for fasta in fasta_files:
        base = fasta.stem  # es: "split_Human vs Virus_run9_seed2033_virus"
        report_path = RESULTS_DIR / f"{base}.report"
        output_path = RESULTS_DIR / f"{base}.kraken"

        # Costruisco il comando da eseguire in bash -lc per poter usare `ml`
        cmd = (
            "ml kraken/2.1.3 && "
            "kraken2 "
            f"--db {shlex.quote(str(DB_PATH))} "
            f"{KRAKEN2_OPTS} "
            f"--report {shlex.quote(str(report_path))} "
            f"--output {shlex.quote(str(output_path))} "
            f"{shlex.quote(str(fasta))}"
        )

        print(f"Eseguo:\n{cmd}\n")
        try:
            # Usiamo bash -lc per caricare il modulo (Lmod) e poi lanciare kraken2
            subprocess.run(["bash", "-lc", cmd], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERRORE] kraken2 fallito per {fasta.name} (codice {e.returncode}).")
            # Continua con i successivi senza interrompere tutto
            continue

    print("Completato.")

if __name__ == "__main__":
    main()
