from pyfaidx import Fasta
import pandas as pd
from collections import defaultdict
import os
import ipdb
from convert_ncbi2ucsc import load_chrom_mapping_genbank
from split_data import split_data

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract sequences from reference genome")
    parser.add_argument("--assembly_report_path", type=str, default="./LONG-BENCH-supp/GCA_000001405.15_GRCh38_assembly_report.txt", help="Path to assembly report file")
    parser.add_argument("--genome_fasta", type=str, default="./LONG-BENCH-supp/GCA_000001405.15_GRCh38_genomic.fna", help="Path to genome FASTA file")
    parser.add_argument("--variant_csv", type=str, required=True, help="Path to variant CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for CSV files")
    parser.add_argument("--window_size", type=int, default=5000, help="Window size for sequence extraction")
    return parser.parse_args()
# # ---- 参数设置 ----
# assembly_report_path = "./LONG-BENCH-supp/GCA_000001405.15_GRCh38_assembly_report.txt"
# genome_fasta = "./LONG-BENCH-supp/GCA_000001405.15_GRCh38_genomic.fna"
# variant_csv = "./LONG-BENCH/variant_effect_causal_eqtl/All_Tissues.csv"
# output_dir = "./LONG-BENCH-process/variant_effect_causal_eqtl"
# window_size = 5000

def main():
    args = parse_args()
    assembly_report_path = args.assembly_report_path
    genome_fasta = args.genome_fasta
    variant_csv = args.variant_csv
    output_dir = args.output_dir
    window_size = args.window_size

    
    chrom_map = load_chrom_mapping_genbank(assembly_report_path)
    allowed_contigs = set(chrom_map.values())

    
    genome = Fasta(genome_fasta, rebuild=True)

    
    df = pd.read_csv(variant_csv)
    output = defaultdict(list)

    for _, row in df.iterrows():
        chrom_raw = row["CHROM"]
        
        if row.get("label") is not None:
            label = row.get("label") 
        else:
            label = row.get("INT_LABEL")
        
        split = row["split"]
        chrom = chrom_map.get(chrom_raw, chrom_raw)
        # ipdb.set_trace()
        if chrom not in genome or chrom not in allowed_contigs:
            continue
        
        make_window = False
        if row.get("POS") and row.get('START') is None and row.get('STOP') is None:
            pos = int(row["POS"])
            ref = row["REF"]
            alt = row["ALT"]
            start = pos - 1 - window_size
            end = pos - 1 + window_size
            make_window = True
        elif row.get("START") and row.get('STOP'):
            start = int(row["START"]) - 1 
            end = int(row["STOP"]) - 1
            
        if start < 1 or end > len(genome[chrom]):
            continue

        try:
            if make_window:
                if genome[chrom][pos-1].seq.upper() != ref:
                    print(f"Warning: Reference mismatch at {chrom}:{pos} (expected {ref}, found {genome[chrom][pos].seq.upper()})")
                    continue
            seq = genome[chrom][start:end].seq.upper()
            # ipdb.set_trace()
            if make_window:
                if len(seq) != 2 * window_size:
                    continue
                mut_seq = seq[:window_size] + alt + seq[window_size + 1:]
            else:
                # ipdb.set_trace()
                if len(seq) != end - start:
                    continue
                mut_seq = seq
            output[split].append((mut_seq, label))
        except Exception:
            continue

    # 保存
    output_df = {}
    for split, records in output.items():
        df_split = pd.DataFrame(records, columns=["sequence", "label"])
        output_df[split] = df_split
    
    # ipdb.set_trace()
    if ( ('dev' not in output_df) or ('valid' not in output_df) ) and 'train' in output_df:
        # ipdb.set_trace()
        output_df['train'], output_df['dev'] = split_data(output_df['train'],output_df["train"]['label'], output_dir, test_size=0.2, random_state=42)
    # ipdb.set_trace()
    for split, records in output_df.items():
        output_csv = os.path.join(output_dir, f"{split}.csv")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        records.to_csv(output_csv, index=False)
        print(f"output file: {output_csv}, contain {len(records)} records.")


if __name__ == "__main__":
    main()