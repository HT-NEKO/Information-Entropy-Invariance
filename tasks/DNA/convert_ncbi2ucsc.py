# 脚本功能：从 NCBI GRCh38 的 assembly report 文件中读取 CHROM -> Contig 映射
# 文件通常名为 GCA_000001405.15_GRCh38_assembly_report.txt
import ipdb
def load_chrom_mapping_genbank(assembly_report_path, start_with="CM"):
    mapping = {}
    with open(assembly_report_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            # parts[0] = sequence name (GenBank: CM000663.2)
            # parts[2] = assigned-molecule (e.g., 1, 2, X, Y, MT)
            # parts[3] = assigned-molecule-location-type (assembled/scaffold)
            genbank_id = parts[4]  # GenBank accession: CM000663.2
            alias = parts[2]
            if alias and genbank_id.startswith(start_with):
                mapping[f"chr{alias}"] = genbank_id
    return mapping


# 示例用法（路径请替换为你本地的 assembly report 文件路径）
assembly_report_file = "./LONG-BENCH-supp/GCA_000001405.15_GRCh38_assembly_report.txt"
chrom_map = load_chrom_mapping_genbank(assembly_report_file)

# 显示前几个映射对
list(chrom_map.items())[:5]
# ipdb.set_trace()