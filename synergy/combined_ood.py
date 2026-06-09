import argparse
import copy
import hashlib
import random
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
import torch_geometric

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synergy.combined_raw import (
    build_dsp_text,
    clean_smiles,
    collect_ddi_rows,
    ensure_item_dir,
    get_gene_tensor,
    load_gene_table,
    mol_to_graph_data_obj_simple,
    save_text,
)


class GraphCache:
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, smiles: str):
        graph = self.cache.get(smiles)
        if graph is not None:
            self.cache.move_to_end(smiles)
            return copy.deepcopy(graph)
        graph = mol_to_graph_data_obj_simple(smiles)
        graph.deratio = degree_distribution(graph)
        self.cache[smiles] = graph
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return copy.deepcopy(graph)


def degree_distribution(graph):
    edges = graph.edge_index[1]
    if edges.numel() == 0:
        return torch.tensor([0.0, 0.0, 0.0])
    degrees = torch_geometric.utils.degree(edges).to(torch.long).cpu().numpy().tolist()
    return torch.tensor([degrees.count(i) for i in range(1, 4)], dtype=torch.float) / graph.num_nodes


def gene_env_vector(row, genes, cell_num_features: int):
    gene_tensor = get_gene_tensor(row, genes, cell_num_features).to(torch.float)
    if gene_tensor.numel() >= 2:
        values = gene_tensor[:2]
        scale = values.abs().mean().clamp_min(1.0)
        return values / scale
    return torch.zeros(2, dtype=torch.float)


def add_dsp_environment(graph, env_vec):
    transformed = copy.deepcopy(graph)
    env = env_vec.to(torch.float).repeat(graph.num_nodes, 1)
    transformed.x = torch.cat([env, graph.x.to(torch.float)], dim=1)
    return transformed


def hashed_target_embeddings(targets, target_dim: int):
    if pd.isna(targets):
        return torch.empty(0)
    names = [name.strip().lower() for name in str(targets).split(";") if name.strip()]
    if not names:
        return torch.empty(0)
    embeddings = []
    for name in names:
        vec = torch.zeros(target_dim, dtype=torch.float)
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        for offset, byte in enumerate(digest):
            idx = (byte + offset * 257) % target_dim
            vec[idx] += 1.0
        norm = vec.norm().clamp_min(1.0)
        embeddings.append(vec / norm)
    return torch.stack(embeddings, dim=0)


def collect_dsp_rows_with_targets(dsp_path: Path, max_rows: int = 0):
    df = pd.read_csv(dsp_path)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=1)

    rows = []
    for _, row in df.iterrows():
        smiles1 = clean_smiles(row.get("Drug1", ""))
        smiles2 = clean_smiles(row.get("Drug2", ""))
        if not smiles1 or not smiles2:
            continue
        try:
            score = float(row.get("value", 0.0))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "task": "DSP",
                "smiles1": smiles1,
                "smiles2": smiles2,
                "text": build_dsp_text(score),
                "target1": row.get("Drug1_Targets", ""),
                "target2": row.get("Drug2_Targets", ""),
                "source": row,
            }
        )
    return rows


def write_split(split_root: Path, rows, graph_cache: GraphCache, genes, cell_num_features: int, target_dim: int):
    kept = 0
    skipped = 0
    for row in rows:
        try:
            graph1 = graph_cache.get(row["smiles1"])
            graph2 = graph_cache.get(row["smiles2"])
        except Exception as exc:
            skipped += 1
            print(f"Skip {row['task']} sample due to graph build failure: {exc}")
            continue

        data1 = {"Valid": True, "Graph": graph1}
        data2 = {"Valid": True, "Graph": graph2}
        if row["task"] == "DSP":
            env_vec = gene_env_vector(row["source"], genes, cell_num_features)
            data1["Transform"] = add_dsp_environment(graph1, env_vec)
            data2["Transform"] = add_dsp_environment(graph2, env_vec)
            data1["Target"] = hashed_target_embeddings(row.get("target1", ""), target_dim)
            data2["Target"] = hashed_target_embeddings(row.get("target2", ""), target_dim)

        idx = kept
        graph1_dir = ensure_item_dir(split_root, "graph1", idx)
        graph2_dir = ensure_item_dir(split_root, "graph2", idx)
        smiles1_dir = ensure_item_dir(split_root, "smiles1", idx)
        smiles2_dir = ensure_item_dir(split_root, "smiles2", idx)
        text_dir = ensure_item_dir(split_root, "text", idx)
        task_dir = ensure_item_dir(split_root, "task", idx)
        genes_dir = ensure_item_dir(split_root, "genes", idx)

        torch.save(data1, graph1_dir / "graph_data.pt")
        torch.save(data2, graph2_dir / "graph_data.pt")
        save_text(smiles1_dir / "text.txt", row["smiles1"])
        save_text(smiles2_dir / "text.txt", row["smiles2"])
        save_text(text_dir / "text.txt", row["text"])
        save_text(task_dir / "text.txt", row["task"])

        if row["task"] == "DSP":
            gene_tensor = get_gene_tensor(row["source"], genes, cell_num_features)
        else:
            gene_tensor = torch.empty(0)
        torch.save(gene_tensor, genes_dir / "gene_data.pt")
        kept += 1
    return kept, skipped


def process_combined_ood(
    ddi_path: Path = Path("data/raw/drugbank_merged.csv"),
    dsp_path: Path = Path("data/raw/drugcomb_bliss.csv"),
    output_root: Path = Path("data/combined_ood"),
    genes_path: Path = Path("dicts/df_rma_landm.tsv"),
    seed: int = 42,
    max_ddi: int = 0,
    max_dsp: int = 0,
    valid_size: int = 5000,
    test_size: int = 5000,
    cell_num_features: int = 908,
    target_dim: int = 1280,
):
    if output_root.exists():
        shutil.rmtree(output_root)

    ddi_rows = collect_ddi_rows(ddi_path, max_ddi)
    dsp_rows = collect_dsp_rows_with_targets(dsp_path, max_dsp)
    rows = ddi_rows + dsp_rows
    random.Random(seed).shuffle(rows)

    genes = load_gene_table(genes_path)
    graph_cache = GraphCache()

    test_start = min(valid_size, len(rows))
    splits = [
        ("train", rows),
        ("valid", rows[: min(valid_size, len(rows))]),
        ("test", rows[test_start : test_start + min(test_size, max(0, len(rows) - test_start))]),
    ]
    for split_name, split_rows in splits:
        kept, skipped = write_split(
            output_root / split_name,
            split_rows,
            graph_cache,
            genes,
            cell_num_features,
            target_dim,
        )
        print(f"{split_name}: kept={kept} skipped={skipped}")

    print(f"Output folder: {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddi_path", type=Path, default=Path("data/raw/drugbank_merged.csv"))
    parser.add_argument("--dsp_path", type=Path, default=Path("data/raw/drugcomb_bliss.csv"))
    parser.add_argument("--output_root", type=Path, default=Path("data/combined_ood"))
    parser.add_argument("--genes_path", type=Path, default=Path("dicts/df_rma_landm.tsv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_ddi", type=int, default=0)
    parser.add_argument("--max_dsp", type=int, default=0)
    parser.add_argument("--valid_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--cell_num_features", type=int, default=908)
    parser.add_argument("--target_dim", type=int, default=1280)
    args = parser.parse_args()
    process_combined_ood(**vars(args))


if __name__ == "__main__":
    main()
