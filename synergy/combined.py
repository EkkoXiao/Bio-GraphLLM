import math
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data


ALLOWABLE_FEATURES = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


class GraphCache:
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, smiles: str) -> Data:
        graph = self.cache.get(smiles)
        if graph is not None:
            self.cache.move_to_end(smiles)
            return graph
        graph = mol_to_graph_data_obj_simple(smiles)
        self.cache[smiles] = graph
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        return graph


def mol_to_graph_data_obj_simple(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            ALLOWABLE_FEATURES["possible_atomic_num_list"].index(atom.GetAtomicNum()),
            ALLOWABLE_FEATURES["possible_chirality_list"].index(atom.GetChiralTag()),
        ]
        atom_features_list.append(atom_feature)

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                ALLOWABLE_FEATURES["possible_bonds"].index(bond.GetBondType()),
                ALLOWABLE_FEATURES["possible_bond_dirs"].index(bond.GetBondDir()),
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def calculate_bounds(number: float):
    lower_bound = math.floor(number * 2) / 2
    upper_bound = math.ceil(number * 2) / 2
    return lower_bound, upper_bound


def safe_text(value) -> str:
    if pd.isna(value):
        return "None"
    text = str(value).strip()
    if not text or text == "Not Available":
        return "None"
    return text


def clean_smiles(value) -> str:
    text = safe_text(value)
    if text == "None":
        return ""
    if ";" in text:
        text = text.split(";")[-1].strip()
    return text


def ensure_item_dir(root: Path, folder: str, idx: int) -> Path:
    path = root / folder / str(idx)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_dsp_text(score: float) -> str:
    lower, upper = calculate_bounds(abs(score))
    prefix = "Yes." if score > 0 else "No."
    return (
        f"{prefix} The absolute value is above {lower} and below {upper}, "
        f"thus the accurate value is {abs(score):.2f}."
    )


def save_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_cosmic_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_gene_table(genes_path: Path):
    if not genes_path.exists():
        print(f"Gene table not found: {genes_path}. Missing DSP genes will be zero vectors.")
        return None
    return pd.read_csv(genes_path, sep="\t")


def get_gene_tensor(row: pd.Series, genes: pd.DataFrame, cell_num_features: int):
    if genes is None:
        return torch.zeros(cell_num_features, dtype=torch.float)
    cosmic_id = normalize_cosmic_id(row.get("Cosmic_ID", ""))
    col = f"DATA.{cosmic_id}" if cosmic_id else ""
    if not col or col not in genes.columns:
        return torch.zeros(cell_num_features, dtype=torch.float)
    return torch.tensor(genes[col].astype(float).values, dtype=torch.float)


def collect_ddi_rows(ddi_path: Path, max_rows: int = 0):
    df = pd.read_csv(ddi_path)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=0)

    rows = []
    for _, row in df.iterrows():
        smiles1 = clean_smiles(row.get("Drug1_smiles", ""))
        smiles2 = clean_smiles(row.get("Drug2_smiles", ""))
        text = safe_text(row.get("Interaction", ""))
        if not smiles1 or not smiles2 or text == "None":
            continue
        rows.append(
            {
                "task": "DDI",
                "smiles1": smiles1,
                "smiles2": smiles2,
                "text": text,
                "source": row,
            }
        )
    return rows


def collect_dsp_rows(dsp_path: Path, max_rows: int = 0):
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
                "source": row,
            }
        )
    return rows


def write_split(split_root: Path, rows, graph_cache: GraphCache, genes, cell_num_features: int):
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

        idx = kept
        graph1_dir = ensure_item_dir(split_root, "graph1", idx)
        graph2_dir = ensure_item_dir(split_root, "graph2", idx)
        smiles1_dir = ensure_item_dir(split_root, "smiles1", idx)
        smiles2_dir = ensure_item_dir(split_root, "smiles2", idx)
        text_dir = ensure_item_dir(split_root, "text", idx)
        task_dir = ensure_item_dir(split_root, "task", idx)
        genes_dir = ensure_item_dir(split_root, "genes", idx)

        torch.save({"Valid": True, "Graph": graph1}, graph1_dir / "graph_data.pt")
        torch.save({"Valid": True, "Graph": graph2}, graph2_dir / "graph_data.pt")
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
        if kept % 5000 == 0:
            print(f"{split_root.name}: wrote {kept} samples")
    return kept, skipped


def process_combined(
    ddi_path: Path = Path("data/raw/drugbank_merged.csv"),
    dsp_path: Path = Path("data/raw/drugcomb_bliss.csv"),
    output_root: Path = Path("data/combined_raw"),
    genes_path: Path = Path("dicts/df_rma_landm.tsv"),
    seed: int = 42,
    max_ddi: int = 0,
    max_dsp: int = 0,
    valid_size: int = 5000,
    test_size: int = 5000,
    cell_num_features: int = 908,
):
    if output_root.exists():
        shutil.rmtree(output_root)

    ddi_rows = collect_ddi_rows(ddi_path, max_ddi)
    dsp_rows = collect_dsp_rows(dsp_path, max_dsp)
    rows = ddi_rows + dsp_rows
    random.Random(seed).shuffle(rows)

    print(f"Collected {len(ddi_rows)} DDI rows and {len(dsp_rows)} DSP rows.")
    print(f"Mixed total before graph validation: {len(rows)}")

    genes = load_gene_table(genes_path)
    graph_cache = GraphCache()

    valid_start = 0
    test_start = min(valid_size, len(rows))
    splits = [
        ("train", rows),
        ("valid", rows[valid_start : valid_start + min(valid_size, len(rows))]),
        ("test", rows[test_start : test_start + min(test_size, max(0, len(rows) - test_start))]),
    ]
    for split_name, split_data in splits:
        kept, skipped = write_split(output_root / split_name, split_data, graph_cache, genes, cell_num_features)
        print(f"{split_name}: kept={kept} skipped={skipped}")

    print(f"Output folder: {output_root}")


if __name__ == "__main__":
    process_combined()
