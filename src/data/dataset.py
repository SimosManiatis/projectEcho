# data/dataset.py

import os
import glob
import json
import logging
import torch
import dgl

from config import INPUT_FEATURE_NAMES, OUTPUT_FEATURE_NAMES

logger = logging.getLogger(__name__)

def load_json_files(json_folder: str) -> list:
    """Load all JSON design files from a folder."""
    logger.info(f"Loading JSON files from {json_folder}")
    if not os.path.isdir(json_folder):
        logger.error(f"Folder not found: {json_folder}")
        return []
    paths = glob.glob(os.path.join(json_folder, "*.json"))
    logger.info(f"Found {len(paths)} JSON files")
    designs = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            did = d.get('design_id') or os.path.splitext(os.path.basename(p))[0]
            d['design_id'] = did
            designs.append(d)
        except Exception as e:
            logger.error(f"Error loading {p}: {e}")
    return designs

def build_graph_from_design(design: dict) -> dgl.DGLGraph:
    """Convert a design dict into a DGL graph with features & labels."""
    did = design.get('design_id', 'Unknown')
    nodes = design.get('nodes', [])
    if not nodes:
        raise ValueError(f"Design {did} has no nodes")
    N = len(nodes)
    idx_map = {n['NodeID']: i for i, n in enumerate(nodes)}

    feat_list = []
    label_list = []
    for n in nodes:
        feat = [n[k] for k in INPUT_FEATURE_NAMES[:-1]] + [float(N)]
        lab  = [n[k] for k in OUTPUT_FEATURE_NAMES]
        feat_list.append(feat)
        label_list.append(lab)

    feats  = torch.tensor(feat_list, dtype=torch.float32)
    labels = torch.tensor(label_list, dtype=torch.float32)

    src, dst = [], []
    for n in nodes:
        i = idx_map[n['NodeID']]
        for nb in n.get('connectivity', []):
            j = idx_map.get(nb)
            if j is None:
                logger.warning(f"{did}: invalid neighbor {nb}")
            elif i != j:
                src.append(i)
                dst.append(j)

    g = dgl.graph((src, dst), num_nodes=N)
    g = dgl.add_self_loop(g)
    g.ndata['feat']  = feats
    g.ndata['label'] = labels
    g.design_id      = did
    return g

def create_dataset(json_folder: str) -> list:
    """Load designs and build a list of DGL graphs."""
    designs = load_json_files(json_folder)
    seen, graphs = set(), []
    for d in designs:
        did = d['design_id']
        if did in seen:
            continue
        try:
            g = build_graph_from_design(d)
            graphs.append(g)
            seen.add(did)
        except Exception as e:
            logger.error(f"Failed building graph {did}: {e}")
    logger.info(f"Dataset: {len(graphs)}/{len(designs)} graphs created")
    return graphs

def create_dataset_dict(json_folder: str) -> dict:
    """Same as create_dataset but returns a dict of graphs keyed by design_id."""
    graphs = create_dataset(json_folder)
    dct = {g.design_id: g for g in graphs}
    logger.info(f"Graph dict with {len(dct)} entries")
    return dct

def collate_fn(samples: list) -> tuple:
    """Custom collate that batches only valid graphs."""
    valid = [g for g in samples if hasattr(g, 'design_id')]
    if not valid:
        return None, []
    bg = dgl.batch(valid)
    return bg, [g.design_id for g in valid]
