# utils.py

import os
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

from config import (
    ENABLE_VISUALIZATIONS,
    PLOT_VISUALIZATIONS,
    VIZ_FOLDER,
    DEVICE
)
from config import logger

# ──────────────────────────────────────────────────────────────────────────────
# Global plotting style
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
plt.rc('font', family='DejaVu Sans', size=12)
plt.rc('axes', titlesize=14, titleweight='bold',
       labelsize=12, labelweight='bold')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# ──────────────────────────────────────────────────────────────────────────────
# Fixed, human-friendly feature labels
# ──────────────────────────────────────────────────────────────────────────────
_HEATMAP_FEATURE_LABELS = [
    "Length (m)", "Width (m)", "Key L.", "Key W.", "Doors (n)",
    "Windows (n)", "Open area (m²)", "Level (0/1)", "Net Area (m²)",
    "Intersect Area (m²)", "Modules (n)"
]

# ──────────────────────────────────────────────────────────────────────────────
# Visualization session helpers
# ──────────────────────────────────────────────────────────────────────────────
CURRENT_SESSION_FOLDER = None

def get_new_viz_session_folder(base_folder: str) -> str:
    """Create a new session folder for saving plots."""
    os.makedirs(base_folder, exist_ok=True)
    existing = [
        d for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d)) and d.startswith("session_")
    ]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_idx = max(nums, default=-1) + 1
    session_folder = os.path.join(base_folder, f"session_{next_idx}")
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def handle_figure(fig: plt.Figure, filename: str):
    """Save or display a matplotlib figure according to flags."""
    global CURRENT_SESSION_FOLDER
    if not ENABLE_VISUALIZATIONS:
        plt.close(fig)
        return
    if CURRENT_SESSION_FOLDER is None:
        CURRENT_SESSION_FOLDER = get_new_viz_session_folder(VIZ_FOLDER)
    if PLOT_VISUALIZATIONS:
        plt.show()
        plt.close(fig)
    else:
        path = os.path.join(CURRENT_SESSION_FOLDER, filename)
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Prediction utilities
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model: torch.nn.Module, g: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
    """Return model predictions for one graph."""
    graph = g.to(device)
    feats = graph.ndata['feat'].to(device)
    return model(graph, feats).cpu()

@torch.no_grad()
def gather_predictions(model, dataset, device):
    """Gather predictions and actuals for ID or OOD graphs."""
    all_p, all_a = [], []
    for g in dataset:
        try:
            gdev = g.to(device)
            p = model(gdev, gdev.ndata['feat']).cpu().numpy()
            a = gdev.ndata['label'].cpu().numpy()
            all_p.append(p)
            all_a.append(a)
        except Exception as e:
            logger.error(f"Gather preds for {getattr(g, 'design_id', '')} failed: {e}")
    from config import OUT_FEATS
    if not all_p:
        return np.empty((0, OUT_FEATS)), np.empty((0, OUT_FEATS))
    return np.vstack(all_p), np.vstack(all_a)

@torch.no_grad()
def gather_predictions_ood(model, dataset, gt_dict, device):
    """Print per-node predictions for OOD graphs."""
    for g in dataset:
        did = getattr(g, 'design_id', None)
        if not did:
            continue
        preds = predict(model, g, device).numpy()
        print(f"\n=== Predictions for OOD design '{did}' ===")
        for idx, (pc, pv) in enumerate(preds):
            print(f" Module {idx+1}: Predicted ObjectCount = {pc:.4f}, ObjectVolume = {pv:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Attribution (IG)
# ──────────────────────────────────────────────────────────────────────────────
def integrated_gradients_attribution_single(
    model: torch.nn.Module,
    graph: dgl.DGLGraph,
    target_node: int,
    baseline: torch.Tensor = None,
    steps: int = 50,
    target_dim: int = 1
) -> tuple:
    """Compute IG attributions for a single node."""
    model.eval()
    g = graph.to(DEVICE)
    feat = g.ndata['feat']
    baseline = baseline.to(DEVICE) if baseline is not None else torch.zeros_like(feat).to(DEVICE)

    def wrapper(input_feats):
        out_list = []
        orig = g.ndata['feat']
        for i in range(input_feats.shape[0]):
            g.ndata['feat'] = input_feats[i]
            o = model(g, input_feats[i])
            val = o[target_node, target_dim] if target_node < o.shape[0] else torch.tensor(0.0, device=DEVICE)
            out_list.append(val)
        g.ndata['feat'] = orig
        return torch.stack(out_list)

    ig = IntegratedGradients(wrapper)
    atts, delta = ig.attribute(
        inputs=feat.unsqueeze(0),
        baselines=baseline.unsqueeze(0),
        return_convergence_delta=True,
        n_steps=steps
    )
    return atts[0, target_node].cpu(), float(delta)

def integrated_gradients_attribution_all_nodes(
    model: torch.nn.Module,
    graph: dgl.DGLGraph,
    baseline: torch.Tensor = None,
    steps: int = 50,
    target_dim: int = 1
) -> tuple:
    """Compute IG attributions across all nodes."""
    N, Fdim = graph.num_nodes(), graph.ndata['feat'].shape[1]
    all_atts = torch.zeros((N, Fdim), device=DEVICE)
    all_deltas = torch.zeros(N, device=DEVICE)
    for idx in range(N):
        att, d = integrated_gradients_attribution_single(
            model, graph, target_node=idx,
            baseline=baseline, steps=steps, target_dim=target_dim
        )
        all_atts[idx] = att
        all_deltas[idx] = d
    avg_d, max_d = all_deltas.mean().item(), all_deltas.max().item()
    if avg_d > 0.05 or max_d > 0.1:
        logger.warning(f"IG deltas large avg={avg_d:.4f} max={max_d:.4f}")
    return all_atts.cpu(), all_deltas.cpu()

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation utilities
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, dataset, device, loss_fn=None):
    """Compute MAPE, accuracy, and loss over a dataset."""
    model.eval()
    mape_list, loss_list = [], []
    for g in dataset:
        graph = g.to(device)
        pred = model(graph, graph.ndata['feat'])
        true = graph.ndata['label']
        p_np, t_np = pred.cpu().numpy(), true.cpu().numpy()
        mape = np.mean(np.abs(t_np - p_np) / (np.abs(t_np) + 1e-6)) * 100
        mape_list.append(mape)
        if loss_fn:
            loss_list.append(loss_fn(pred, true).item())
    if not mape_list:
        return 0.0, 100.0, 0.0
    avg_mape = np.mean(mape_list)
    acc = 100 - avg_mape
    avg_loss = np.mean(loss_list) if loss_list else 0.0
    return avg_mape, acc, avg_loss

@torch.no_grad()
def evaluate_extra_validation(model, extra_val_dataset, gt_dict, device):
    """Evaluate OOD dataset against ground truth graphs."""
    model.eval()
    mape_list = []
    no_gt = mismatch = errors = 0
    for g in extra_val_dataset:
        if g is None or not hasattr(g, 'design_id'):
            errors += 1
            continue
        did = g.design_id
        if did not in gt_dict:
            no_gt += 1
            continue
        try:
            gdev = g.to(device)
            pred = model(gdev, gdev.ndata['feat'])
            true = gt_dict[did].ndata['label'].to(device)
            if pred.shape != true.shape:
                mismatch += 1
                continue
            p_np, t_np = pred.cpu().numpy(), true.cpu().numpy()
            mape_list.append(np.mean(np.abs(t_np - p_np) / (np.abs(t_np) + 1e-6)) * 100)
        except Exception as e:
            logger.error(f"OOD eval {did} error: {e}")
            errors += 1
    if not mape_list:
        logger.warning(
            f"OOD eval skipped all (no data). no_gt={no_gt} mismatch={mismatch} errors={errors}"
        )
        return None, None
    mape = np.mean(mape_list)
    acc = 100 - mape
    logger.info(f"OOD aggregated MAPE={mape:.2f}% Acc={acc:.2f}%")
    return mape, acc

# ──────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_metrics(train_losses, val_accuracies, val_losses=None):
    """Plot training & validation losses and accuracy with cleaner layout."""
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss curves
    ax1.plot(epochs, train_losses, linewidth=2.5, label="Training Loss")
    if val_losses is not None:
        ax1.plot(epochs, val_losses, linestyle="--", linewidth=2.5, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(labelsize=10)
    ax1.grid(True, linestyle="--", alpha=0.25)

    # Accuracy on twin axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accuracies, linestyle="-.", linewidth=2.5, label="Validation Acc")
    ax2.set_ylabel("Accuracy (%)")
    ax2.tick_params(labelsize=10)

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2,
               loc="lower center", ncol=3, frameon=False)

    fig.suptitle("Training Progress", fontsize=16)
    fig.tight_layout()
    handle_figure(fig, "training_metrics.png")

def plot_prediction_vs_actual_scatter(
    predictions: np.ndarray,
    actuals: np.ndarray,
    output_dim: int,
    output_name: str,
    dataset_label: str,
    filename_suffix: str = ""
):
    """Scatter of predicted vs actual values."""
    if predictions.size == 0 or actuals.size == 0:
        logger.warning(f"No data for scatter {output_name}")
        return

    pred = predictions[:, output_dim]
    act = actuals[:, output_dim]
    mn, mx = min(act.min(), pred.min()), max(act.max(), pred.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(act, pred, alpha=0.7, edgecolor='k', s=40)
    ax.plot([mn, mx], [mn, mx], '--', linewidth=1, color='gray')
    ax.set_xlabel(f'Actual {output_name}')
    ax.set_ylabel(f'Predicted {output_name}')
    ax.set_title(f'{dataset_label}: {output_name} Pred vs Actual')
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout()
    fname = f"{dataset_label}_{output_name}{filename_suffix}.png"
    handle_figure(fig, fname)

def plot_feature_attribution_heatmap(
    attributions: np.ndarray,
    feature_names: list,
    graph_id: str,
    dataset_label: str = "OOD"
):
    """Raw IG attribution heatmap (styled, correct aspect)."""
    if attributions is None or attributions.size == 0:
        logger.warning(f"No attributions for heatmap {graph_id}")
        return

    data = attributions
    N, F = data.shape
    vmax = np.abs(data).max()

    fig, ax = plt.subplots(
        figsize=(F * 0.7 + 1, N * 0.7 + 1),
        facecolor='white'
    )
    im = ax.imshow(data, cmap='seismic', vmin=-vmax, vmax=vmax,
                   aspect='auto', alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attribution (raw)', rotation=270, labelpad=15)

    ax.set_xticks(np.arange(F))
    ax.set_xticklabels(_HEATMAP_FEATURE_LABELS, rotation=45, ha='right')
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels([f"Module {i+1}" for i in range(N)])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Module')
    ax.set_title('Raw Feature Attribution', pad=15)

    for i in range(N):
        for j in range(F):
            val = data[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color=color, fontsize=9)

    plt.tight_layout()
    handle_figure(fig, f"raw_heatmap_{dataset_label}_{graph_id}.png")

def plot_scaled_feature_contributions_heatmap(
    attributions: np.ndarray,
    predictions: np.ndarray,
    feature_names: list,
    graph_id: str,
    dataset_label: str = "OOD",
    alpha: float = 0.5
):
    """Scaled contribution heatmap (styled, correct aspect)."""
    vols = predictions[:, 1].reshape(-1, 1)
    sums = attributions.sum(axis=1, keepdims=True)
    scale = np.divide(vols, sums, out=np.ones_like(vols), where=(sums != 0))
    data = attributions * scale

    N, F = data.shape
    vmax = np.abs(data).max()

    fig, ax = plt.subplots(
        figsize=(F * 0.7 + 1, N * 0.7 + 1),
        facecolor='white'
    )
    im = ax.imshow(data, cmap='seismic', vmin=-vmax, vmax=vmax,
                   aspect='auto', alpha=alpha)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Contribution (m³)', rotation=270, labelpad=15)

    ax.set_xticks(np.arange(F))
    ax.set_xticklabels(_HEATMAP_FEATURE_LABELS, rotation=45, ha='right')
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels([f"Module {i+1}" for i in range(N)])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Module')
    ax.set_title('Feature Contribution for Material Volume (m3)', pad=15)

    for i in range(N):
        for j in range(F):
            val = data[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color=color, fontsize=9)

    plt.tight_layout()
    handle_figure(fig, f"scaled_heatmap_{dataset_label}_{graph_id}.png")
