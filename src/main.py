# main.py
import torch
import time
import random
import os
from dgl.dataloading import GraphDataLoader

from config import (
    JSON_FOLDER,
    EXTRA_VALIDATION_FOLDER,
    EXTRA_GROUND_TRUTH_FOLDER,
    VIZ_FOLDER,
    MODEL_SAVE_FOLDER,
    TRAIN,
    ENABLE_VISUALIZATIONS,
    PLOT_VISUALIZATIONS,
    TRAIN_SPLIT,
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    OUTPUT_FEATURE_NAMES,
    INPUT_FEATURE_NAMES
)
from config import logger

from data.dataset import create_dataset, create_dataset_dict, collate_fn
import utils
from utils import (
    predict,
    gather_predictions,
    gather_predictions_ood,
    integrated_gradients_attribution_all_nodes,
    evaluate_model,
    evaluate_extra_validation,
    plot_training_metrics,
    plot_prediction_vs_actual_scatter,
    plot_scaled_feature_contributions_heatmap
)
from model import GraphRegressor
from training.train import train_and_evaluate

def main():
    start = time.time()
    logger.info(f"Script start | device={DEVICE} | train={TRAIN}")

    # initialize viz session once
    if ENABLE_VISUALIZATIONS and not PLOT_VISUALIZATIONS:
        utils.CURRENT_SESSION_FOLDER = utils.get_new_viz_session_folder(VIZ_FOLDER)
        logger.info(f"Viz folder: {utils.CURRENT_SESSION_FOLDER}")

    # load & split ID dataset
    id_graphs = create_dataset(JSON_FOLDER)
    if not id_graphs:
        logger.critical("No ID graphs loaded. Exiting.")
        return
    random.shuffle(id_graphs)
    split = int(len(id_graphs) * TRAIN_SPLIT)
    train_g, val_g = id_graphs[:split], id_graphs[split:]
    logger.info(f"ID graphs: train={len(train_g)}, val={len(val_g)}")

    # DataLoader
    train_loader = None
    if TRAIN:
        train_loader = GraphDataLoader(
            train_g,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    # model, loss, optimizer
    model = GraphRegressor().to(DEVICE)
    import torch.nn as nn
    import torch.optim as optim
    loss_fn = nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train or load
    if TRAIN:
        tl, va, vl = train_and_evaluate(model, train_loader, val_g, optimizer, loss_fn, DEVICE)
        if ENABLE_VISUALIZATIONS:
            plot_training_metrics(tl, va, vl)
    else:
        model_path = os.path.join(MODEL_SAVE_FOLDER, "graph_regressor_state.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            logger.info(f"Loaded model from {model_path}; generating visualizations.")
            if ENABLE_VISUALIZATIONS:
                p_id, a_id = gather_predictions(model, val_g, DEVICE)
                for i, name in enumerate(OUTPUT_FEATURE_NAMES):
                    plot_prediction_vs_actual_scatter(p_id, a_id, i, name, 'ID')
        else:
            logger.critical(f"Model file not found: {model_path}")
            return

    # ID evaluation
    id_mape, id_acc, id_loss = evaluate_model(model, val_g, DEVICE, loss_fn)
    logger.info(f"ID Eval -> MAPE={id_mape:.2f}% Acc={id_acc:.2f}% Loss={id_loss:.4f}")

    # OOD evaluation & predictions + visualizations
    ood_graphs = create_dataset(EXTRA_VALIDATION_FOLDER)
    ood_gt_dict = create_dataset_dict(EXTRA_GROUND_TRUTH_FOLDER)
    if ood_graphs:
        ood_mape, ood_acc = evaluate_extra_validation(model, ood_graphs, ood_gt_dict, DEVICE)
        if ood_mape is not None:
            logger.info(f"OOD Eval -> MAPE={ood_mape:.2f}% Acc={ood_acc:.2f}%")
        if ENABLE_VISUALIZATIONS:
            for g in ood_graphs:
                did = getattr(g, 'design_id', None)
                if not did:
                    continue
                atts, _ = integrated_gradients_attribution_all_nodes(model, g, baseline=None, steps=50)
                preds = predict(model, g, DEVICE).numpy()
                plot_scaled_feature_contributions_heatmap(
                    attributions=atts.numpy(),
                    predictions=preds,
                    feature_names=INPUT_FEATURE_NAMES,
                    graph_id=did,
                    dataset_label='OOD',
                    alpha=0.3
                )
        gather_predictions_ood(model, ood_graphs, ood_gt_dict, DEVICE)

    logger.info(f"Done in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
