# training/train.py

import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import MODEL_SAVE_FOLDER, NUM_EPOCHS, LAMBDA_SIGN, INPUT_FEATURE_NAMES
from config import logger
from utils import evaluate_model

def train_and_evaluate(
    model, train_loader, val_dataset, optimizer, loss_fn, device,
    num_epochs=NUM_EPOCHS, patience=100
):
    """Train loop with ReduceLROnPlateau, mixed precision, early stopping, and sign penalty."""
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scaler    = torch.cuda.amp.GradScaler()
    best_mape = float('inf')
    no_improve = 0

    train_losses, val_accs, val_losses = [], [], []
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        batches = 0
        for bg, _ in train_loader:
            if bg is None: continue
            gdev = bg.to(device)
            # ensure input features require gradients for penalty
            feat = gdev.ndata['feat']
            feat.requires_grad_(True)

            with torch.cuda.amp.autocast():
                out       = model(gdev, feat)
                loss_pred = loss_fn(out, gdev.ndata['label'])
                # sign-regularisation: penalise positive ∂Volume/∂feat for the three features
                vol   = out[:,1].sum()
                grads = torch.autograd.grad(vol, feat, create_graph=True)[0]
                idxs  = [INPUT_FEATURE_NAMES.index(f) for f in ("DoorCount","WindowCount","OpeningArea")]
                bad   = grads[:, idxs]       # [N,3]
                penalty = F.relu(bad).mean()
                loss    = loss_pred + LAMBDA_SIGN * penalty

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_pred.item()
            batches += 1

        train_loss = total_loss / batches if batches else 0.0
        mape, acc, vloss = evaluate_model(model, val_dataset, device, loss_fn)
        train_losses.append(train_loss)
        val_accs.append(acc)
        val_losses.append(vloss)

        scheduler.step(vloss)

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"TrainPredLoss={train_loss:.4f} | ValLoss={vloss:.4f} | "
            f"ValMAPE={mape:.2f}% | ValAcc={acc:.2f}%"
        )

        if mape < best_mape:
            best_mape = mape
            no_improve = 0
            os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_FOLDER, "graph_regressor_state.pth"))
            logger.info(f" New best model saved (MAPE={best_mape:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"No improvement for {patience} epochs, stopping early.")
                break

    return train_losses, val_accs, val_losses
