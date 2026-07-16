import torch
import numpy as np


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Perform mixup on the data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    if isinstance(lam, torch.Tensor):
        # For remix with tensor lambda
        loss_a = criterion(pred, y_a)
        loss_b = criterion(pred, y_b)
        if loss_a.dim() == 0:  # scalar loss
            loss_a = loss_a.unsqueeze(0).expand(len(lam))
        if loss_b.dim() == 0:  # scalar loss
            loss_b = loss_b.unsqueeze(0).expand(len(lam))
        return (lam * loss_a + (1 - lam) * loss_b).mean()
    else:
        # For standard mixup with scalar lambda
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)