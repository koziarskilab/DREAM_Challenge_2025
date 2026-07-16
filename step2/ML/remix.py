import torch
import numpy as np


def remix_data(x, y, alpha=1.0, kappa=3.0, tau=0.5, num_class_list=None, device='cuda'):
    """Perform remix on the data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    # Remix logic: adjust lambda based on class frequencies
    if num_class_list is not None:
        lam_list = torch.empty(batch_size).fill_(lam).float().to(device)
        
        n_i = torch.tensor([num_class_list[label.item()] for label in y_a]).float().to(device)
        n_j = torch.tensor([num_class_list[label.item()] for label in y_b]).float().to(device)
        
        # Apply remix conditions
        if lam < tau:
            lam_list[n_i / n_j >= kappa] = 0
        if 1 - lam < tau:
            lam_list[(n_i * kappa) / n_j <= 1] = 1
            
        return mixed_x, y_a, y_b, lam_list
    
    return mixed_x, y_a, y_b, lam