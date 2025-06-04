import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def get_one_hot(targets, num_classes):
    """Convert targets to one-hot encoding"""
    return torch.eye(num_classes)[targets].to(targets.device)


class CrossEntropy(nn.Module):
    """Standard Cross Entropy Loss"""
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        if para_dict is not None:
            self.num_classes = para_dict["num_classes"]
            self.num_class_list = para_dict["num_class_list"]
            self.device = para_dict["device"]
            self.cfg = para_dict.get("cfg", {})
            
            # Two-stage training parameters
            self.drw = self.cfg.get("train", {}).get("two_stage", {}).get("drw", False)
            self.drw_start_epoch = self.cfg.get("train", {}).get("two_stage", {}).get("start_epoch", 160)
            
            # Initialize weight list
            self.weight_list = torch.ones(self.num_classes).to(self.device)
        else:
            self.weight_list = None

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        loss = F.cross_entropy(input=inputs, target=targets.long(), weight=self.weight_list)
        return loss

    def update(self, epoch):
        """Update method for epoch-dependent changes"""
        pass


class CDT(CrossEntropy):
    """
    Class-Dependent Temperatures (CDT) Loss
    
    Reference: 
    Ye et al., Identifying and Compensating for Feature Deviation in Imbalanced Deep Learning, arXiv 2020.
    
    Equation: Loss(x, c) = -log(exp(x_c / a_c) / sum_i(exp(x_i / a_i)))
    where a_j = (N_max/n_j)^gamma, gamma is a hyper-parameter, N_max is the number 
    of images in the largest class, and n_j is the number of images in class j.
    
    Args:
        gamma (float): Controls the punishment to feature deviation. 
                      For binary classification, typically γ ∈ [0.0, 0.1]
    """
    
    def __init__(self, para_dict=None):
        super(CDT, self).__init__(para_dict)
        self.gamma = self.para_dict["cfg"]["loss"]["CDT"]["GAMMA"]
        
        # Calculate class-dependent temperatures: a_j = (max(n_i) / n_j)^gamma
        self.cdt_weight = torch.FloatTensor(
            [(max(self.num_class_list) / i) ** self.gamma for i in self.num_class_list]
        ).to(self.device)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        # Apply class-dependent temperatures by dividing logits by temperature weights
        temperature_adjusted_inputs = inputs / self.weight_list
        loss = F.cross_entropy(temperature_adjusted_inputs, targets.long())
        return loss

    def update(self, epoch):
        """
        Args:
            epoch: int. starting from 1.
        """
        if not self.drw:
            self.weight_list = self.cdt_weight
        else:
            self.weight_list = torch.ones(self.cdt_weight.shape).to(self.device)
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.cdt_weight


class ClassBalanceCE(CrossEntropy):
    """
    Class-Balanced Cross Entropy Loss
    Reference: Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.
    """
    def __init__(self, para_dict=None):
        super(ClassBalanceCE, self).__init__(para_dict)
        self.beta = self.para_dict["cfg"]["loss"]["ClassBalanceCE"]["BETA"]
        self.class_balanced_weight = np.array(
            [(1 - self.beta) / (1 - self.beta**N) for N in self.num_class_list]
        )
        self.class_balanced_weight = torch.FloatTensor(
            self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes
        ).to(self.device)

    def update(self, epoch):
        """Args: epoch: int. starting from 1."""
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight
            else:
                self.weight_list = torch.ones(self.class_balanced_weight.shape).to(self.device)


class ClassBalanceFocal(CrossEntropy):
    """
    Class-Balanced Focal Loss
    Reference: 
    - Li et al., Focal Loss for Dense Object Detection. ICCV 2017.
    - Cui et al., Class-Balanced Loss Based on Effective Number of Samples. CVPR 2019.
    """
    def __init__(self, para_dict=None):
        super(ClassBalanceFocal, self).__init__(para_dict)
        self.beta = self.para_dict["cfg"]["loss"]["ClassBalanceFocal"]["BETA"]
        self.gamma = self.para_dict["cfg"]["loss"]["ClassBalanceFocal"]["GAMMA"]
        self.class_balanced_weight = np.array(
            [(1 - self.beta) / (1 - self.beta**N) for N in self.num_class_list]
        )
        self.class_balanced_weight = torch.FloatTensor(
            self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes
        ).to(self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        weight = (self.weight_list[targets.long()]).to(targets.device)
        preds = inputs.view(-1, inputs.size(-1))
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_logsoft = preds_logsoft - 1e-6
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, targets.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, targets.view(-1, 1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = (loss * weight.view(-1, 1)).mean()
        return loss

    def update(self, epoch):
        """Args: epoch: int. starting from 1."""
        if not self.drw:
            self.weight_list = self.class_balanced_weight
        else:
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.class_balanced_weight
            else:
                self.weight_list = torch.ones(self.class_balanced_weight.shape).to(self.device)


class BalancedSoftmaxCE(CrossEntropy):
    """
    Balanced Softmax Cross Entropy Loss
    Reference: Ren et al., Balanced Meta-Softmax for Long-Tailed Visual Recognition, NeurIPS 2020.
    """
    def __init__(self, para_dict=None):
        super(BalancedSoftmaxCE, self).__init__(para_dict)
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        logits = (
            inputs + self.weight_list.unsqueeze(0).expand(inputs.shape[0], -1).log()
        )
        loss = F.cross_entropy(input=logits, target=targets.long())
        return loss

    def update(self, epoch):
        """Args: epoch: int"""
        if not self.drw:
            self.weight_list = self.bsce_weight
        else:
            self.weight_list = torch.ones(self.bsce_weight.shape).to(self.device)
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.bsce_weight


class CostSensitiveCE(CrossEntropy):
    """
    Cost-Sensitive Cross Entropy Loss
    """
    def __init__(self, para_dict=None):
        super(CostSensitiveCE, self).__init__(para_dict)
        gamma = self.para_dict["cfg"]["loss"]["CostSensitiveCE"]["GAMMA"]
        self.csce_weight = torch.FloatTensor(
            np.array(
                [(min(self.num_class_list) / N) ** gamma for N in self.num_class_list]
            )
        ).to(self.device)

    def update(self, epoch):
        """Args: epoch: int. starting from 1."""
        if not self.drw:
            self.weight_list = self.csce_weight
        else:
            start = (epoch - 1) // self.drw_start_epoch
            if start:
                self.weight_list = self.csce_weight


class InfluenceBalancedLoss(CrossEntropy):
    """
    Influence-Balanced Loss
    Reference: Seulki et al., Influence-Balanced Loss for Imbalanced Visual Classification, ICCV 2021.
    """
    def __init__(self, para_dict=None):
        super(InfluenceBalancedLoss, self).__init__(para_dict)

        ib_weight = 1.0 / np.array(self.num_class_list)
        ib_weight = ib_weight / np.sum(ib_weight) * self.num_classes
        self.ib_weight = torch.FloatTensor(ib_weight).to(self.device)
        self.use_vanilla_ce = False
        self.alpha = self.para_dict["cfg"]["loss"]["InfluenceBalancedLoss"]["ALPHA"]

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            feature: feature tensor from the model (required for influence calculation)
        """
        if self.use_vanilla_ce:
            return F.cross_entropy(inputs, targets.long(), weight=self.weight_list)
        
        # Get feature from kwargs
        feature = kwargs.get('feature', None)
        if feature is None:
            # Fallback to standard cross entropy if no features provided
            return F.cross_entropy(inputs, targets.long(), weight=self.weight_list)
        
        # Calculate gradients for influence
        grads = torch.sum(
            torch.softmax(inputs, dim=1) * inputs
            - F.one_hot(targets.to(torch.int64), self.num_classes) * inputs,
            1,
        )
        
        # Calculate influence-based weights
        # Ensure feature is properly shaped - take mean across feature dimensions
        if len(feature.shape) > 1:
            feature_norm = torch.mean(feature, dim=1)  # Average across feature dimensions
        else:
            feature_norm = feature
            
        ib = grads * feature_norm
        ib = self.alpha / (ib + 1e-3)
        
        # Apply influence-balanced loss
        ib_loss = (
            F.cross_entropy(
                inputs, targets.long(), reduction="none", weight=self.weight_list
            )
            * ib
        )
        return ib_loss.mean()

    def update(self, epoch):
        """Args: epoch: int"""
        if not self.drw:
            self.weight_list = self.ib_weight
        else:
            self.weight_list = torch.ones(self.ib_weight.shape).to(self.device)
            start = (epoch - 1) // self.drw_start_epoch
            self.use_vanilla_ce = True
            if start:
                self.use_vanilla_ce = False
                self.weight_list = self.ib_weight


def get_loss_function(loss_type, num_class_list, device):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: str, one of ['CE', 'CB_F', 'BS', 'CB_CE', 'CS', 'IB', 'CDT']
        num_class_list: list of class sample counts
        device: torch device
    
    Returns:
        Loss function instance
    """
    num_classes = len(num_class_list)
    
    # Create parameter dictionary required by the loss functions
    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "device": device,
        "cfg": {
            "loss": {
                "ClassBalanceFocal": {
                    "BETA": 0.999,
                    "GAMMA": 0.5,
                },
                "ClassBalanceCE": {
                    "BETA": 0.999,
                },
                "CostSensitiveCE": {
                    "GAMMA": 1.0,
                },
                "InfluenceBalancedLoss": {
                    "ALPHA": 1000.0,
                },
                "CDT": {
                    "GAMMA": 0.1,  # Adjusted for binary classification
                },
            },
            "train": {
                "two_stage": {
                    "drw": False,
                    "start_epoch": 160,
                }
            }
        }
    }
    
    if loss_type == "CE":
        return CrossEntropy(para_dict)
    elif loss_type == "CB_F":
        return ClassBalanceFocal(para_dict)
    elif loss_type == "BS":
        return BalancedSoftmaxCE(para_dict)
    elif loss_type == "CB_CE":
        return ClassBalanceCE(para_dict)
    elif loss_type == "CS":
        return CostSensitiveCE(para_dict)
    elif loss_type == "IB":
        return InfluenceBalancedLoss(para_dict)
    elif loss_type == "CDT":
        return CDT(para_dict)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: ['CE', 'CB_F', 'BS', 'CB_CE', 'CS', 'IB', 'CDT']")


# Example usage:
if __name__ == "__main__":
    # Example with imbalanced dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class_list = [1000, 100]  # Class 0: 1000 samples, Class 1: 100 samples
    
    # Create different loss functions
    ce_loss = get_loss_function("CE", num_class_list, device)
    cb_focal_loss = get_loss_function("CB_F", num_class_list, device)
    balanced_softmax_loss = get_loss_function("BS", num_class_list, device)
    cb_ce_loss = get_loss_function("CB_CE", num_class_list, device)
    cs_loss = get_loss_function("CS", num_class_list, device)
    ib_loss = get_loss_function("IB", num_class_list, device)
    cdt_loss = get_loss_function("CDT", num_class_list, device)  # New CDT loss
    
    # Example forward pass
    batch_size = 32
    num_classes = 2
    inputs = torch.randn(batch_size, num_classes).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Standard losses
    ce_output = ce_loss(inputs, targets)
    cb_focal_output = cb_focal_loss(inputs, targets)
    cdt_output = cdt_loss(inputs, targets)  # New CDT loss
    
    # For InfluenceBalancedLoss, we need features
    features = torch.randn(batch_size, 128).to(device)  # Example feature tensor
    ib_output = ib_loss(inputs, targets, feature=features)
    
    print(f"CrossEntropy Loss: {ce_output.item():.4f}")
    print(f"Class-Balanced Focal Loss: {cb_focal_output.item():.4f}")
    print(f"CDT Loss: {cdt_output.item():.4f}")
    print(f"Influence-Balanced Loss: {ib_output.item():.4f}")