import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: [B, C, H, W] logits
        target: [B, H, W] long tensor with class indices
        """
        # 应用softmax获取概率
        pred_prob = F.softmax(pred, dim=1)

        # 处理ignore_index（如果指定）
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            target = target * valid_mask.long()
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        # 将目标转换为one-hot编码
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # 应用有效掩码
        pred_prob = pred_prob * valid_mask.unsqueeze(1)
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1)

        # 计算每个类别的Dice系数（二分类时排除背景）
        intersection = torch.sum(pred_prob * target_one_hot, dim=(2, 3))
        union = torch.sum(pred_prob, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # 返回所有类别的平均Dice损失（二分类时只关注前景）
        if num_classes == 2:
            # 对于二值分割，重点关注前景类（索引1）
            dice_loss = 1 - dice_score[:, 1].mean()
        else:
            dice_loss = 1 - dice_score.mean()

        return dice_loss

class TripleCombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.4, dice_weight=0.4, ce_weight=0.2,   
                 focal_alpha=0.75, focal_gamma=2.0, ignore_index=255):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        # Focal Loss（来自现有的utils/loss.py）
        from utils.loss import FocalLoss
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)

        # Dice Loss
        self.dice_loss = DiceLoss(smooth=1e-6, ignore_index=ignore_index)

        # CE Loss
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index != 255 else -100)

    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target)

        total_loss = (self.focal_weight * focal_loss +
                     self.dice_weight * dice_loss +
                     self.ce_weight * ce_loss)

        return total_loss
    