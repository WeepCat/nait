from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None, pos_neg_ratio: float = 1.0):
        super().__init__()
        self.IGNORE_INDEX = -100
        # 将 weight 注册为 buffer，这样它会自动跟随模型移动到正确的设备
        self.register_buffer('class_weight', torch.tensor([1.0, pos_neg_ratio]))
        self.he_loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX, weight=self.class_weight)
        self.se_loss = SoftLabelCrossEntropyLoss(ignore_index=self.IGNORE_INDEX, weight=self.class_weight)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        # TODO: 在用 hard label 训练时需要考虑样本不均衡的问题，为不同的类别设置不同的权重
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask].squeeze(1)
        labels = labels[placeholder_mask]

        if logits.numel() == 0:
            zero = logits.sum() * 0.0
            if return_acc:
                acc = logits.new_zeros(())
                return zero, acc
            return zero

        print(labels.dtype)
        if labels.dtype == torch.float:
            # 修改这部分代码，当 labels.dtype 是 float 时，表示使用软标签，
            logits = logits[..., self.reward_token_ids]
            labels = labels[..., self.reward_token_ids]
            
            loss = self.se_loss(logits, labels)
            if not return_acc:
                return loss
            acc = 0.
            return loss, acc

        if self.reward_token_ids is not None:
            logits = logits[..., self.reward_token_ids]
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.he_loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight: Optional[Tensor] = None,
            ignore_index: int = -100,
            reduction: str = 'mean',
            ):
        """
        weight: 表示每个类别的权重，形状为 (num_classes,)
        ignore_index: 指定一个标签值，在计算损失时会被忽略
        该损失函数用于处理软标签的交叉熵损失计算
        软标签是指标签不是单一类别，而是一个概率分布，表示样本属于各个类别的可能性
        例如，对于二分类问题，软标签可以是 [0.8, 0.2]，表示样本有80%的概率属于类别0，20%的概率属于类别1
        该损失函数在计算交叉熵损失时，会考虑标签的概率分布，而不是单一类别
        适用于需要处理软标签的任务，例如知识蒸馏、多标签分类等
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: Tensor, target: Tensor, soft_target: Tensor) -> Tensor:
        """
        inputs: 模型输出的 logits
                形状可以是:
                - (bs, num_classes): 2D 分类任务
                - (bs, seq_len, num_classes): 3D 序列分类任务
        
        target: 硬标签，用于确定样本权重
                硬标签形状:
                - (bs,): 对应 2D inputs
                - (bs, seq_len): 对应 3D inputs
        
        soft_target: 软标签，用于计算损失
                软标签形状:
                - (bs, num_classes): 对应 2D inputs
                - (bs, seq_len, num_classes): 对应 3D inputs
        
        返回: 计算得到的损失值
        
        工作流程:
        1. 使用 soft_target 计算交叉熵损失
        2. 根据 target (硬标签) 的类别查找对应的 weight
        3. 将权重应用到损失上
        """
        # 1. 判断输入维度
        is_3d = (inputs.dim() == 3)
        
        if is_3d:
            # 3D 输入: (bs, seq_len, num_classes)
            bs, seq_len, num_classes = inputs.shape
            
            # 展平为 2D
            inputs_flat = inputs.reshape(-1, num_classes)  # (N, C) where N = bs * seq_len
            soft_target_flat = soft_target.reshape(-1, num_classes)  # (N, C)
            target_flat = target.reshape(-1)  # (N,)
        else:
            # 2D 输入: (bs, num_classes)
            bs, num_classes = inputs.shape
            inputs_flat = inputs  # (bs, C)
            soft_target_flat = soft_target  # (bs, C)
            target_flat = target  # (bs,)
        
        # 2. 验证 soft_target 的形状
        if soft_target_flat.shape != inputs_flat.shape:
            raise ValueError(
                f"soft_target shape {soft_target.shape} doesn't match "
                f"expected shape based on inputs {inputs.shape}. "
                f"For 2D inputs (bs, C), soft_target should be (bs, C). "
                f"For 3D inputs (bs, L, C), soft_target should be (bs, L, C)."
            )
        
        # 3. 创建 mask: 忽略 ignore_index 的位置
        mask = (target_flat != self.ignore_index)  # (N,)
        
        # 4. 计算 log softmax
        log_probs = F.log_softmax(inputs_flat, dim=-1)  # (N, C)
        
        # 5. 计算软标签交叉熵损失: -sum(soft_target * log_prob)
        loss = -(soft_target_flat * log_probs).sum(dim=-1)  # (N,)
        
        # 6. 根据硬标签 (target) 的类别应用权重
        if self.weight is not None:
            # 验证 weight 的形状
            if self.weight.size(0) != num_classes:
                raise ValueError(
                    f"Weight size {self.weight.size(0)} doesn't match "
                    f"num_classes {num_classes}"
                )
            
            # 根据硬标签的类别索引获取对应的权重
            # target_flat: (N,) 包含类别索引
            # self.weight: (num_classes,) 每个类别的权重
            
            # 将 ignore_index 位置的索引替换为 0（避免索引越界）
            # 这些位置的权重会被 mask 掩盖，所以实际值无关紧要
            target_for_weight = target_flat.clone()
            target_for_weight[~mask] = 0
            
            # 确保索引在有效范围内
            target_for_weight = target_for_weight.clamp(0, num_classes - 1).long()
            
            # 通过索引获取每个样本的权重
            sample_weights = self.weight[target_for_weight]  # (N,)
            
            # 应用权重
            loss = loss * sample_weights
        
        # 7. 应用 mask，将 ignore_index 位置的损失置零
        loss = loss * mask.float()
        
        # 8. 根据 reduction 方式返回结果
        if self.reduction == 'none':
            # 恢复原始形状
            if is_3d:
                return loss.reshape(bs, seq_len)
            else:
                return loss
        elif self.reduction == 'mean':
            # 只对有效位置求平均
            valid_count = mask.sum().clamp(min=1.0)
            return loss.sum() / valid_count
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                f"Invalid reduction mode: {self.reduction}. "
                f"Choose from 'none', 'mean', or 'sum'."
            )
        # # 将输入的logits转换为概率分布
        # log_probs = F.log_softmax(inputs, dim=-1)  # (N, C)
        
        # # 创建一个掩码，标记出需要忽略的样本
        # valid_mask = target.sum(dim=-1) != self.ignore_index  # (N,)
        
        # if not valid_mask.any():
        #     # 如果没有有效样本，返回0损失
        #     zero = input.sum() * 0.0
        #     return zero

        # # 只保留有效样本的logits和目标
        # valid_log_probs = log_probs[valid_mask]  # (M, C)
        # valid_targets = target[valid_mask]  # (M, C)

        # # 计算软标签交叉熵损失
        # loss = -(valid_targets * valid_log_probs).sum(dim=-1)  # (M,)
        
        # if self.weight is not None:
        #     # 如果提供了权重，应用权重
        #     print(self.weight)
        #     class_weights = self.weight.unsqueeze(0)  # (1, C)
        #     loss = (loss * class_weights).sum(dim=-1)  # (M,)

        # # 根据指定的reduction方式计算最终损失
        # if self.reduction == 'mean':
        #     loss = loss.mean()
        # elif self.reduction == 'sum':
        #     loss = loss.sum()
        # elif self.reduction == 'none':
        #     pass
        # return loss



# 测试
if __name__ == "__main__":
    logits = torch.tensor([[2.0, 0.5], [0.5, 1.5], [1.0, 1.0]])
    soft_labels = torch.tensor([[0.8, 0.2], [0.0, 1.0], [0.5, 0.5]])
    targets = torch.tensor([0, 1, 0])  # 硬标签，用于权重索引
    weight = torch.tensor([1.0, 2.0])
    loss_fn = SoftLabelCrossEntropyLoss(reduction='none', weight=weight)
    # 手动计算第一个样本的 loss (带权重) 再与 loss_fn 计算的结果对比
    log_probs = F.log_softmax(logits, dim=-1)
    print(log_probs)
    expected_loss_0 = -(soft_labels[0] * log_probs[0]).sum() * weight[0]
    expected_loss_1 = -(soft_labels[1] * log_probs[1]).sum() * weight[1]
    expected_loss_2 = -(soft_labels[2] * log_probs[2]).sum() * weight[0]
    expected_loss = torch.tensor([expected_loss_0, expected_loss_1, expected_loss_2])
    computed_loss = loss_fn(logits, targets, soft_labels)
    print(f"Expected Loss: {expected_loss}")
    print(f"Computed Loss: {computed_loss}")
    assert torch.allclose(expected_loss, computed_loss), "Loss computation is incorrect."
    loss = loss_fn(logits, targets, soft_labels)
    print(f"Soft Label Cross Entropy Loss: {loss}")