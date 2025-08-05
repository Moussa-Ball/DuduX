#!/usr/bin/env python3
"""
BSO: Binary Spiking Online Optimizer for DUDUX-GPT
==================================================

🔥 ICML 2025 Binary Spiking Online Optimization Algorithm
⚡ Memory-efficient binary weight optimization without latent storage
🧠 Adapted for DUDUX-GPT binary neural architecture
💾 ~50% memory reduction compared to standard binary training

Based on: "BSO: Binary Spiking Online Optimization Algorithm" (ICML 2025)
Authors: Yu Liang, Yu Yang, Wenjie Wei, et al.

Key Innovations:
- Eliminates latent weight storage
- Direct weight updates via flip signals  
- Time-independent memory requirements
- Adaptive threshold adjustment

Authors: DUDUX Research Team
Version: 1.0.0 Professional
Created: August 5, 2025
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
from typing import Dict, Any, Optional
import math


class DuduxBSO(Optimizer):
    """
    Binary Spiking Online (BSO) optimizer adapted for DUDUX-GPT

    Eliminates latent weight storage and directly updates binary weights
    through flip signals triggered by gradient momentum vs threshold.

    Args:
        binary_params: Parameters of binary layers (BinaryNeuron, MassiveBinaryLayer)
        other_params: Parameters of other layers (embeddings, layer norm, etc.)
        lr: Learning rate for non-binary parameters
        threshold: Threshold for flip signal triggering
        beta1: Momentum coefficient for gradient moving average
        beta2: Second moment coefficient for adaptive threshold
        weight_decay: L2 regularization coefficient
        eps: Small constant for numerical stability

    Example:
        >>> binary_params = [p for n, p in model.named_parameters() if 'binary' in n]
        >>> other_params = [p for n, p in model.named_parameters() if 'binary' not in n]
        >>> optimizer = DuduxBSO(binary_params, other_params, lr=0.001, threshold=1e-7)
    """

    def __init__(
        self,
        binary_params,
        other_params,
        lr: float = 1e-3,
        threshold: float = 1e-7,
        beta1: float = 0.999,
        beta2: float = 0.99999,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        adaptive_threshold: bool = True,
        verbose: bool = True
    ):
        # Separate binary and non-binary parameters
        self.binary_params = list(binary_params)
        self.other_params = list(other_params)
        self.adaptive_threshold = adaptive_threshold

        # Standard optimizer for non-binary parameters
        if other_params:
            self._standard_optimizer = torch.optim.AdamW(
                other_params, lr=lr, weight_decay=weight_decay,
                betas=(0.9, 0.999), eps=eps
            )
        else:
            self._standard_optimizer = None

        # BSO state tracking
        self.global_step = 0

        # BSO defaults for binary parameters
        defaults = dict(
            lr=lr,
            threshold=threshold,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps
        )

        # Initialize BSO optimizer for binary parameters only
        super(DuduxBSO, self).__init__(self.binary_params, defaults)

        if verbose:
            print(f"🔥 DuduxBSO Optimizer Initialized:")
            print(f"   Binary Parameters: {len(self.binary_params):,}")
            print(f"   Other Parameters: {len(self.other_params):,}")
            print(f"   Threshold: {threshold:.2e}")
            print(f"   Beta1: {beta1}, Beta2: {beta2}")
            print(f"   Adaptive Threshold: {adaptive_threshold}")

    def __setstate__(self, state):
        super(DuduxBSO, self).__setstate__(state)

    def zero_grad(self):
        """Zero gradients for both binary and non-binary parameters"""
        super().zero_grad()
        if self._standard_optimizer:
            self._standard_optimizer.zero_grad()

    def get_binary_params_info(self) -> Dict[str, int]:
        """Get information about binary parameters"""
        total_params = sum(p.numel() for p in self.binary_params)
        total_memory_mb = sum(
            p.numel() * 4 for p in self.binary_params) / (1024**2)  # float32

        return {
            'total_binary_params': total_params,
            'memory_mb': total_memory_mb,
            'latent_memory_saved_mb': total_memory_mb  # BSO saves this much vs standard
        }

    def step(self, closure=None):
        """
        Perform BSO optimization step

        Args:
            closure: Optional closure to re-evaluate the loss

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.global_step += 1

        # 1. BSO step for binary parameters
        self._bso_step()

        # 2. Standard step for non-binary parameters
        if self._standard_optimizer:
            self._standard_optimizer.step()

        return loss

    def _bso_step(self):
        """Core BSO algorithm for binary parameter updates"""

        for group in self.param_groups:
            threshold = group['threshold']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Momentum buffer (gradient moving average)
                    state['m'] = torch.zeros_like(p.data)
                    # Second moment for adaptive threshold
                    if self.adaptive_threshold:
                        state['v'] = torch.zeros_like(p.data)

                state['step'] += 1

                # Update momentum
                state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)

                if self.adaptive_threshold:
                    # Update second moment for adaptive threshold
                    state['v'].mul_(beta2).add_(grad**2, alpha=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    # Adaptive threshold based on second moment
                    corrected_m = state['m'] / bias_correction1
                    corrected_v = state['v'] / bias_correction2

                    # BSO flip condition with adaptive threshold
                    adaptive_threshold = threshold * \
                        torch.sqrt(corrected_v + eps)
                    flip_condition = p.data * corrected_m - adaptive_threshold
                else:
                    # Standard BSO with fixed threshold
                    flip_condition = p.data * state['m'] - threshold

                # Generate flip signals (-1 for flip, +1 for keep)
                reverse = -torch.sign(flip_condition)

                # Direct binary weight update (core BSO innovation)
                p.data.copy_(torch.sign(reverse * p.data))

                # Update momentum based on flip decisions
                # If flipped (reverse=-1), reduce momentum; if kept (reverse=+1), maintain
                # Convert {-1,+1} to {0,1}
                momentum_update_mask = (reverse + 1) / 2
                state['m'].mul_(momentum_update_mask)

    def get_lr(self) -> float:
        """Get current learning rate"""
        if self._standard_optimizer:
            return self._standard_optimizer.param_groups[0]['lr']
        return self.param_groups[0]['lr']

    def get_threshold(self) -> float:
        """Get current threshold for binary parameters"""
        return self.param_groups[0]['threshold']

    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        stats = {
            'global_step': self.global_step,
            'threshold': self.get_threshold(),
            'lr': self.get_lr(),
            'binary_params_info': self.get_binary_params_info()
        }

        # Add flip statistics if available
        total_flips = 0
        total_params = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.state[p]
                    if 'step' in state and state['step'] > 0:
                        # Calculate how many weights would flip this step
                        threshold = group['threshold']
                        flip_condition = p.data * state['m'] - threshold
                        flips = torch.sum(flip_condition < 0).item()
                        total_flips += flips
                        total_params += p.numel()

        if total_params > 0:
            stats['flip_rate'] = total_flips / total_params
        else:
            stats['flip_rate'] = 0.0

        return stats


class DuduxBSOScheduler:
    """
    Scheduler for BSO optimizer parameters

    Gradually reduces threshold and adjusts momentum coefficients
    during training for better convergence.
    """

    def __init__(
        self,
        optimizer: DuduxBSO,
        param_name: str = "threshold",
        decay_epochs: int = 50,
        decay_factor: float = 0.1,
        min_value: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.param_name = param_name
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor
        self.min_value = min_value
        self.last_epoch = 0

        # Store initial values
        self.initial_values = []
        for group in optimizer.param_groups:
            self.initial_values.append(group[param_name])

    def step(self, epoch: Optional[int] = None):
        """Update scheduler"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Calculate decay factor
        decay_count = epoch // self.decay_epochs
        current_factor = self.decay_factor ** decay_count

        # Update parameter values
        for group, initial_value in zip(self.optimizer.param_groups, self.initial_values):
            new_value = initial_value * current_factor

            # Apply minimum value constraint
            if self.min_value is not None:
                new_value = max(new_value, self.min_value)

            group[self.param_name] = new_value

    def get_current_value(self) -> float:
        """Get current parameter value"""
        return self.optimizer.param_groups[0][self.param_name]


def create_dudux_bso_optimizer(model, lr: float = 1e-3, threshold: float = 1e-7, **kwargs) -> DuduxBSO:
    """
    Create BSO optimizer for DUDUX-GPT model

    Automatically separates binary and non-binary parameters based on layer names.

    Args:
        model: DUDUX-GPT model
        lr: Learning rate
        threshold: BSO threshold for flip signals
        **kwargs: Additional BSO parameters

    Returns:
        Configured DuduxBSO optimizer
    """
    binary_params = []
    other_params = []

    # Separate parameters based on layer types and names
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Binary layers: BinaryNeuron, MassiveBinaryLayer weights
            is_binary = any(keyword in name.lower() for keyword in [
                'weight_matrix',  # MassiveBinaryLayer
                'weights',        # BinaryNeuron
                'binary'          # Any layer with 'binary' in name
            ])

            if is_binary:
                binary_params.append(param)
            else:
                other_params.append(param)

    print(f"🔍 Parameter Classification:")
    print(f"   Binary Parameters: {len(binary_params):,} tensors")
    print(f"   Other Parameters: {len(other_params):,} tensors")

    return DuduxBSO(
        binary_params=binary_params,
        other_params=other_params,
        lr=lr,
        threshold=threshold,
        **kwargs
    )
