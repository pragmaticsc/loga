"""
train/bitlinear.py
==================
MLX implementation of the BitLinear layer for ternary-weight transformer
training, following BitNet b1.58 (Ma et al., arXiv:2402.17764).

Weights are quantized to {-1, 0, +1} via absmean rounding during the forward
pass. A per-layer float16 scaling factor recovers weight magnitude. Gradients
flow through the quantization step via the straight-through estimator (STE).
Activations remain at float16 — fully ternary activations are not used.

Usage:
    Replace mlx.nn.Linear with BitLinear in your model definition:

        # Before
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # After
        from train.bitlinear import BitLinear
        self.proj = BitLinear(d_model, d_model, bias=False)

    Everything else (optimizer, loss, training loop) is unchanged.

Measuring sparsity:
    from train.bitlinear import weight_sparsity, model_sparsity

    sparsity = model_sparsity(model)   # fraction of weights that are zero
    print(f"Weight sparsity: {sparsity:.1%}")
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class BitLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1} (BitNet b1.58 style).

    During training, weights are stored at float16 but quantized to ternary
    on every forward pass via the absmean function. Gradients pass through
    the quantization step via the straight-through estimator.

    During inference, identical behaviour — weights are re-quantized on each
    call. For deployment, call `bake_weights()` to permanently store the
    ternary values and discard the float weights.

    Args:
        in_features:  Input dimension.
        out_features: Output dimension.
        bias:         Whether to include a bias term (default False, following
                      BitNet convention).
        warmup_steps: Number of forward passes before quantization is applied.
                      During warmup, the layer behaves as standard float Linear.
                      Useful for stabilising early training dynamics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.warmup_steps = warmup_steps
        self._step = 0

        # Float weights — stored for gradient flow
        scale = 1.0 / math.sqrt(in_features)
        self.weight = mx.random.uniform(
            low=-scale, high=scale, shape=(out_features, in_features)
        )
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def _quantize_weights(self, w: mx.array) -> tuple[mx.array, mx.array]:
        """
        Quantize weight matrix to {-1, 0, +1} via absmean thresholding.

        Returns:
            w_ternary: Quantized weights.
            scale:     Per-layer scaling factor (mean absolute value).
        """
        scale = mx.mean(mx.abs(w))
        # Avoid division by zero on degenerate layers
        scale = mx.maximum(scale, mx.array(1e-8))
        w_scaled = w / scale
        # Round to nearest of {-1, 0, +1} by clipping then rounding
        # Values in (-0.5, 0.5) → 0; outside → ±1
        w_ternary = mx.clip(mx.round(w_scaled), a_min=-1.0, a_max=1.0)
        return w_ternary, scale

    def __call__(self, x: mx.array) -> mx.array:
        self._step += 1

        if self._step <= self.warmup_steps:
            # Warmup: use full-precision weights
            out = x @ self.weight.T
        else:
            # Quantize weights; STE means gradients treat w_ternary as w
            w_ternary, scale = self._quantize_weights(self.weight)
            # STE: in forward pass use w_ternary; in backward pass gradient
            # flows as if we used self.weight (standard MLX autograd handles
            # this because w_ternary is computed from self.weight)
            w_effective = self.weight + mx.stop_gradient(w_ternary - self.weight)
            out = x @ (w_effective * scale).T

        if self.bias is not None:
            out = out + self.bias

        return out

    def bake_weights(self) -> None:
        """
        Replace float weights with baked ternary values in-place.
        Call after training for efficient inference storage.
        After baking, the scaling factor is absorbed and weights are integers.
        """
        w_ternary, scale = self._quantize_weights(self.weight)
        self.weight = w_ternary * scale
        mx.eval(self.weight)

    @property
    def sparsity(self) -> float:
        """Fraction of weights that are (or would be) zero after quantization."""
        w_ternary, _ = self._quantize_weights(self.weight)
        mx.eval(w_ternary)
        n_zero = mx.sum(w_ternary == 0).item()
        n_total = w_ternary.size
        return n_zero / n_total


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def weight_sparsity(layer: BitLinear) -> float:
    """Return the zero-weight fraction for a single BitLinear layer."""
    return layer.sparsity


def model_sparsity(model: nn.Module) -> dict[str, float]:
    """
    Compute per-layer and aggregate sparsity for all BitLinear layers in a
    model. Returns a dict mapping layer path → sparsity fraction, plus a
    summary key 'overall'.

    Example output:
        {
            'transformer.h.0.attn.q_proj': 0.42,
            'transformer.h.0.attn.k_proj': 0.38,
            ...
            'overall': 0.41,
        }
    """
    results: dict[str, float] = {}
    total_zero = 0
    total_params = 0

    def _visit(module: nn.Module, prefix: str) -> None:
        nonlocal total_zero, total_params
        for name, child in module.named_modules():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, BitLinear):
                s = child.sparsity
                results[path] = round(s, 4)
                n_total = child.weight.size
                total_zero += int(s * n_total)
                total_params += n_total

    _visit(model, "")
    if total_params > 0:
        results["overall"] = round(total_zero / total_params, 4)
    return results


def replace_linear_with_bitlinear(
    model: nn.Module,
    warmup_steps: int = 0,
    skip_modules: Optional[list[str]] = None,
) -> nn.Module:
    """
    Recursively replace all nn.Linear layers in a model with BitLinear.
    Useful for converting an existing float model config to ternary without
    rewriting the architecture.

    Args:
        model:        The model to convert.
        warmup_steps: Passed to each BitLinear layer.
        skip_modules: List of attribute names to leave as nn.Linear
                      (e.g. ['lm_head'] to keep the output projection float).

    Returns:
        The modified model (in-place).
    """
    skip = set(skip_modules or [])

    for name, module in model.named_modules():
        if name in skip:
            continue
        if isinstance(module, nn.Linear):
            # Preserve weight shape and bias setting
            new_layer = BitLinear(
                in_features=module.weight.shape[1],
                out_features=module.weight.shape[0],
                bias=module.bias is not None,
                warmup_steps=warmup_steps,
            )
            # Copy existing weights as initialisation
            new_layer.weight = module.weight
            if module.bias is not None:
                new_layer.bias = module.bias
            # Set attribute on parent module
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = model
                for part in parts[0].split("."):
                    parent = getattr(parent, part)
                setattr(parent, parts[1], new_layer)
            else:
                setattr(model, name, new_layer)

    return model
