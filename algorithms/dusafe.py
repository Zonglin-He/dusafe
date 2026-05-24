"""Key dusafe components: SSAW and triple safety gating.

This file is a compact excerpt for review. It omits trainer code,
setup tables, logging utilities, and checkpoint handling.
"""

from collections import deque
import math

import torch
import torch.nn.functional as F


def softmax_entropy(logits):
    """Return per-sample entropy from logits."""
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)


def safe_update_loss(warped_logits, gate_outputs, regularizer=None):
    """Final update loss on samples accepted by the safety gates."""
    safe_mask = gate_outputs["safe_mask"]
    entropy = softmax_entropy(warped_logits)
    if torch.any(safe_mask):
        adaptation_loss = entropy[safe_mask].mean()
    else:
        adaptation_loss = entropy.new_zeros(())

    if regularizer is None:
        regularizer = entropy.new_zeros(())

    total_loss = adaptation_loss + regularizer
    return {
        "loss": total_loss,
        "adaptation_loss": adaptation_loss,
        "regularizer": regularizer,
        "selected_count": safe_mask.sum(),
    }


def prototype_quality(stat_entropy, num_classes):
    """Confidence-derived prototype weight used after gate selection."""
    quality = 1.0 - (stat_entropy.detach() / math.log(max(2, int(num_classes))))
    return quality.clamp(min=0.0, max=1.0)


class SSAWSearch:
    """Spline-based stochastic amplitude warping search."""

    def __init__(self, control_points=10, candidates=16, sigma=0.1):
        self.control_points = max(2, int(control_points))
        self.candidates = max(0, int(candidates))
        self.sigma = float(sigma)
        self.last_metadata = None

    def _sample_controls(self, batch_size, device, dtype, sigma):
        controls = torch.ones(
            batch_size,
            self.candidates,
            self.control_points,
            device=device,
            dtype=dtype,
        )
        if sigma <= 0.0 or self.candidates <= 0:
            return controls
        noise = torch.randn_like(controls) * sigma + 1.0
        return noise.clamp(1.0 - 3.0 * sigma, 1.0 + 3.0 * sigma)

    @staticmethod
    def _natural_cubic_spline(controls, target_len):
        if controls.dim() != 2:
            raise ValueError(f"Expected [N, M], got {tuple(controls.shape)}")
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        if controls.size(1) == 1:
            return controls.repeat(1, target_len)

        device = controls.device
        dtype = controls.dtype
        rows, cols = controls.shape
        work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        y = controls.to(work_dtype)

        ctrl_x = torch.linspace(0.0, float(target_len - 1), cols, device=device, dtype=work_dtype)
        h = ctrl_x[1:] - ctrl_x[:-1]
        second = torch.zeros(rows, cols, device=device, dtype=work_dtype)

        if cols > 2:
            rhs = 6.0 * (
                (y[:, 2:] - y[:, 1:-1]) / h[1:].unsqueeze(0)
                - (y[:, 1:-1] - y[:, :-2]) / h[:-1].unsqueeze(0)
            )
            system = torch.zeros(cols - 2, cols - 2, device=device, dtype=work_dtype)
            system.diagonal().copy_(2.0 * (h[:-1] + h[1:]))
            if cols - 3 > 0:
                system.diagonal(offset=1).copy_(h[1:-1])
                system.diagonal(offset=-1).copy_(h[1:-1])
            second[:, 1:-1] = torch.linalg.solve(system.unsqueeze(0).expand(rows, -1, -1), rhs)

        eval_x = torch.linspace(0.0, float(target_len - 1), target_len, device=device, dtype=work_dtype)
        idx = torch.bucketize(eval_x, ctrl_x[1:-1], right=False).clamp(max=cols - 2)
        x0 = ctrl_x[idx]
        x1 = ctrl_x[idx + 1]
        h_eval = x1 - x0
        left = x1 - eval_x
        right = eval_x - x0

        y0 = y[:, idx]
        y1 = y[:, idx + 1]
        m0 = second[:, idx]
        m1 = second[:, idx + 1]
        curves = (
            m0 * (left**3) / (6.0 * h_eval)
            + m1 * (right**3) / (6.0 * h_eval)
            + (y0 - m0 * (h_eval**2) / 6.0) * (left / h_eval)
            + (y1 - m1 * (h_eval**2) / 6.0) * (right / h_eval)
        )
        return curves.to(dtype)

    @staticmethod
    def _features(model, x):
        features = model.feature_extractor(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features

    @torch.no_grad()
    def __call__(self, x, model, sigma=None):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")

        sigma = self.sigma if sigma is None else float(sigma)
        batch_size, channels, target_len = x.shape
        if self.candidates <= 0 or sigma <= 0.0:
            curve = torch.ones(batch_size, target_len, device=x.device, dtype=x.dtype)
            self.last_metadata = {"mode": "identity", "curve": curve.detach().cpu()}
            return x

        controls = self._sample_controls(batch_size, x.device, x.dtype, sigma)
        flat_controls = controls.reshape(batch_size * self.candidates, self.control_points)
        curves = self._natural_cubic_spline(flat_controls, target_len)
        curves = curves.reshape(batch_size, self.candidates, target_len)

        warped = x.unsqueeze(1) * curves.unsqueeze(2)
        flat_x = warped.reshape(batch_size * self.candidates, channels, target_len)
        logits = model.classifier(self._features(model, flat_x))
        entropy = softmax_entropy(logits).reshape(batch_size, self.candidates)

        best = entropy.argmax(dim=1)
        batch_index = torch.arange(batch_size, device=x.device)
        best_curve = curves[batch_index, best]
        self.last_metadata = {
            "mode": "ssaw",
            "curve": best_curve.detach().cpu(),
            "control_points": controls[batch_index, best].detach().cpu(),
            "score": entropy[batch_index, best].detach().cpu(),
        }
        return x * best_curve.unsqueeze(1)


class TripleSafetyGate:
    """Statistical, semantic, and consistency gates for reliable updates."""

    def __init__(
        self,
        num_classes,
        entropy_quantile=0.7,
        entropy_window=512,
        min_history=32,
        semantic_threshold=0.5,
        consistency_threshold=0.5,
        min_entropy=0.0,
        prototype_momentum=0.9,
        warmup_count=1,
    ):
        self.num_classes = int(num_classes)
        self.entropy_quantile = float(entropy_quantile)
        self.entropy_history = deque(maxlen=max(1, int(entropy_window)))
        self.min_history = max(1, int(min_history))
        self.semantic_threshold = float(semantic_threshold)
        self.consistency_threshold = float(consistency_threshold)
        self.min_entropy = float(min_entropy)
        self.prototype_momentum = float(prototype_momentum)
        self.warmup_count = max(1, int(warmup_count))
        self.prototypes = None
        self.prototype_counts = None
        self.last_log = {}

    def _entropy_threshold(self, entropy):
        if len(self.entropy_history) >= self.min_history:
            history = torch.tensor(list(self.entropy_history), device=entropy.device, dtype=entropy.dtype)
            threshold = torch.quantile(history, self.entropy_quantile)
        else:
            threshold = entropy.new_tensor(math.log(max(2, self.num_classes)))
        return torch.clamp(threshold, min=self.min_entropy)

    def _ensure_prototypes(self, features):
        feature_dim = features.size(1)
        device = features.device
        if self.prototypes is None or self.prototypes.size(1) != feature_dim:
            self.prototypes = torch.zeros(self.num_classes, feature_dim, device=device)
            self.prototype_counts = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        else:
            self.prototypes = self.prototypes.to(device)
            self.prototype_counts = self.prototype_counts.to(device)

    def _update_history(self, entropy):
        self.entropy_history.extend(entropy.detach().cpu().tolist())

    @torch.no_grad()
    def update_prototypes(self, features, labels, weights=None):
        if features.numel() == 0:
            return
        self._ensure_prototypes(features)
        features = F.normalize(features, dim=1)
        labels = labels.detach()

        for class_idx in range(self.num_classes):
            mask = labels == class_idx
            if not torch.any(mask):
                continue
            if weights is None:
                class_mean = features[mask].mean(dim=0)
            else:
                class_weights = weights[mask].clamp_min(0.0)
                if float(class_weights.sum().item()) == 0.0:
                    continue
                class_weights = class_weights / class_weights.sum()
                class_mean = (features[mask] * class_weights.unsqueeze(1)).sum(dim=0)

            if int(self.prototype_counts[class_idx].item()) == 0:
                self.prototypes[class_idx] = class_mean
            else:
                self.prototypes[class_idx] = (
                    self.prototype_momentum * self.prototypes[class_idx]
                    + (1.0 - self.prototype_momentum) * class_mean
                )
            self.prototypes[class_idx] = F.normalize(self.prototypes[class_idx], dim=0)
            self.prototype_counts[class_idx] += int(mask.sum().item())

    def select(self, features, raw_logits, warped_logits):
        self._ensure_prototypes(features)
        raw_probs = torch.softmax(raw_logits, dim=1)
        warped_probs = torch.softmax(warped_logits, dim=1)
        labels = raw_probs.argmax(dim=1)

        mixture = 0.5 * (raw_probs + warped_probs)
        stat_entropy = -(mixture * mixture.clamp_min(1e-8).log()).sum(dim=1)
        entropy_threshold = self._entropy_threshold(stat_entropy.detach())
        stat_gate = stat_entropy <= entropy_threshold
        self._update_history(stat_entropy)

        normalized_features = F.normalize(features.detach(), dim=1)
        prototype_vectors = self.prototypes[labels]
        normalized_prototypes = F.normalize(prototype_vectors.detach(), dim=1)
        prototype_ready = self.prototype_counts[labels] >= self.warmup_count
        prototype_nonzero = prototype_vectors.detach().abs().sum(dim=1) > 0
        cosine_similarity = F.cosine_similarity(normalized_features, normalized_prototypes, dim=1)
        semantic_gate = (~prototype_ready) | (~prototype_nonzero) | (
            cosine_similarity >= self.semantic_threshold
        )

        warped_log_probs = warped_probs.detach().clamp_min(1e-8).log()
        consistency_score = F.kl_div(
            warped_log_probs,
            raw_probs.detach(),
            reduction="none",
        ).sum(dim=1)
        consistency_gate = consistency_score <= self.consistency_threshold

        safe_mask = stat_gate & semantic_gate & consistency_gate
        self.last_log = {
            "stat_gate_rate": float(stat_gate.float().mean().item()),
            "semantic_gate_rate": float(semantic_gate.float().mean().item()),
            "consistency_gate_rate": float(consistency_gate.float().mean().item()),
            "safe_gate_rate": float(safe_mask.float().mean().item()),
            "entropy_threshold": float(entropy_threshold.detach().item()),
        }
        return {
            "safe_mask": safe_mask,
            "labels": labels,
            "stat_gate": stat_gate,
            "semantic_gate": semantic_gate,
            "consistency_gate": consistency_gate,
            "stat_entropy": stat_entropy,
            "consistency_score": consistency_score,
            "cosine_similarity": cosine_similarity,
        }
