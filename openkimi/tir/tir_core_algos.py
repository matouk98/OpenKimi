"""
TIR-specific PMD policy loss with correct rollout IS weight handling.

The base opmd loss (openkimi/pmd/core_algos.py) applies rollout_is_weights
*after* torch.mean(), i.e. scalar * (batch, seq_len) — incorrect.

"tir_opmd" fixes this: IS weights are reduced to per-sequence scalars and
multiplied *before* torch.mean(), so each sequence's loss is properly
weighted before aggregation.

Registration: "tir_opmd"
    actor_rollout_ref.actor.policy_loss.loss_mode=tir_opmd
"""

from typing import Any, Optional

import torch
from omegaconf import DictConfig

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_policy_loss


@register_policy_loss("tir_opmd")
def compute_policy_loss_tir_opmd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    extra_loss_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Online PMD loss with correct rollout IS weight application.

    Identical to opmd except rollout_is_weights (shape: batch x seq_len) are
    reduced to per-sequence scalars and multiplied *before* torch.mean(), so
    that each sequence's contribution to the batch loss is correctly weighted.

    For sequence-level IS: every valid token position carries the same weight,
    so masked-mean over tokens gives the sequence scalar.
    For token-level IS: averaging over valid tokens reduces to a sequence scalar.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"

    pmd_tau = (
        config.policy_loss.get("pmd_tau", 0.01)
        if hasattr(config, "policy_loss") and config.policy_loss is not None
        else 0.01
    )

    response_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)  # (batch,)

    seq_log_prob = torch.sum(log_prob * response_mask, dim=-1)          # (batch,)
    seq_old_log_prob = torch.sum(old_log_prob * response_mask, dim=-1)  # (batch,)
    seq_log_prob_ratio = seq_log_prob - seq_old_log_prob                 # (batch,)

    seq_advantages = torch.sum(advantages * response_mask, dim=-1) / response_lengths  # (batch,)

    if extra_loss_kwargs is not None and "partition_weights" in extra_loss_kwargs:
        pw = extra_loss_kwargs["partition_weights"]
        seq_partition_weights = torch.sum(pw * response_mask, dim=-1) / response_lengths
    else:
        seq_partition_weights = torch.ones_like(seq_advantages)

    # Reduce (batch, seq_len) IS weights → per-sequence scalar (batch,).
    # For sequence-level IS all valid positions share the same weight;
    # for token-level IS this averages over valid tokens.
    if rollout_is_weights is not None:
        seq_is_weights = (rollout_is_weights * response_mask).sum(dim=-1) / response_lengths
    else:
        seq_is_weights = torch.ones_like(seq_advantages)

    # Per-sequence MSE loss (before batch mean).
    seq_loss = (
        seq_is_weights
        * 0.5
        * pmd_tau
        * seq_partition_weights
        * ((seq_advantages / pmd_tau - seq_log_prob_ratio) ** 2)
    )

    max_response_length = response_mask.shape[1]
    if loss_agg_mode == "seq-mean-token-mean":
        pg_loss = torch.mean(seq_loss / response_lengths)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        pg_loss = torch.mean(seq_loss) / max_response_length
    else:
        raise NotImplementedError(f"Unsupported loss_agg_mode: {loss_agg_mode}")

    seq_kl = -seq_log_prob_ratio / response_lengths
    ppo_kl = torch.mean(seq_kl)

    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics
