"""
TIRPMDTrainer: RayPMDTrainer extended with void-turn masking for TIR training.

Adds mask_void_turns support without touching any existing files:
- Reads `+trainer.tir.mask_void_turns=True` from config (trainer.tir is a free
  dict that bypasses Hydra's strict FSDPActorConfig schema).
- Overrides _balance_batch() to apply void_turn_mask (produced by TIRAgentLoop)
  to response_mask before the balance step, zeroing out policy gradient updates
  for "void" rollouts (no tool call and no boxed answer).
- Imports and registers TIRAgentLoop ("tir_agent") on init.

Use actor_rollout_ref.rollout.agent.default_agent_loop=tir_agent in your config.
"""

import numpy as np
import torch

from verl import DataProto
from openkimi.pmd.pmd_ray_trainer import RayPMDTrainer


class TIRPMDTrainer(RayPMDTrainer):
    """RayPMDTrainer + void-turn masking for Tool-Integrated Reasoning."""

    def __init__(self, *args, **kwargs):
        # Register TIRAgentLoop ("tir_agent") before any rollout workers start.
        import examples.tir.tir_agent_loop  # noqa: F401

        super().__init__(*args, **kwargs)

        # Read from trainer.tir (free dict, not schema-validated) to avoid
        # Hydra rejecting unknown fields in FSDPActorConfig.
        mask_void = self.config.trainer.get("tir", {}).get("mask_void_turns", False)
        if isinstance(mask_void, str):
            self.mask_void_turns = mask_void.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            self.mask_void_turns = bool(mask_void)

    def _balance_batch(self, batch: DataProto, metrics, **kwargs):
        """Log void-turn / sandbox metrics and optionally apply void_turn_mask, then balance."""
        ntb = batch.non_tensor_batch

        # -- void_turn metrics (always logged) -------------------------------
        if "void_turn_mask" in ntb:
            void_mask_np = np.asarray(ntb["void_turn_mask"], dtype=np.float32).reshape(-1)
            metrics["tir/void_turn_rate"] = float(1.0 - void_mask_np.mean())
            metrics["tir/active_turn_rate"] = float(void_mask_np.mean())

            if self.mask_void_turns:
                response_mask = batch.batch["response_mask"]
                if void_mask_np.shape[0] == response_mask.shape[0]:
                    void_mask = torch.from_numpy(void_mask_np).to(
                        device=response_mask.device,
                        dtype=response_mask.dtype,
                    )
                    batch.batch["response_mask"] = response_mask * void_mask.unsqueeze(-1)

        # -- sandbox metrics (always logged when present) --------------------
        for key, metric_name in [
            ("sandbox_called", "tir/sandbox_call_rate"),
            ("sandbox_success", "tir/sandbox_success_rate"),
            ("sandbox_call_count", "tir/sandbox_call_count_mean"),
            ("sandbox_success_count", "tir/sandbox_success_count_mean"),
        ]:
            if key in ntb:
                arr = np.asarray(ntb[key], dtype=np.float32).reshape(-1)
                metrics[metric_name] = float(arr.mean())

        super()._balance_batch(batch, metrics, **kwargs)
