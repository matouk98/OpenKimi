"""
tir_reward_manager.py – TIR-specific reward manager.

Registers "tir_dapo" in both verl registries so it works regardless of
which code path instantiates the reward manager:
  1. verl.workers.reward_manager  (load_reward_manager in trainer/ppo/reward.py)
  2. verl.experimental.reward_loop.reward_manager  (RewardLoopWorker)

Ensemble scoring: tries math_reward, math_dapo, math_verify and returns
max — any hit counts as correct, data-source-agnostic.
"""

from verl.experimental.reward_loop.reward_manager.dapo import (
    DAPORewardManager as ExpDAPORewardManager,
)


def _tir_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    from verl.utils.reward_score import math_dapo, math_reward, math_verify

    def _hit(res) -> float:
        if isinstance(res, dict):
            return float(res.get("score", 0.0))
        if isinstance(res, int | float | bool):
            return float(res)
        return float(res[0])

    hits: list[float] = []
    for scorer in (math_reward, math_dapo, math_verify):
        try:
            hits.append(_hit(scorer.compute_score(solution_str, ground_truth)))
        except Exception:
            pass

    return max(hits) if hits else 0.0


class TIRDAPORewardManager(ExpDAPORewardManager):
    """Experimental DAPORewardManager with data-source-agnostic ensemble scoring."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            compute_score=compute_score or _tir_compute_score,
            reward_router_address=reward_router_address,
            reward_model_tokenizer=reward_model_tokenizer,
        )


# -- Register in experimental reward_loop registry -----------------------
from verl.experimental.reward_loop.reward_manager.registry import (  # noqa: E402
    REWARD_LOOP_MANAGER_REGISTRY,
)

REWARD_LOOP_MANAGER_REGISTRY.setdefault("tir_dapo", TIRDAPORewardManager)

# -- Register in workers registry (fallback) -----------------------------
try:
    from verl.workers.reward_manager.registry import (  # noqa: E402
        REWARD_MANAGER_REGISTRY,
    )

    REWARD_MANAGER_REGISTRY.setdefault("tir_dapo", TIRDAPORewardManager)
except Exception:
    pass
