# TIR Training

PMD + Tool-Integrated Reasoning: multi-turn code interpreter with three additional features on top of vanilla OpenKimi PMD.

**Void-turn masking** (`openkimi/tir/tir_pmd_trainer.py`): turns where the model neither calls a tool nor produces a boxed answer are treated as "void". Their gradient contribution is zeroed in `response_mask` before the policy update. The void-turn rate is always logged as `tir/void_turn_rate` regardless of whether masking is enabled.

**Train-inference mismatch correction** (`openkimi/pmd/core_algos.py`): use the fixed upstream `opmd` rollout IS correction. `rollout_is_weights` are reduced to per-sequence scalars via masked mean over valid token positions, then applied before batch aggregation:

```python
# opmd (fixed upstream)
seq_is_weights = (rollout_is_weights * response_mask).sum(-1) / response_lengths  # (batch,)
pg_loss = torch.mean(seq_is_weights * per_seq_loss)   # weighted mean ✓
```

**Data-source-agnostic reward scoring** (`openkimi/tir/tir_reward_manager.py`): registers `tir_dapo`. Instead of routing by `data_source` (which raises `NotImplementedError` for unknown sources), it tries all three math scorers (`math_reward`, `math_dapo`, `math_verify`) and returns the max — any hit counts as correct.

## 1. Start the sandbox

Dependencies:
```bash
apt-get install -y bubblewrap
pip install fastapi uvicorn pydantic
```

Start:
```bash
cd sandbox && ./start_sandbox.sh
```

Verify it's up:
```bash
python3 sandbox/sandbox_smoke_test.py
```

## 2. Agent loop (`tir_agent`)

Implemented in `openkimi/tir/tir_agent_loop.py`, registered as `"tir_agent"`.

**System prompt injection**: if the conversation has no system prompt, or the existing one doesn't contain the TIR instructions, the agent prepends:

> Solve the following problem step by step. You can write executable Python code to help your reasoning. The code will be executed by an external sandbox, and execution output will be returned as observation text starting with `Code execution result:`.

**Tool call format**: the model must output a fenced code block to invoke the sandbox:

```
```python
# your code here
print(result)
```
```

The sandbox returns stdout/stderr as the next user turn. The model can call the tool multiple times before giving a final answer.

**Answer format**: end the response with `\boxed{answer}` on the last line, nothing after it.

**Void-turn definition**: a turn is "void" if the model neither emits a ```` ```python ```` block nor a `\boxed{}`. Void turns are always tracked; set `MASK_VOID_TURNS=True` to also zero their gradients.

## 3. Run training

```bash
bash examples/tir/run_pmd_tir.sh
```

Key env vars:

| Variable | Default | Notes |
|---|---|---|
| `MASK_VOID_TURNS` | `False` | void-turn masking (metrics always logged regardless) |
| `ROLLOUT_IS_LEVEL` | `sequence` | IS correction granularity |
| `ROLLOUT_IS_THRESHOLD` | `5.0` | IS weight clip |
