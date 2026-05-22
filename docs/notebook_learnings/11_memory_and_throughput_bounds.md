# Memory & Throughput Bounds for Nemotron-30B LoRA (Round-4 note)

Source discussed: Kaggle thread by **Tong Hui Kang** on theoretical memory/throughput limits for Nemotron-3-Nano-30B-A3B-BF16 with LoRA rank 32 at sequence length 8192.

## Why this note matters

The thread is useful because it explains **why training feels hard even when LoRA is "small"**:
- Base model memory is still dominant.
- Activation memory grows with microbatch and sequence length.
- End-to-end MFU is far from peak FLOPs in real training.
- Fused CE/chunked loss can remove a major logits-memory spike.

This does not directly give a winning recipe, but it clarifies hard constraints and prevents wasting runs on impossible configs.

## What we should trust from the thread

1. **Directional memory math is correct**: for 30B-scale models, microbatch and seq length are the first OOM levers.
2. **Fused CE matters**: unchunked logits can be a large hidden memory cliff.
3. **MFU is much lower than peak**: wall-clock estimates assuming near-peak FLOPs are optimistic.
4. **8192 is the practical target** for this competition (eval max model len is 8192), so training beyond this has little value.

## What to treat cautiously

1. Exact GB totals are implementation-dependent (kernel versions, optimizer states, checkpointing style, framework internals).
2. Claims about specific microbatch values fitting should be validated empirically on our stack, not copied blindly.
3. FlashAttention support in Nemotron paths can vary by implementation (`trust_remote_code`, transformers version, and patches).

## Actionable guidance for our setup (NVIDIA RTX PRO 6000 Blackwell, 96GB)

For our hardware, this supports an **efficiency-first strategy on a single 96GB GPU**:

1. Prefer **targeted continuation SFT** over long full-corpus runs.
2. Keep conservative memory settings first:
   - microbatch 1 as default (try 2 only after stable runs)
   - use grad accumulation to reach effective batch targets
   - sequence length reduction as first OOM fallback
3. Treat **GRPO as optional Stage 2** after SFT, not a replacement.
4. Require promotion gates on local CV (overall + weak label), because compute alone is not the bottleneck.

## Recommended operating envelope (single Blackwell 96GB)

Start here, then scale only after stable runs:

| Knob | Start | If unstable |
|---|---|---|
| Sequence length | 8192 | 6144, then 4096 |
| Microbatch | 1 | keep 1 (do not scale) |
| Grad accumulation | 16 to 32 (depends on step-time budget) | lower if unstable |
| Objective | continuation SFT | shorter run, tighter checkpoint cadence |
| RL stage | optional tiny GRPO | skip if no clear gain over SFT checkpoint |

Practical note:
- Compared with L4-class GPUs, the 96GB Blackwell gives much more headroom for 8192-token training, but this is still a 30B model. Stability comes from conservative microbatch and accumulation, not aggressive microbatch scaling.

## Bottom line

This thread is a good **constraints map**, not a full training plan.  
For us, it reinforces the current strategy: **SFT-first, narrowly targeted, Blackwell-safe settings, strict CV-based promotion**.
