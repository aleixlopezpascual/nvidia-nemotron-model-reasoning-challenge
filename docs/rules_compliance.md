# Competition Rules — Compliance Notes

Source: `kaggle competitions pages nvidia-nemotron-model-reasoning-challenge --page-name rules --content`. Pulled 2026-05-18. Re-pull before final submission to catch any host updates.

This is a working summary tied to **our specific resources**: Kaggle free GPU hours, unlimited Vertex AI, Vertex AI cloud notebooks. Read the source rules end-to-end before signing anything.

---

## TL;DR — what we can use

| Resource | Allowed? | Rule reference |
|---|---|---|
| **Training on Vertex AI / GCP** | ✅ Yes | No restriction on training compute. Host *recommends* GCP G4 VMs in the Compute section. |
| **Training in Kaggle notebooks** | ✅ Yes | Standard. |
| **Vertex AI to call paid teacher LLMs (Gemini, Claude, GPT-4 via API) for synthetic CoT** | ✅ Yes — under "Reasonableness Standard" | §2.6.b explicitly cites "Gemini Advanced" subscription as an acceptable example. |
| **Open-weight teachers (DeepSeek, Qwen, Llama, Nemotron-Super-49B)** | ✅ Yes | Publicly available, equally accessible — meets §2.6.a. |
| **Public Kaggle datasets like `tonghuikang/nemotron` CoT corpus** | ✅ Yes | Publicly available, equally accessible. |
| **Hand-labeling part of the test set** | ❌ No | §3.4.b: "submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records." (Train is fine; test isn't.) |
| **Proprietary dataset costing > $106K to license** | ❌ No | §2.6.b.2: license costs exceeding the prize pool are not "reasonable". |
| **Submitting from multiple accounts** | ❌ No | §3.5.a — instant disqualification. |
| **Privately sharing code outside our team** | ❌ No | §3.5.d & §3.6.a — must share publicly on Kaggle forum/notebooks if at all. |

## Obligations if we win (Section 2.5 & 2.8)

These are the gating conditions for actually receiving prize money:

1. **License our LoRA + training source code under CC BY 4.0** (§2.5.a.1). This includes training code, inference code, and environment description.
2. **Carve-out for incompatible-license inputs**: if we use a proprietary teacher model or proprietary dataset to *generate* our solution, we *don't* have to open-source that input — only our own code & adapter (§2.5.a.3). So calling Gemini/Claude/GPT-4 to generate CoT data is fine; we don't have to redistribute the API responses or the model weights.
3. **Publish a public Kaggle notebook + write-up** documenting our method (§2.8.a.1.c). Prize ineligibility otherwise. Must be detailed enough to reproduce — architecture, preprocessing, loss, hyperparams, training details, plus a link to a code repo.
4. **Deliver the training code itself**, capable of regenerating the winning submission, with a description of required compute (§2.8.a.1).
5. **Sign IRS tax forms** (W-9 for US, W-8BEN for non-US), prize-acceptance docs.

## Operational limits

- **5 submissions per day max** (§1.2.a). Be intentional — don't burn submissions on small ablations.
- **2 final submissions** picked for the private leaderboard (§1.2.b). Diversify selection.
- **Max team size: 5** (§1.1.a). Team-merge deadline same as entry deadline: **2026-06-08**.
- **Final submission deadline: 2026-06-15 23:59 UTC**.
- Public leaderboard ≠ private leaderboard (§3.7.a). The 0.87 public score might shake.

## Submission environment (from notebooks, NOT the rules text)

The rules page itself doesn't pin down the inference-env constraints — those come from the Evaluation page and from the public submission demos. From our earlier notebook analysis (`docs/notebook_learnings/01_baseline_and_submission.md`):

- **No internet** in the submission scoring environment. Ship any non-stdlib deps via a `wheelhouse` Kaggle dataset.
- The submission is a **`submission.zip` containing a LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`, base = `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`).
- vLLM is the inference engine with: `max_lora_rank=32`, `max_tokens=7680`, `top_p=1.0`, `temperature=0.0`, `max_num_seqs=64`, `gpu_memory_utilization=0.85`, `max_model_len=8192`.

These are scoring-time constraints — they apply to the submitted adapter's behavior, not to where we *trained* it.

## Open Contribution Awards (extra prize pots worth knowing about)

Beyond the LB-position prizes, three independent DGX Sparks are awarded for "techniques that meaningfully advance reasoning performance":

- **Best Data/Synthetic Data Method**
- **Best RL Method**
- **Best Fine-tuning Method**

Eligibility: top 10% of final LB *and* explicit form submission per category. So if we do something genuinely novel on data or RL — even at a mid-rank LB position — there's a separate prize lane for it.

## Citation requirement

Any use of the competition data must include attribution to **NVIDIA Research** (§1.4.a.2, CC BY 4.0). Include in any public write-up.

## Things explicitly NOT prohibited (worth using)

- Using paid APIs for synthetic data generation (Gemini/Claude/GPT-4).
- Using arbitrary external compute (Vertex AI, GCP, our own GPUs, anything).
- Using *any* publicly available pre-trained model as a teacher for distillation.
- Mixing multiple public CoT corpora (we already found `tonghuikang/nemotron`, `kienngx/...`, `konbu17/...`).
- Generating very large synthetic datasets on Vertex AI and uploading them as public Kaggle datasets (then they're "equally accessible" by definition).

## Open questions worth asking the host (via Discussion forum)

1. Is there a hard cap on the *runtime* of the inference scoring (besides `max_tokens` per query)? Some Kaggle code competitions have a 9-hour total runtime ceiling.
2. Are there any restrictions on using the official Nemotron-3-Nano-30B-A3B-BF16 weights for training (e.g., are derived adapters subject to NVIDIA's model license on top of CC BY 4.0)?
3. Can a single team member's account be used for shared GCP compute, or does each teammate need to attest separately?

I'd suggest checking the Discussion forum's pinned host posts before raising new threads — likely already answered.
