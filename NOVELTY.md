# Novelty & Positioning

This document situates **persona-CI red-team** against prior art. It is deliberately
honest: most of the *scaffolding* is established work. The defensible contribution is a
specific **recombination** plus a specific **reward signal**.

---

## Verdict

**Novelty: ~5.5 / 10.**

The individual building blocks — synthetic personas, contextual integrity (CI) for LLMs,
LLM-as-judge, GRPO attacker/defender co-evolution, guardrail evasion — are each already
established in the literature. The fresh, defensible part is the **combination** and the
**optimization target**:

> A **weight-level GRPO** attacker optimized against a **per-persona, structured
> care-vector** subjective contextual-integrity reward — *"allowed-but-compromising for
> THIS person"* — with **Llama Guard repurposed as a personalized-CI gate.**

---

## The two differentiators that matter

For a technicality-weighted judge, lead with these:

1. **Weight-level GRPO, not prompt search.** The attack policy lives in the model's
   weights and improves via GRPO advantage. Most adjacent privacy/CI red-teaming searches
   over *prompts*, leaving the underlying model fixed.
2. **Structured per-persona care vector as the RL optimization target, not generic CI.**
   The reward is defined by a structured `info_type -> {sensitivity, forbidden_recipients,
   allowed_recipients, forbidden_purposes}` map. "Compromise" is *person-relative*, so the
   policy must learn to manipulate against a specific subject's norms — not against a
   universal leakage label.

A supporting third element: **Llama Guard 3 is used as a personalized CI gate** by injecting
the persona's care vector as a custom contextual-integrity policy, so the guard blocks
*blatant* violations for *this* person and the attacker must find subtle, guard-evading
compromises.

---

## Related work

| Work | What it is | Overlap | How we differ |
|------|-----------|:-------:|---------------|
| **Searching for Privacy Risks in LLM Agents via Simulation** ([arXiv:2508.10880](https://arxiv.org/abs/2508.10880)) | Closest prior art: synthetic personas, CI framing, LLM judge, attacker/defender co-evolution | **~70%** | Uses **prompt search**, not weight-level GRPO; targets **generic CI leakage**, not a **structured per-persona care vector** |
| **ConfAIde** ([arXiv:2310.17884](https://arxiv.org/abs/2310.17884)) | CI-for-LLM benchmark / eval | Med | We *optimize an attacker* against CI; ConfAIde is a static eval, not an RL target |
| **AirGapAgent** ([arXiv:2405.05175](https://arxiv.org/abs/2405.05175)) | Contextual privacy for LLM agents | Med | Defense / containment framing; we train an adversary against a personalized care vector |
| **PrivacyLens / "Privacy in Action"** ([arXiv:2509.17488](https://arxiv.org/abs/2509.17488)) | CI evaluation for LLM agent actions | Med | Evaluation, not a weight-level RL attacker |
| **AdvGRPO** ([arXiv:2606.09701](https://arxiv.org/abs/2606.09701)) | GRPO-based adversarial attacker | Med | GRPO co-evolution, but for **jailbreaks/toxicity**, not per-persona privacy/CI |
| **CHASE / Self-RedTeam** ([arXiv:2506.07468](https://arxiv.org/abs/2506.07468)) | Self-play attacker–defender red-teaming | Med | Co-evolution, but for **jailbreaks/toxicity**, not per-persona privacy/CI |
| **Bad Characters** ([arXiv:2106.09898](https://arxiv.org/abs/2106.09898)) | Char-level adversarial text | Low | Char-level guardrail evasion — a saturated adjacent area we **deliberately do not center on** |
| **Bypassing LLM Guardrails** ([arXiv:2504.11168](https://arxiv.org/abs/2504.11168)) | Guardrail evasion techniques | Low | Same — we treat guard-evasion as a *constraint on the reward*, not the contribution itself |

**Reading of the field:** the two clusters nearest to us are (a) *CI-for-LLM privacy*
(ConfAIde, AirGapAgent, PrivacyLens, arXiv:2508.10880) — strong on contextual integrity but
mostly **evaluation** or **prompt-level search**; and (b) *GRPO red-team co-evolution*
(AdvGRPO, CHASE/Self-RedTeam) — strong on **weight-level** adversarial training but aimed at
**universal jailbreak/toxicity** rather than person-specific privacy. We sit at the
intersection: weight-level GRPO **and** structured per-persona CI.

---

## Novelty-sharpening experiments

Planned ablations to demonstrate that the contribution is real, not incidental:

1. **Care-vector ablation.** Strip the structured care vector (fall back to generic CI). We
   expect the attacker to **collapse to generic social-engineering** — evidence that the
   structured target is what drives person-specific manipulation.
2. **Persona-specificity transfer.** Train a policy on persona A, evaluate on persona B. We
   expect **degraded compromise rate** on B — evidence the policy learns *person-relative*
   norms rather than a universal attack.
3. **Judge calibration.** Validate Gemma's CI scores against a **second judge** and/or a
   **persona self-report**, to show the reward signal is measuring contextual integrity and
   not a single judge's idiosyncrasy.

---

## Honest caveat

Training has **not** run yet. The novelty argument here is about *design and positioning*;
the empirical case rests on the planned deliverables in [`README.md`](./README.md) — chiefly
**D1**, the subjective-compromise-rate rising curve vs a static baseline.
