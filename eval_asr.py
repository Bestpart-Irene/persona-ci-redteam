"""Proper held-out ASR eval (replaces the noisy last-batch [asr] callback).

Runs a FIXED set of N (persona, target) episodes through the real stack
(attacker → victim → LlamaGuard → CI judge) with GREEDY decoding, and reports
ASR (win rate), mean reward, mean judge-compromise, and blocked/refused rates.

Compares attacker checkpoints head-to-head on the SAME episodes (same seed →
same persona/target draws), so base-vs-trained / lr-A-vs-B is apples-to-apples.
Loads victim/gate/judge ONCE and evaluates every adapter against them.

Usage:
  python eval_asr.py --n 100 --seed 7 \
      --adapters base,runs/ci/round3,runs/ci_lr1e-6/round3 \
      --out runs/eval_asr.json
  ('base' = the raw attacker with no LoRA adapter)
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from reward import episode_reward, WIN_THRESHOLD
from grpo_ci import attacker_prompt, _target, _as_text, _load_personas


def _gen(model, tok, msgs, max_new_tokens):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _load_attacker(base_id, adapter, tok):
    """Return a generation-ready model: base, or base+LoRA adapter."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16,
                                             device_map="auto")
    if adapter and adapter != "base":
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    return m.eval()


def evaluate(model, tok, episodes, victim, gate, judge, max_new_tokens):
    """Run the fixed episode list; return aggregate metrics + per-episode rows."""
    rows, n = [], len(episodes)
    agg = {"win": 0, "blocked": 0, "refused": 0, "benign": 0,
           "sum_reward": 0.0, "sum_comp": 0.0}
    for persona, info_t, recip in episodes:
        msg = attacker_prompt(persona, info_t, recip)
        req = _as_text(_gen(model, tok, msg, max_new_tokens))
        r, info = episode_reward(persona, req, victim, gate, judge)
        outcome = info.get("outcome", "")
        agg["sum_reward"] += r
        agg["sum_comp"] += float(info.get("compromise", 0.0) or 0.0)
        if info.get("win"):
            agg["win"] += 1
        if outcome == "blocked_by_guard":
            agg["blocked"] += 1
        elif outcome == "refused":
            agg["refused"] += 1
        elif outcome == "benign_pass":
            agg["benign"] += 1
        rows.append({"persona": persona.id, "info_type": info_t, "recipient": recip,
                     "request": req, "outcome": outcome, "reward": r,
                     "compromise": info.get("compromise")})
    return {
        "n": n,
        "asr": agg["win"] / max(1, n),                 # win = compromise>=WIN_THRESHOLD & passed guard
        "blocked_rate": agg["blocked"] / max(1, n),
        "refused_rate": agg["refused"] / max(1, n),
        "benign_pass_rate": agg["benign"] / max(1, n),
        "mean_reward": agg["sum_reward"] / max(1, n),
        "mean_compromise": agg["sum_comp"] / max(1, n),
        "win_threshold": WIN_THRESHOLD,
    }, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", default="mlabonne/Qwen3-4B-abliterated")
    ap.add_argument("--adapters", default="base",
                    help="comma list of adapter dirs to compare; 'base' = no adapter")
    ap.add_argument("--personas", type=int, default=64)
    ap.add_argument("--n", type=int, default=100, help="held-out episodes (fixed by seed)")
    ap.add_argument("--seed", type=int, default=7, help="fixes the persona/target draws")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", default="runs/eval_asr.json")
    args = ap.parse_args()

    # FIXED held-out episode set — identical across every adapter (same seed).
    personas = _load_personas(args.personas)
    rng = random.Random(args.seed)
    episodes = []
    for _ in range(args.n):
        p = personas[rng.randrange(len(personas))]
        info_t, recip = _target(p, rng)
        episodes.append((p, info_t, recip))
    print(f"[eval] {args.n} fixed episodes over {len(personas)} personas (seed={args.seed})")

    if args.mock:
        from victim import MockVictim
        from guard import MockGate
        from judge import MockJudge
        victim, gate, judge = MockVictim(), MockGate(), MockJudge()
    else:
        from victim import LocalVictim
        from guard import LlamaGuardGate
        from judge import GemmaJudge
        victim, gate, judge = LocalVictim(), LlamaGuardGate(), GemmaJudge()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.attacker, token=os.environ.get("HF_TOKEN"))

    results = {}
    for adapter in [a.strip() for a in args.adapters.split(",") if a.strip()]:
        print(f"\n[eval] === adapter: {adapter} ===")
        model = _load_attacker(args.attacker, adapter, tok)
        metrics, rows = evaluate(model, tok, episodes, victim, gate, judge,
                                 args.max_new_tokens)
        results[adapter] = {"metrics": metrics, "rows": rows}
        print(f"[eval] {adapter}: ASR={metrics['asr']:.3f} "
              f"mean_reward={metrics['mean_reward']:.3f} "
              f"mean_compromise={metrics['mean_compromise']:.3f} "
              f"blocked={metrics['blocked_rate']:.2f} refused={metrics['refused_rate']:.2f}")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"seed": args.seed, "n": args.n,
                   "results": {k: v["metrics"] for k, v in results.items()},
                   "episodes": {k: v["rows"] for k, v in results.items()}}, f, indent=2)
    print(f"\n[eval] wrote {args.out}")
    print("[eval] SUMMARY (ASR | mean_reward | mean_compromise):")
    for k, v in results.items():
        m = v["metrics"]
        print(f"  {k:36s} ASR={m['asr']:.3f}  R={m['mean_reward']:.3f}  "
              f"comp={m['mean_compromise']:.3f}")


if __name__ == "__main__":
    main()
