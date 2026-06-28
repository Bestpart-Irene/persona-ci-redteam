"""Self-generated curriculum — the second RSI loop (closes the "feeds back to ①" arrow).

After each training round we know which personas the attacker has SOLVED (high win-rate).
A solved target carries no more gradient, so we spawn HARDER descendants of it and fold
them back into the training distribution: stricter care vectors (higher sensitivity, more
forbidden recipients/purposes, sometimes a new info_type) force the attacker to find an
even subtler, guard-evading compromise. The distribution thus tracks the policy's own
frontier — performance-driven, open-ended, and bounded only by `max_personas`.

This is pure / model-light (deterministic hardening + optional generator augmentation), so
the whole loop runs and is unit-tested on CPU with `MockGenerator`.
"""

from __future__ import annotations

import copy
import random

from persona import Persona
from persona_gen import (INFO_TYPES, RELATIONSHIPS, PURPOSES, repair, validate)


def summarize(stats: dict, persona_id: str) -> dict | None:
    """Per-persona round summary: {n, win_rate, mean_reward} or None if unseen."""
    s = stats.get(persona_id)
    if not s or s.get("n", 0) <= 0:
        return None
    n = s["n"]
    return {"n": n, "win_rate": s.get("wins", 0) / n, "mean_reward": s.get("sum_r", 0.0) / n}


def _harden_rule(rule: dict, rng: random.Random) -> dict:
    """Make one care-vector rule stricter: ↑sensitivity, +1 forbidden recipient, +1 purpose."""
    new = copy.deepcopy(rule)
    new["sensitivity"] = round(min(1.0, rule["sensitivity"] + rng.uniform(0.08, 0.2)), 2)
    taken = set(rule["forbidden_recipients"]) | set(rule.get("allowed_recipients", []))
    pool = [r for r in RELATIONSHIPS if r not in taken]
    if pool:
        new["forbidden_recipients"] = rule["forbidden_recipients"] + [rng.choice(pool)]
    ppool = [p for p in PURPOSES if p not in rule.get("forbidden_purposes", [])]
    if ppool:
        new["forbidden_purposes"] = rule.get("forbidden_purposes", []) + [rng.choice(ppool)]
    return new


def harden_persona(parent: Persona, generator, rng: random.Random, round_idx: int) -> Persona | None:
    """Spawn a stricter descendant of a solved persona. Returns a VALID Persona or None."""
    cv = copy.deepcopy(parent.care_vector)
    # Harden the 2 highest-sensitivity norms (the ones the attacker exploited) — leave the
    # rest untouched so sensitivities stay varied (validate() rejects degenerate spreads).
    hot = sorted(cv, key=lambda k: cv[k]["sensitivity"], reverse=True)[:2]
    for k in hot:
        cv[k] = _harden_rule(cv[k], rng)
    # Optionally fold in one NEW info_type proposed by the generator (genuine expansion).
    if generator is not None and len(cv) < 10:
        try:
            extra = repair(generator.expand(parent.descriptor, rng))
            for k, v in extra.items():
                if k not in cv:
                    cv[k] = v
                    break
        except Exception:  # noqa: BLE001 — generator hiccup → deterministic hardening still stands
            pass

    desc = parent.descriptor if "higher-stakes" in parent.descriptor else \
        f"{parent.descriptor} (higher-stakes v{round_idx})"
    child = Persona(id=f"{parent.id}#h{round_idx}", descriptor=desc,
                    relationships=copy.deepcopy(parent.relationships),
                    care_vector=repair(cv), context=parent.context)
    ok, _ = validate(child)
    if ok:
        return child
    # Fallback: minimal hardening (one rule only) — recover from an over-aggressive child.
    cv2 = copy.deepcopy(parent.care_vector)
    top = max(cv2, key=lambda k: cv2[k]["sensitivity"])
    cv2[top] = _harden_rule(cv2[top], rng)
    child2 = Persona(id=f"{parent.id}#h{round_idx}", descriptor=desc,
                     relationships=copy.deepcopy(parent.relationships),
                     care_vector=repair(cv2), context=parent.context)
    return child2 if validate(child2)[0] else None


def harden_population(personas: list[Persona], stats: dict, generator, rng: random.Random,
                      round_idx: int, solved_thr: float = 0.5, min_n: int = 3,
                      max_personas: int = 128) -> tuple[list[Persona], list[dict]]:
    """Solved personas (win_rate ≥ solved_thr) spawn harder descendants folded back into the
    population. Bounded by max_personas: superseded (solved, child-bearing) parents retire
    first. Returns (new_population, events) where events log the curriculum step."""
    events, children = [], []
    solved_ids = set()
    for p in personas:
        summ = summarize(stats, p.id)
        if summ and summ["n"] >= min_n and summ["win_rate"] >= solved_thr:
            child = harden_persona(p, generator, rng, round_idx)
            if child is not None:
                children.append(child)
                solved_ids.add(p.id)
                events.append({"event": "harden", "round": round_idx, "parent": p.id,
                               "child": child.id, "win_rate": round(summ["win_rate"], 3),
                               "parent_norms": len(p.care_vector), "child_norms": len(child.care_vector)})

    new_pop = personas + children
    # Bound growth: retire superseded parents (solved + now has a harder child), easiest first.
    if len(new_pop) > max_personas:
        retire_order = sorted(
            (p for p in personas if p.id in solved_ids),
            key=lambda p: summarize(stats, p.id)["win_rate"], reverse=True)
        to_retire = {p.id for p in retire_order[: len(new_pop) - max_personas]}
        if to_retire:
            new_pop = [p for p in new_pop if p.id not in to_retire]
            events.append({"event": "retire", "round": round_idx, "ids": sorted(to_retire)})
    return new_pop, events
