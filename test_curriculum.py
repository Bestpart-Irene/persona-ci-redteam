"""Offline test for the self-generated curriculum loop (CPU, no GPU/API).

Verifies the RSI second loop: solved personas spawn HARDER, still-VALID descendants that
are folded back into the population, growth stays bounded, and unsolved personas survive.
"""

from __future__ import annotations

import random

from persona_gen import MockGenerator, build_population, validate
from curriculum import harden_population, summarize


def _fake_stats(personas, solved_ids, n=8):
    """High win-rate for solved_ids, ~0 for the rest."""
    stats = {}
    for p in personas:
        win = n if p.id in solved_ids else 0
        stats[p.id] = {"n": n, "wins": win, "sum_r": 1.0 * win}
    return stats


def main() -> None:
    rng = random.Random(0)
    gen = MockGenerator()
    pop = build_population(6, gen, seed=0)
    assert all(validate(p)[0] for p in pop), "seed population must be valid"
    base_ids = {p.id for p in pop}

    # Mark half the population as SOLVED → those must spawn harder children.
    solved = {pop[0].id, pop[1].id, pop[2].id}
    stats = _fake_stats(pop, solved)

    new_pop, events = harden_population(pop, stats, gen, rng, round_idx=1,
                                        solved_thr=0.5, min_n=3, max_personas=128)

    children = [p for p in new_pop if p.id not in base_ids]
    hardens = [e for e in events if e["event"] == "harden"]

    # 1) every solved persona produced exactly one child
    assert len(children) == len(solved), f"expected {len(solved)} children, got {len(children)}"
    print(f"  [PASS] {len(children)} solved personas spawned harder descendants")

    # 2) every child is schema-valid (would be rejected by the persona gate otherwise)
    assert all(validate(c)[0] for c in children), "all hardened children must validate"
    print("  [PASS] all hardened children pass the persona validation gate")

    # 3) children are genuinely harder: ≥ parent sensitivity sum, ≥ parent forbidden count
    for e in hardens:
        parent = next(p for p in pop if p.id == e["parent"])
        child = next(c for c in children if c.id == e["child"])
        ps = sum(r["sensitivity"] for r in parent.care_vector.values())
        cs = sum(r["sensitivity"] for r in child.care_vector.values())
        pf = sum(len(r["forbidden_recipients"]) for r in parent.care_vector.values())
        cf = sum(len(r["forbidden_recipients"]) for r in child.care_vector.values())
        assert cs >= ps and cf > pf, f"{e['child']} not strictly harder than {e['parent']}"
    print("  [PASS] children are strictly stricter (sensitivity ↑, forbidden recipients ↑)")

    # 4) unsolved personas survive (still room to climb)
    assert all(pid in {p.id for p in new_pop} for pid in (base_ids - solved)), "unsolved dropped"
    print("  [PASS] unsolved personas retained")

    # 5) growth is bounded: with a tiny cap, superseded parents retire
    capped, ev2 = harden_population(pop, stats, gen, rng, round_idx=2,
                                    solved_thr=0.5, min_n=3, max_personas=len(pop))
    assert len(capped) <= len(pop), f"cap not enforced: {len(capped)} > {len(pop)}"
    assert any(e["event"] == "retire" for e in ev2), "expected a retire event under cap"
    print(f"  [PASS] population bounded under cap ({len(capped)} ≤ {len(pop)}); parents retired")

    # 6) no-op when nothing is solved
    none_pop, none_ev = harden_population(pop, _fake_stats(pop, set()), gen, rng, round_idx=1)
    assert none_pop == pop and not none_ev, "no curriculum change expected with zero solves"
    print("  [PASS] no-op when nothing is solved yet")

    print("ALL PASS ✅")


if __name__ == "__main__":
    main()
