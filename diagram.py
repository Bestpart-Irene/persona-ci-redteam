"""Render the persona-CI red-team training-loop architecture as a PNG."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

C = {
    "persona": "#6C5CE7",
    "attacker": "#E84393",
    "victim":  "#0984E3",
    "guard":   "#00B894",
    "judge":   "#E17055",
    "reward":  "#FDCB6E",
    "curric":  "#A29BFE",
}

def box(x, y, w, h, title, sub, color, tcolor="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=color, ec="black", lw=1.4, alpha=0.95, zorder=2))
    ax.text(x + w/2, y + h*0.66, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=tcolor, zorder=3)
    ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center",
            fontsize=8, color=tcolor, zorder=3)

def arrow(x1, y1, x2, y2, color="black", style="-|>", lw=2.0, ls="-",
          rad=0.0, label=None, lx=0, ly=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=18, lw=lw, color=color,
                 ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        ax.text((x1+x2)/2 + lx, (y1+y2)/2 + ly, label, ha="center",
                va="center", fontsize=7.5, color=color, style="italic")

# ---- PERSONA (top, defines the target) ----
box(4.2, 7.7, 5.6, 1.0, "PERSONA  ·  structured care vector",
    "info_type → {sensitivity, forbidden/allowed_recipients, forbidden_purposes}",
    C["persona"])

# ---- main 5-stage chain (middle row) ----
y0 = 4.6
box(0.4, y0, 2.4, 1.3, "① ATTACKER", "Qwen3-4B-ablit.\nGRPO + LoRA  (TRAINED)", C["attacker"])
box(3.3, y0, 2.4, 1.3, "② VICTIM", "Llama-3.1-8B-It\nhelpful / aligned", C["victim"])
box(6.2, y0, 2.4, 1.3, "③ LLAMA GUARD 3", "8B 4-bit\nCI policy injected", C["guard"])
box(9.1, y0, 2.4, 1.3, "④ CI JUDGE", "Qwen2.5-32B 4-bit\ncontinuous compromise", C["judge"])
box(11.3, y0-2.0, 2.3, 1.2, "⑤ REWARD", "4-branch state machine\n→ GRPO advantage", C["reward"], tcolor="black")

# persona feeds the chain
arrow(7.0, 7.7, 7.0, 5.95, color=C["persona"], rad=0.0,
      label="defines\n“compromise”", lx=0.95, ly=0)

# left-to-right chain arrows
arrow(2.8, y0+0.65, 3.3, y0+0.65, label="request", ly=0.32)
arrow(5.7, y0+0.65, 6.2, y0+0.65, label="action", ly=0.32)
arrow(8.6, y0+0.65, 9.1, y0+0.65, label="if passes", ly=0.32)

ry = y0 - 2.0  # reward box bottom-ish y
# judge -> reward (compromise branch, down-right)
arrow(11.5, y0, 12.45, ry+1.2, color=C["judge"], rad=-0.15,
      label="compromise\n+0.05 +(0..1)", lx=0.95, ly=0.15)
# guard -> reward (blocked branch) — hug under the boxes
arrow(7.4, y0, 12.35, ry+0.75, color=C["guard"], rad=-0.18,
      label="blocked −0.05", lx=-1.4, ly=0.95)
# victim refuse -> reward
arrow(4.5, y0, 12.3, ry+0.45, color=C["victim"], rad=-0.22,
      label="refused −0.10", lx=-2.6, ly=1.05)

# reward -> attacker (policy update) — big clean arc UNDER everything
arrow(12.4, ry, 1.6, y0, color="#D63031", rad=0.55, lw=2.8,
      style="-|>")
ax.text(7.0, 0.05, "⑤ GRPO policy update  →  attacker weights improve, repeat",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#D63031")

# ---- reward ladder legend (clear of all arrow labels) ----
ladder = ("REWARD LADDER (monotone in subtlety):\n"
          "refused −0.10  <  blocked −0.05  <  benign +0.05  <  "
          "compromise_evaded_guard  +0.05 + compromise(0..1)")
ax.text(4.6, 3.05, ladder, ha="center", va="center", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFF8E1", ec="#FDCB6E", lw=1.2))

# ---- LOOP 2: self-generated curriculum (bottom-left) — closes back onto ① ----
box(0.4, 1.15, 3.7, 1.0, "SELF-GEN CURRICULUM",
    "solved personas → harder\ndescendants (curriculum.py)",
    C["curric"])
arrow(11.3, ry+0.2, 4.1, 1.85, color=C["curric"], rad=0.22, ls="--",
      label="per-persona win-rate", lx=1.3, ly=0.5)
arrow(1.0, 2.15, 4.2, 7.9, color=C["curric"], rad=0.32, ls="--",
      label="harder personas\nrebuild PERSONA pool", lx=-0.7, ly=-0.3)

ax.text(7.0, 8.85, "persona-CI red-team  —  GRPO weight-level RSI loop",
        ha="center", fontsize=14, fontweight="bold")

# legend for arrow meaning
ax.add_line(Line2D([9.3,9.9],[0.55,0.55], color="#D63031", lw=2.6))
ax.text(10.0, 0.55, "recursive weight update", va="center", fontsize=7.5)
ax.add_line(Line2D([9.3,9.9],[0.15,0.15], color=C["curric"], lw=2.0, ls="--"))
ax.text(10.0, 0.15, "self-curriculum (RSI)", va="center", fontsize=7.5)

plt.tight_layout()
plt.savefig("architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote architecture.png")
