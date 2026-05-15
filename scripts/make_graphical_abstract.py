"""
FERMAT – NEJM-style Graphical Abstract
Generates: graphical_abstract.png (2400 × 1200 px, 200 dpi)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
import matplotlib.patheffects as pe
import numpy as np

# ── Palette ─────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
PANEL   = "#F7F8FA"
BORDER  = "#D0D5DD"
DARK    = "#1A1A2E"
MID     = "#4A5568"
LIGHT   = "#718096"

C_DX   = "#2563EB"   # blue    — diagnoses
C_RX   = "#059669"   # green   — prescriptions
C_PX   = "#7C3AED"   # purple  — procedures
C_LAB  = "#D97706"   # amber   — lab/screening
C_LIF  = "#DB2777"   # pink    — lifestyle
C_DTH  = "#DC2626"   # red     — death
C_SEX  = "#6B7280"   # gray    — sex
ACCENT = "#1D4ED8"

TOKEN_COLORS = {
    "DX": C_DX, "RX": C_RX, "PX": C_PX,
    "LAB": C_LAB, "LIFESTYLE": C_LIF, "DTH": C_DTH,
}

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 12), dpi=100, facecolor=BG)

# Overall title
fig.text(0.5, 0.965,
         "FERMAT: Foundation Model for Multimodal Clinical Trajectory Prediction",
         ha="center", va="top", fontsize=18, fontweight="bold",
         color=DARK, fontfamily="DejaVu Sans")
fig.text(0.5, 0.935,
         "Autoregressive modeling of nationwide Korean health data"
         "  |  Diagnoses · Prescriptions · Procedures · Labs · Lifestyle",
         ha="center", va="top", fontsize=11, color=MID, style="italic")

# ── Panel boundaries (left / middle / right) ─────────────────────────────────
# Each panel is an Axes; we draw everything manually inside.
ax_L = fig.add_axes([0.01, 0.05, 0.30, 0.84])   # Data sources
ax_M = fig.add_axes([0.34, 0.05, 0.33, 0.84])   # FERMAT architecture
ax_R = fig.add_axes([0.69, 0.05, 0.30, 0.84])   # Ablation / validation

for ax in [ax_L, ax_M, ax_R]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Panel border
    rect = FancyBboxPatch((0, 0), 1, 1,
                          boxstyle="round,pad=0.01",
                          linewidth=1.5, edgecolor=BORDER,
                          facecolor=PANEL, transform=ax.transAxes,
                          zorder=0)
    ax.add_patch(rect)


# ════════════════════════════════════════════════════════════════════════════
# LEFT PANEL  –  Data Sources
# ════════════════════════════════════════════════════════════════════════════
ax = ax_L

# Panel label
ax.text(0.5, 0.96, "① Study Population & Data Sources",
        ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)

# Sub-header
ax.text(0.5, 0.90, "Korean National Healthcare Big Data Integration Platform",
        ha="center", va="top", fontsize=9, color=MID, style="italic")

# ── Data source boxes ──────────────────────────────────────────────────────
sources = [
    ("NHIS", "National Health Insurance\nService", "#EFF6FF", C_DX),
    ("HIRA", "Health Insurance Review\n& Assessment Service", "#F0FDF4", C_RX),
    ("KDCA", "Korea Disease Control\n& Prevention Agency", "#FDF4FF", C_PX),
    ("NCC",  "National Cancer Center\n& Death Registry", "#FFF7ED", C_LAB),
    ("CDM",  "University Hospital CDMs\n(Pusan · Chonnam · Kyungpook)", "#FFF1F2", C_LIF),
]

box_h = 0.10
box_y_start = 0.82
gap = 0.025
for i, (abbr, label, fc, ec) in enumerate(sources):
    y = box_y_start - i * (box_h + gap)
    rect = FancyBboxPatch((0.05, y - box_h), 0.9, box_h,
                          boxstyle="round,pad=0.01",
                          linewidth=1.5, edgecolor=ec,
                          facecolor=fc, zorder=2)
    ax.add_patch(rect)
    ax.text(0.13, y - box_h/2, abbr,
            ha="left", va="center", fontsize=11, fontweight="bold", color=ec)
    ax.text(0.31, y - box_h/2, label,
            ha="left", va="center", fontsize=8.5, color=MID)

# Convergence arrow
arrow_y_top = box_y_start - 5 * (box_h + gap) - 0.01
ax.annotate("", xy=(0.5, arrow_y_top - 0.06),
            xytext=(0.5, arrow_y_top),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

# Population stats box
stat_y = arrow_y_top - 0.22
rect2 = FancyBboxPatch((0.05, stat_y), 0.9, 0.16,
                       boxstyle="round,pad=0.015",
                       linewidth=2, edgecolor=ACCENT,
                       facecolor="#EFF6FF", zorder=2)
ax.add_patch(rect2)
ax.text(0.5, stat_y + 0.13, "Study Cohort",
        ha="center", va="top", fontsize=10, fontweight="bold", color=ACCENT)
ax.text(0.5, stat_y + 0.09, "~50 M patients  ·  20+ years",
        ha="center", va="top", fontsize=9.5, color=DARK)
ax.text(0.5, stat_y + 0.05, "Diagnoses · Prescriptions · Procedures",
        ha="center", va="top", fontsize=8.5, color=MID)
ax.text(0.5, stat_y + 0.02, "Lab results · Lifestyle · Mortality",
        ha="center", va="top", fontsize=8.5, color=MID)

# Token-type legend
legend_y = stat_y - 0.10
ax.text(0.5, legend_y, "Token Types",
        ha="center", va="top", fontsize=9, fontweight="bold", color=DARK)
items = [("DX", C_DX), ("RX", C_RX), ("PX", C_PX),
         ("LAB", C_LAB), ("LIFESTYLE", C_LIF), ("DTH", C_DTH)]
cols = 3
for idx, (label, color) in enumerate(items):
    col = idx % cols
    row = idx // cols
    x = 0.10 + col * 0.30
    y = legend_y - 0.045 - row * 0.045
    circ = plt.Circle((x, y), 0.018, color=color, zorder=3,
                      transform=ax.transData)
    ax.add_patch(circ)
    ax.text(x + 0.028, y, label,
            ha="left", va="center", fontsize=8, color=MID)


# ════════════════════════════════════════════════════════════════════════════
# MIDDLE PANEL  –  FERMAT Architecture & Patient Trajectory
# ════════════════════════════════════════════════════════════════════════════
ax = ax_M

ax.text(0.5, 0.96, "② FERMAT Architecture",
        ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)
ax.text(0.5, 0.90, "Multimodal autoregressive transformer on clinical event sequences",
        ha="center", va="top", fontsize=8.5, color=MID, style="italic")

# ── Patient trajectory row ─────────────────────────────────────────────────
traj_y = 0.83
ax.text(0.04, traj_y + 0.03, "Patient trajectory (ordered by age in days):",
        ha="left", va="bottom", fontsize=8.5, color=MID)

events = [
    ("HTN\nDx", "DX",  "age 25y"),
    ("Amlo\nRx", "RX", "age 26y"),
    ("GLU\nQ2",  "LAB", "age 27y"),
    ("EKG\nPx",  "PX", "age 27y"),
    ("GLU\nQ3",  "LAB", "age 28y"),
    ("Smkg\nLS", "LIFESTYLE", "age 28y"),
    ("DM\nDx",   "DX", "age 29y"),
    ("→?",       "DTH", "next?"),
]

n = len(events)
xs = np.linspace(0.05, 0.95, n)
for i, (label, ttype, age) in enumerate(events):
    x = xs[i]
    color = TOKEN_COLORS.get(ttype, C_SEX)
    alpha = 0.35 if ttype == "DTH" else 1.0
    # Token box
    rect = FancyBboxPatch((x - 0.05, traj_y - 0.09), 0.10, 0.09,
                          boxstyle="round,pad=0.008",
                          linewidth=1.5 if ttype != "DTH" else 2.5,
                          edgecolor=color, facecolor=color + "22",
                          alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, traj_y - 0.045, label,
            ha="center", va="center", fontsize=7.5, color=color,
            fontweight="bold", alpha=alpha)
    ax.text(x, traj_y - 0.105, age,
            ha="center", va="top", fontsize=6.5, color=LIGHT)
    # Type chip
    ax.text(x, traj_y - 0.125, ttype,
            ha="center", va="top", fontsize=6, color=color, style="italic")
    # Arrow
    if i < n - 1:
        ax.annotate("", xy=(xs[i+1] - 0.052, traj_y - 0.045),
                    xytext=(x + 0.052, traj_y - 0.045),
                    arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=1))

# ── Embedding layers ───────────────────────────────────────────────────────
emb_y = 0.58
ax.text(0.5, emb_y + 0.02, "Input Representation = TokenEmb + AgeEncoding + TypeEmb",
        ha="center", va="bottom", fontsize=8.5, color=DARK, fontweight="bold")

emb_items = [
    ("Token\nEmbedding", C_DX,   "shared w/ output\n(weight tying)"),
    ("Age\nEncoding",    C_LAB,  "sinusoidal +\nlinear projection"),
    ("Type\nEmbedding",  C_PX,   "DX / RX / PX\nLAB / LS / DTH"),
]
box_w = 0.24
for idx, (label, color, sub) in enumerate(emb_items):
    x = 0.08 + idx * 0.31
    rect = FancyBboxPatch((x, emb_y - 0.14), box_w, 0.13,
                          boxstyle="round,pad=0.01",
                          linewidth=1.5, edgecolor=color,
                          facecolor=color + "18", zorder=2)
    ax.add_patch(rect)
    ax.text(x + box_w/2, emb_y - 0.065, label,
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=color)
    ax.text(x + box_w/2, emb_y - 0.12, sub,
            ha="center", va="center", fontsize=7, color=MID)
    if idx < 2:
        ax.text(x + box_w + 0.015, emb_y - 0.075, "+",
                ha="center", va="center", fontsize=14, color=DARK, fontweight="bold")

# Sum arrow
ax.annotate("", xy=(0.5, emb_y - 0.20),
            xytext=(0.5, emb_y - 0.15),
            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2))

# ── Transformer block ──────────────────────────────────────────────────────
tf_y = 0.24
tf_h = 0.155
rect_tf = FancyBboxPatch((0.08, tf_y - tf_h), 0.84, tf_h,
                         boxstyle="round,pad=0.015",
                         linewidth=2.5, edgecolor=ACCENT,
                         facecolor="#EFF6FF", zorder=2)
ax.add_patch(rect_tf)
ax.text(0.5, tf_y - 0.02, "Causal Transformer",
        ha="center", va="top", fontsize=10, fontweight="bold", color=ACCENT)
ax.text(0.5, tf_y - 0.06, "LayerNorm  →  Causal Self-Attention  →  MLP",
        ha="center", va="top", fontsize=8.5, color=MID)
ax.text(0.5, tf_y - 0.095, "Causal mask + padding mask + co-occurrence mask",
        ha="center", va="top", fontsize=7.5, color=LIGHT, style="italic")
ax.text(0.5, tf_y - 0.125, "×  N  blocks",
        ha="center", va="top", fontsize=8.5, color=MID)

# Output arrows + boxes
out_y = tf_y - tf_h - 0.01
ax.annotate("", xy=(0.27, out_y - 0.09),
            xytext=(0.35, out_y),
            arrowprops=dict(arrowstyle="-|>", color=C_DX, lw=2))
ax.annotate("", xy=(0.73, out_y - 0.09),
            xytext=(0.65, out_y),
            arrowprops=dict(arrowstyle="-|>", color=C_LAB, lw=2))

rect_o1 = FancyBboxPatch((0.04, out_y - 0.22), 0.43, 0.11,
                         boxstyle="round,pad=0.01",
                         linewidth=1.5, edgecolor=C_DX,
                         facecolor="#EFF6FF", zorder=2)
ax.add_patch(rect_o1)
ax.text(0.255, out_y - 0.135, "Next-Token Head",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_DX)
ax.text(0.255, out_y - 0.17, "P(event | history)",
        ha="center", va="center", fontsize=8, color=MID)

rect_o2 = FancyBboxPatch((0.53, out_y - 0.22), 0.43, 0.11,
                         boxstyle="round,pad=0.01",
                         linewidth=1.5, edgecolor=C_LAB,
                         facecolor="#FFF7ED", zorder=2)
ax.add_patch(rect_o2)
ax.text(0.745, out_y - 0.135, "Time-to-Event Head",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_LAB)
ax.text(0.745, out_y - 0.17, "E[Δt | history]",
        ha="center", va="center", fontsize=8, color=MID)


# ════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL  –  Ablation / Validation Strategy
# ════════════════════════════════════════════════════════════════════════════
ax = ax_R

ax.text(0.5, 0.96, "③ Incremental Ablation Design",
        ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)
ax.text(0.5, 0.90, "Isolating the contribution of each data modality",
        ha="center", va="top", fontsize=8.5, color=MID, style="italic")

ablation_steps = [
    ("Step 1", "DX only",
     "Diagnosis codes alone\n(baseline)",
     [("DX", C_DX)], "#EFF6FF", C_DX),
    ("Step 2", "+ Prescriptions",
     "Does drug history\nimprove prediction?",
     [("DX", C_DX), ("RX", C_RX)], "#F0FDF4", C_RX),
    ("Step 3", "+ Procedures",
     "Does procedure history\nadd independent value?",
     [("DX", C_DX), ("RX", C_RX), ("PX", C_PX)], "#F5F3FF", C_PX),
    ("Step 4", "+ Lab (single)",
     "Does a single biomarker\nsnapshot help?",
     [("DX", C_DX), ("RX", C_RX), ("PX", C_PX), ("LAB", C_LAB)],
     "#FFFBEB", C_LAB),
    ("Step 5", "+ Lab (trajectory)",
     "Is biomarker trend better\nthan a single value?",
     [("DX", C_DX), ("RX", C_RX), ("PX", C_PX), ("LAB★", C_LAB)],
     "#FFF7ED", C_LAB),
    ("Step 6", "+ Lifestyle  [Full]",
     "Does behavioral context\nfurther improve outcomes?",
     [("DX", C_DX), ("RX", C_RX), ("PX", C_PX),
      ("LAB★", C_LAB), ("LS", C_LIF)], "#FFF1F2", C_LIF),
]

step_h = 0.108
gap_s  = 0.012
y_top  = 0.87

for i, (step_lbl, title, desc, chips, fc, ec) in enumerate(ablation_steps):
    y = y_top - i * (step_h + gap_s)
    rect = FancyBboxPatch((0.04, y - step_h), 0.92, step_h,
                          boxstyle="round,pad=0.01",
                          linewidth=1.5, edgecolor=ec,
                          facecolor=fc, zorder=2)
    ax.add_patch(rect)

    # Step label
    ax.text(0.10, y - step_h/2, step_lbl,
            ha="center", va="center", fontsize=8, color=ec,
            fontweight="bold")

    # Title
    ax.text(0.23, y - step_h*0.30, title,
            ha="left", va="center", fontsize=8.5, fontweight="bold", color=DARK)
    # Description
    ax.text(0.23, y - step_h*0.70, desc,
            ha="left", va="center", fontsize=7.2, color=MID)

    # Chips
    chip_x = 0.65
    for (chip_lbl, chip_c) in chips:
        w = 0.042 + len(chip_lbl) * 0.008
        chip_rect = FancyBboxPatch((chip_x, y - step_h*0.72),
                                   w, step_h*0.44,
                                   boxstyle="round,pad=0.005",
                                   linewidth=1.2, edgecolor=chip_c,
                                   facecolor=chip_c + "22", zorder=3)
        ax.add_patch(chip_rect)
        ax.text(chip_x + w/2, y - step_h*0.50,
                chip_lbl, ha="center", va="center",
                fontsize=6.5, color=chip_c, fontweight="bold")
        chip_x += w + 0.012

# Key result callout
kres_y = y_top - 6 * (step_h + gap_s) - 0.005
rect_key = FancyBboxPatch((0.04, kres_y - 0.13), 0.92, 0.13,
                          boxstyle="round,pad=0.015",
                          linewidth=2.5, edgecolor="#B45309",
                          facecolor="#FEF3C7", zorder=2)
ax.add_patch(rect_key)
ax.text(0.5, kres_y - 0.025, "★  Key Hypothesis",
        ha="center", va="top", fontsize=9, fontweight="bold", color="#92400E")
ax.text(0.5, kres_y - 0.063,
        "Biomarker trajectory (Step 5) captures",
        ha="center", va="top", fontsize=8.5, color="#78350F")
ax.text(0.5, kres_y - 0.090,
        "pre-diagnostic deterioration invisible",
        ha="center", va="top", fontsize=8.5, color="#78350F")
ax.text(0.5, kres_y - 0.117,
        "to diagnosis-only models",
        ha="center", va="top", fontsize=8.5, color="#78350F")

# Validation status
val_y = kres_y - 0.155
rect_v = FancyBboxPatch((0.04, val_y - 0.105), 0.92, 0.105,
                        boxstyle="round,pad=0.012",
                        linewidth=1.5, edgecolor=BORDER,
                        facecolor="#F9FAFB", zorder=2)
ax.add_patch(rect_v)
ax.text(0.5, val_y - 0.018, "Validation Status",
        ha="center", va="top", fontsize=8.5, fontweight="bold", color=DARK)
status_items = [
    ("✓", "Model architecture & training loop",  "#059669"),
    ("✓", "Smoke test on synthetic 4-col data",   "#059669"),
    ("◎", "Real-data preprocessing (planned)",    "#D97706"),
    ("◎", "CDM Playground deployment (pending)",  "#D97706"),
]
for j, (sym, txt, col) in enumerate(status_items):
    yy = val_y - 0.045 - j * 0.020
    ax.text(0.09, yy, sym, ha="center", va="center",
            fontsize=9, color=col, fontweight="bold")
    ax.text(0.14, yy, txt, ha="left", va="center",
            fontsize=7.5, color=MID)


# ── Inter-panel arrows ───────────────────────────────────────────────────────
# These go on the figure level
for x_pos in [0.315, 0.660]:
    fig.patches.append(
        FancyArrowPatch((x_pos, 0.47), (x_pos + 0.025, 0.47),
                        arrowstyle="-|>", mutation_scale=18,
                        color=ACCENT, lw=2,
                        transform=fig.transFigure, figure=fig)
    )

# ── Footer ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.025,
         "FERMAT v0.1  ·  Foundation model for Exploring Real-world Multimodal health data using "
         "Autoregressive Trajectory modeling  ·  Korean NHIS/HIRA/KDCA/NCC/CDM",
         ha="center", va="bottom", fontsize=8, color=LIGHT, style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = "/home/user/FERMAT/graphical_abstract.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Saved → {out_path}")
plt.close(fig)
