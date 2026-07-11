"""Shared plotting style."""
import matplotlib.pyplot as plt

PLOT_RC = {
    "font.family":           "DejaVu Sans",
    "font.size":             14,
    "axes.titlesize":        15,
    "axes.titleweight":      "bold",
    "axes.labelsize":        14,
    "xtick.labelsize":       13,
    "ytick.labelsize":       13,
    "legend.fontsize":       12,
    "legend.title_fontsize": 12,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#cccccc",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        1.3,
    "xtick.major.width":     1.2,
    "ytick.major.width":     1.2,
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "lines.linewidth":       2.2,
    "patch.linewidth":       0.8,
    "axes.grid":             False,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "figure.constrained_layout.use": False,
}

def apply():
    plt.rcParams.update(PLOT_RC)

# ── Shared colours ──────────────────────────────────────────────────────────
PALETTE = {
    "flat_llm":         "#4C72B0",
    "ai_scientist_v2":  "#DD8452",
    "research_agent":   "#55A868",
    "agent_laboratory": "#C44E52",
    "co_scientist":     "#8172B2",
}
AGENT_LABELS = {
    "flat_llm":         "Flat LLM",
    "ai_scientist_v2":  "AI Scientist v2",
    "research_agent":   "ResearchAgent",
    "agent_laboratory": "AgentLaboratory",
    "co_scientist":     "Co-Scientist",
}
AGENTS   = ["flat_llm", "ai_scientist_v2", "research_agent", "agent_laboratory", "co_scientist"]
MARKERS  = {"flat_llm": "o", "ai_scientist_v2": "s",
            "research_agent": "^", "agent_laboratory": "D",
            "co_scientist": "P"}
