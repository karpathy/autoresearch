"""Progress chart generation from evaluation history.

Generates a matplotlib chart showing discarded (gray dots), kept (green dots),
and running best (green step line) from a SQLite evaluation history database.
"""

import sqlite3


def generate_chart(db_path, output_path, title=None, direction="minimize",
                   score_label="Score"):
    """Generate a progress chart from a history database.

    Args:
        db_path: Path to SQLite history.db (must have an `evaluations` table).
        output_path: Where to save the PNG.
        title: Custom chart title (auto-generated if None).
        direction: "minimize" or "maximize".
        score_label: Y-axis label for the metric.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If the database has no evaluations.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    # Load data
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, score, status, description FROM evaluations ORDER BY id"
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError("No evaluations found in database.")

    # Classify points
    kept_x, kept_y, kept_labels = [], [], []
    disc_x, disc_y = [], []

    for i, (_, score, status, desc) in enumerate(rows):
        if status in ("accepted", "baseline"):
            kept_x.append(i)
            kept_y.append(score)
            kept_labels.append(desc or "")
        elif score is not None:
            disc_x.append(i)
            disc_y.append(score)

    # Build step-line coordinates for running best
    step_x, step_y = [], []
    running = None
    for i, (_, score, status, _) in enumerate(rows):
        if status in ("accepted", "baseline"):
            if running is not None:
                # Horizontal extension to current x at previous best
                step_x.append(i)
                step_y.append(running)
            running = score
            step_x.append(i)
            step_y.append(running)
    # Extend to end of chart
    if running is not None and len(rows) > 1:
        step_x.append(len(rows) - 1)
        step_y.append(running)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 6))

    # Discarded (gray dots)
    ax.scatter(disc_x, disc_y, c="#C0C0C0", s=28, alpha=0.45,
               label="Discarded", zorder=2, linewidths=0)

    # Running best (step line)
    if step_x:
        ax.plot(step_x, step_y, color="#66BB6A", linewidth=2, alpha=0.7,
                label="Running best", zorder=3)

    # Kept (green dots)
    ax.scatter(kept_x, kept_y, c="#43A047", s=55, alpha=0.9,
               label="Kept", zorder=4, edgecolors="white", linewidths=0.5)

    # Italic labels on kept points
    for x, y, label in zip(kept_x, kept_y, kept_labels):
        if label:
            short = (label[:35] + "\u2026") if len(label) > 38 else label
            ax.annotate(
                short, (x, y),
                textcoords="offset points", xytext=(5, 8),
                fontsize=7.5, fontstyle="italic", color="#43A047",
                rotation=30, ha="left", va="bottom",
            )

    # Axes
    dir_str = "lower is better" if direction == "minimize" else "higher is better"
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(f"{score_label} ({dir_str})", fontsize=12)

    # Title
    n_total = len(rows)
    n_kept = sum(1 for r in rows if r[2] == "accepted")
    if title is None:
        title = (f"AutoAnything Progress: {n_total} Experiments, "
                 f"{n_kept} Kept Improvement{'s' if n_kept != 1 else ''}")
    ax.set_title(title, fontsize=14)

    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
