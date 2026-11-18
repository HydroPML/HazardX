#!/usr/bin/env python3
"""Create one six-axis vector radar chart per dataset in datasettable.txt.

The table mixes numbers and descriptions, so each axis is converted to a
transparent 1--5 ordinal score.  See ``score_dataset`` for the rubric.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


AXES = [
    "Modality\nrichness",
    "Temporal\nrichness",
    "Spatial\nresolution",
    "Dataset\nscale",
    "Event / region\ndiversity",
    "Annotation / task\nvalue",
]

COLORS = {"Flood": "#2878B5", "Landslide": "#D97904", "Wildfire": "#C83E4D"}


def clean_latex(text: str) -> str:
    """Turn the small subset of LaTeX used in the source table into plain text."""
    text = re.sub(r"\\cite\{[^}]+\}", "", text)
    text = text.replace("$\\sim$", "~").replace("$>$", ">").replace("$^2$", "²")
    text = text.replace("$0.25^{\\circ}$", "0.25 deg")
    text = text.replace("--", "–").replace("\\&", "&")
    return re.sub(r"\s+", " ", text).strip()


def parse_table(path: Path) -> list[dict[str, str]]:
    """Parse the regular seven-column LaTeX rows in datasettable.txt."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows, hazard = [], None
    hazards = {"FLOOD": "Flood", "LANDSLIDE": "Landslide", "WILDFIRE": "Wildfire"}
    i = 0
    while i < len(lines):
        marker = re.search(r"={4,}\s*(FLOOD|LANDSLIDE|WILDFIRE)\s*=+", lines[i])
        if marker:
            hazard = hazards[marker.group(1)]
        line = lines[i].strip()
        # Dataset rows always begin with &, followed by six one-line fields.
        if hazard and line.startswith("& ") and "\\cite{" in line:
            fields = [line[1:].strip()]
            j = i + 1
            while j < len(lines) and len(fields) < 6:
                value = lines[j].strip()
                if value.startswith("&"):
                    value = value[1:].strip()
                    if value.endswith(r"\\"):
                        value = value[:-2].strip()
                    fields.append(value)
                j += 1
            if len(fields) != 6:
                raise ValueError(f"Incomplete dataset row near line {i + 1}")
            keys = ["dataset", "inputs", "temporal", "gsd", "scale", "task"]
            row = {k: clean_latex(v) for k, v in zip(keys, fields)}
            row["hazard"] = hazard
            rows.append(row)
            i = j - 1
        i += 1
    if not rows:
        raise ValueError(f"No dataset rows found in {path}")
    return rows


def clamp_round(value: float) -> int:
    return int(np.clip(np.floor(value + 0.5), 1, 5))


def best_gsd_metres(gsd: str) -> float | None:
    s = gsd.lower().replace(",", "")
    if "cm-scale" in s:
        return 0.1
    if "variable" in s or s.strip() in {"–", "-"}:
        return None
    if "deg" in s:
        return 27_750.0
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not vals:
        return None
    value = min(vals)
    return value * 1000 if "km" in s else value


def largest_quantity(scale: str) -> float:
    """Largest stated count; area/year spans are deliberately not sample counts."""
    s = scale.lower().replace(",", "")
    if "million" in s:
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        return float(nums[0]) * 1_000_000 if nums else 1
    vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    return max(vals, default=1.0)


def score_dataset(row: dict[str, str], all_sizes: list[float]) -> list[int]:
    inp, temp, gsd, scale, task = (row[k].lower() for k in ("inputs", "temporal", "gsd", "scale", "task"))

    # 1) distinct input families, capped at 5 (semicolon-delimited in the table).
    modality = min(5, max(1, len([x for x in inp.split(";") if x.strip()])))

    # 2) temporal design: static/post=1, paired=3, event/daily=4, time series=5.
    temporal = 1
    if any(x in temp for x in ("pre/post", "2 pre", "cross-modal")): temporal = 3
    if any(x in temp for x in ("event-based", "near-event", "during-event", "daily", "8-day")): temporal = max(temporal, 4)
    if "time series" in temp or "t →" in temp: temporal = 5

    # 3) best nominal GSD, logarithmically mapped: <=1 m gets 5; >=1 km gets 1.
    metres = best_gsd_metres(gsd)
    resolution = 2 if metres is None else clamp_round(5 - 4 * np.clip(math.log10(max(metres, 1)) / 3, 0, 1))

    # 4) largest stated quantity, log-normalized across this table.
    size = largest_quantity(scale)
    lo, hi = math.log10(min(all_sizes)), math.log10(max(all_sizes))
    dataset_scale = clamp_round(1 + 4 * (math.log10(size) - lo) / (hi - lo))

    # 5) event/region diversity from explicit counts and global/cross-region cues.
    diversity = 1
    # Only counts explicitly tied to geographic/event units qualify; image,
    # patch and sequence counts describe volume rather than diversity.
    diversity_counts = re.findall(
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:events?|regions?|sites?|landslides?|burns?|pairs?)",
        scale,
    )
    count = max((float(x.replace(",", "")) for x in diversity_counts), default=1)
    if count >= 10: diversity = 2
    if count >= 20: diversity = 3
    if count >= 60: diversity = 4
    if "global" in scale or "global" in task or "cross-region" in task or count >= 100: diversity = 5

    # 6) dense/manual/pixel labels and multi-purpose tasks receive higher utility.
    annotation = 2
    if any(x in task for x in ("segmentation", "delineation", "detection", "extraction", "mapping")): annotation = 3
    if any(x in task for x in ("pixel-level", "manual label", "fine-grained", "severity", "progression", "spread")): annotation = 4
    if any(x in task for x in ("cross-domain", "cross-event", "benchmark", "physical priors")): annotation = 5
    return [modality, temporal, resolution, dataset_scale, diversity, annotation]


def select_font() -> str:
    names = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" in names:
        return "Times New Roman"
    print("Warning: Times New Roman is not installed; using metric-compatible Liberation Serif.")
    return "Liberation Serif"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def draw_radar(
    row: dict[str, str], scores: list[int], output: Path, *, no_text: bool = False
) -> None:
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    closed_angles, closed_scores = angles + angles[:1], scores + scores[:1]
    color = COLORS[row["hazard"]]
    fig, ax = plt.subplots(figsize=(6.2, 6.0), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    if no_text:
        ax.set_xticks(angles)
        ax.set_xticklabels([])
    else:
        ax.set_xticks(angles, AXES, fontsize=10)
        ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([] if no_text else ["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_rlabel_position(12)
    ax.grid(color="#A8A8A8", linewidth=0.65, alpha=0.75)
    ax.spines["polar"].set_color("#777777")
    ax.plot(closed_angles, closed_scores, color=color, linewidth=2.0)
    ax.fill(closed_angles, closed_scores, color=color, alpha=0.20)
    ax.scatter(angles, scores, color=color, s=30, zorder=3)
    if no_text:
        fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
    else:
        ax.set_title(row["dataset"], fontsize=16, fontweight="bold", pad=26)
        fig.text(0.5, 0.025, f'{row["hazard"]} dataset · ordinal score (1–5)', ha="center", fontsize=9, color="#444444")
        fig.subplots_adjust(left=0.16, right=0.84, top=0.83, bottom=0.15)
    stem = safe_name(row["dataset"])
    fig.savefig(output / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("datasettable.txt"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("dataset_radars"))
    parser.add_argument(
        "--no-text-output",
        type=Path,
        default=Path(__file__).with_name("dataset_radars_no_text"),
        help="directory for versions without any surrounding text",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.no_text_output.mkdir(parents=True, exist_ok=True)
    font = select_font()
    plt.rcParams.update({
        "font.family": font,
        "font.serif": [font],
        "svg.fonttype": "none",       # editable text in SVG
        "pdf.fonttype": 42,            # TrueType embedding in PDF
        "axes.unicode_minus": False,
    })
    rows = parse_table(args.input)
    sizes = [largest_quantity(r["scale"]) for r in rows]
    scored = []
    for row in rows:
        scores = score_dataset(row, sizes)
        draw_radar(row, scores, args.output)
        draw_radar(row, scores, args.no_text_output, no_text=True)
        score_names = ["modality", "temporal", "spatial_resolution", "dataset_scale", "event_region_diversity", "annotation_task_value"]
        scored.append({"hazard": row["hazard"], "dataset": row["dataset"], **dict(zip(score_names, scores))})
    with (args.output / "radar_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scored[0].keys())
        writer.writeheader()
        writer.writerows(scored)
    print(f"Created {len(rows)} labeled radar charts (SVG + PDF) in {args.output}")
    print(f"Created {len(rows)} no-text radar charts (SVG + PDF) in {args.no_text_output}")


if __name__ == "__main__":
    main()
