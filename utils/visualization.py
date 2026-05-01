"""Matplotlib visualization helpers for FedCausal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_lines(
    series: Mapping[str, Sequence[tuple[float, float]]],
    output_path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path | None:
    """Plot one or more line series."""
    if not series:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for label, points in series.items():
        if not points:
            continue
        xs, ys = zip(*sorted(points))
        plt.plot(xs, ys, marker="o", label=label)
        plotted = True
    if not plotted:
        plt.close()
        return None
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    output_path: str | Path,
    ylabel: str,
    title: str,
) -> Path | None:
    """Plot a bar chart."""
    if not labels or not values:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    plt.figure(figsize=(max(6, len(labels) * 1.2), 4.5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_grouped_bars(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
    x_key: str,
    y_key: str,
    group_key: str,
    ylabel: str,
    title: str,
) -> Path | None:
    """Plot grouped bars from row dictionaries."""
    rows = list(rows)
    if not rows:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = sorted({str(row[x_key]) for row in rows})
    groups = sorted({str(row[group_key]) for row in rows})
    values = {(str(row[x_key]), str(row[group_key])): float(row[y_key]) for row in rows}
    width = 0.8 / max(len(groups), 1)
    positions = list(range(len(x_values)))

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    plt.figure(figsize=(max(7, len(x_values) * 1.4), 4.5))
    for idx, group in enumerate(groups):
        offset = (idx - (len(groups) - 1) / 2.0) * width
        ys = [values.get((x, group), 0.0) for x in x_values]
        xs = [pos + offset for pos in positions]
        plt.bar(xs, ys, width=width, label=group)

    plt.xticks(positions, x_values, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_mask_heatmap(mask_path: str | Path, output_path: str | Path) -> Path | None:
    """Plot a channel-averaged global mask heatmap from a torch checkpoint."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None

    try:
        import torch
    except ImportError:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = torch.load(mask_path, map_location="cpu")
    mask = payload["global_mask"] if isinstance(payload, dict) and "global_mask" in payload else payload
    heatmap = mask.detach().cpu().squeeze(0).mean(dim=0)
    heatmap = torch.fft.fftshift(heatmap).numpy()

    output_path = Path(output_path)
    _ensure_dir(output_path.parent)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Global FFT Mask")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
