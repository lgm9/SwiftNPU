import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator


def load_cycles_txt(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Supports either:
            #   n cycles
            # or
            #   n grid_size cycles
            n = int(parts[0])
            cycles = float(parts[-1])
            data.append((n, cycles))

    if not data:
        raise ValueError(f"No usable data found in {path}")

    data.sort(key=lambda t: t[0])
    n = np.array([t[0] for t in data], dtype=int)
    cycles = np.array([t[1] for t in data], dtype=float)
    return n, cycles


NO_DEC_N = {5, 8, 11}


def add_top_labels_custom(ax, bars, values, ns):
    for b, v, ni in zip(bars, values, ns):
        fmt = "{:.0f}" if ni in NO_DEC_N else "{:.2f}"
        x = b.get_x() + b.get_width() / 2
        y = b.get_height()
        ax.annotate(
            fmt.format(v),
            xy=(x, y),
            xytext=(0, 1),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0),
            clip_on=False,
        )


def plot_log_only(n, y, ylabel, outname):
    fig, ax = plt.subplots(figsize=(4.8, 2.2))

    x = np.arange(len(n))
    bars = ax.bar(
        x, y, width=0.55,
        color="#dfe8ff",
        edgecolor="black",
        linewidth=1.0
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ni}" for ni in n])
    ax.set_xlabel("n (mesh size)")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")

    ymin = max(min(y[y > 0]) / 2, 1e-12)
    ymax = max(y) * 8.0
    ax.set_ylim(ymin, ymax)

    exp_min = int(np.floor(np.log10(ymin)))
    exp_max = int(np.ceil(np.log10(ymax)))
    start = exp_min if (exp_min % 2 == 0) else (exp_min - 1)
    ticks = [10.0 ** e for e in range(start, exp_max + 1, 2)]

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())

    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.4, alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    add_top_labels_custom(ax, bars, y, n)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.94)

    plt.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"Saved: {outname}")


def main():
    parser = argparse.ArgumentParser(description="Plot GED/vNPU allocation scaling from a txt result file.")
    parser.add_argument("--input", default="results/vNPU.txt", help="Path to GED/vNPU result txt")
    parser.add_argument("--output", default="results/motivation_allocation_scaling.png", help="Output PNG path")
    parser.add_argument("--ghz", type=float, default=2.9, help="CPU clock in GHz for cycle->ms conversion")
    args = parser.parse_args()

    n, cycles = load_cycles_txt(args.input)
    ms = cycles / (args.ghz * 1e6)
    plot_log_only(n, ms, "Time (ms, log scale)", args.output)


if __name__ == "__main__":
    main()
