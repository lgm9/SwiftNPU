import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator


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
            n = int(parts[0])
            cycles = float(parts[-1])
            data.append((n, cycles))

    if not data:
        raise ValueError(f"No usable data found in {path}")

    data.sort(key=lambda t: t[0])
    return data


def align_series(data_a, data_b):
    map_a = {n: cycles for n, cycles in data_a}
    map_b = {n: cycles for n, cycles in data_b}
    common_n = sorted(set(map_a) & set(map_b))
    if not common_n:
        raise ValueError("No overlapping n values between the two result files")
    cycles_a = np.array([map_a[n] for n in common_n], dtype=float)
    cycles_b = np.array([map_b[n] for n in common_n], dtype=float)
    return np.array(common_n, dtype=int), cycles_a, cycles_b


plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.0,
})

VNPU_FILL = "#fbf7e6"
NAS_FILL = "#2b4f81"
NO_DEC_N = {5, 8, 11}


def add_labels(ax, bars, values, ns, kind, y_factor=1.06):
    for b, v, ni in zip(bars, values, ns):
        cx = b.get_x() + b.get_width() / 2
        h = b.get_height()

        if kind == "vnpu":
            txt = f"{v:.0f}" if ni in NO_DEC_N else f"{v:.2f}"
        else:
            txt = f"{v:.2f}"

        ax.text(
            cx, h * y_factor, txt,
            ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                      edgecolor="none", alpha=0),
            clip_on=False
        )


def main():
    parser = argparse.ArgumentParser(description="Plot GED/vNPU vs NAS allocation scaling from txt results.")
    parser.add_argument("--vnpu", default="results/vNPU.txt", help="Path to GED/vNPU result txt")
    parser.add_argument("--nas", default="results/NAS.txt", help="Path to NAS result txt")
    parser.add_argument("--output", default="results/eval_alloc_compare.png", help="Output PNG path")
    parser.add_argument("--ghz", type=float, default=2.9, help="CPU clock in GHz for cycle->ms conversion")
    args = parser.parse_args()

    data_vnpu = load_cycles_txt(args.vnpu)
    data_nas = load_cycles_txt(args.nas)
    n, vnpu_cycles, nas_cycles = align_series(data_vnpu, data_nas)

    ms_vnpu = vnpu_cycles / (args.ghz * 1e6)
    ms_nas = nas_cycles / (args.ghz * 1e6)

    fig, ax = plt.subplots(figsize=(13.0, 3.1))

    x = np.arange(len(n)) * 0.8
    bar_w = 0.24
    gap = 0.0
    off = bar_w / 2 + gap / 2

    bars_v = ax.bar(
        x - off, ms_vnpu, width=bar_w,
        color=VNPU_FILL, edgecolor="black", linewidth=1.1,
        label="vNPU"
    )
    bars_n = ax.bar(
        x + off, ms_nas, width=bar_w,
        color=NAS_FILL, edgecolor="black", linewidth=1.1,
        label="SwiftNPU-NAS"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ni}" for ni in n])
    ax.set_xlabel("N (grid size)")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_yscale("log")

    allv = np.concatenate([ms_vnpu, ms_nas])
    ymin = max(allv[allv > 0].min() / 1.8, 1e-9)
    ymax = allv.max()
    ax.set_ylim(ymin, ymax * 5.0)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())

    def sparse_pow10(y, _pos):
        if y <= 0:
            return ""
        k = np.log10(y)
        if abs(k - round(k)) < 1e-10:
            k = int(round(k))
            return rf"$10^{{{k}}}$" if (k % 2 == 0) else ""
        return ""

    ax.yaxis.set_major_formatter(FuncFormatter(sparse_pow10))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_labels(ax, bars_v, ms_vnpu, n, kind="vnpu", y_factor=1.045)
    add_labels(ax, bars_n, ms_nas, n, kind="nas", y_factor=1.045)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=True, edgecolor="black", facecolor="white"
    )

    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.16, top=0.95)
    plt.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
