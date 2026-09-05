#!/usr/bin/env python3
"""yawpose データセットの yaw / pitch / roll 分布を 1 枚の画像に出力する。

値の出所:
  - yaw  : labels_fixed.jsonl の yaw_deg(確定ラベル、全 35,713 件)
  - pitch: s002/s003/s004 は generation intent(pitch_deg)、
           s001 は intent が無いため sixdrepnet360 の測定値(qa_sixd.jsonl)で補完
  - roll : 全ソースとも生成 intent は 0 のため、sixdrepnet360 の測定値を表示
           (準背面では sixd の信頼性が落ちる点に注意)

出力: data/yawpose/ypr_hist.png(3 段、ソース別スタック)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"
SOURCES = ["synthetic_001", "synthetic_002", "synthetic_003", "synthetic_004", "synthetic_005", "synthetic_006"]
COLORS = {"synthetic_001": "#4878cf", "synthetic_002": "#e8a33d",
          "synthetic_003": "#6acc65", "synthetic_004": "#d65f5f",
          "synthetic_005": "#9467bd", "synthetic_006": "#8c564b"}


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def stacked_hist(ax: plt.Axes, values: dict[str, np.ndarray], bins: np.ndarray,
                 title: str, xlabel: str) -> None:
    bottom = np.zeros(len(bins) - 1)
    width = (bins[1] - bins[0]) * 0.9
    for s in SOURCES:
        h, _ = np.histogram(values[s], bins=bins)
        ax.bar(bins[:-1], h, width=width, align="edge", bottom=bottom,
               color=COLORS[s], label=s, edgecolor="none")
        bottom += h
    n = int(sum(len(v) for v in values.values()))
    ax.set_title(f"{title} (n={n:,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.grid(axis="y", alpha=0.3)


def main() -> None:
    rows = [json.loads(l) for l in open(OUT / "labels_fixed.jsonl")]
    qa = {q["image"]: q for q in (json.loads(l) for l in open(OUT / "qa_sixd.jsonl"))}

    yaw: dict[str, list[float]] = {s: [] for s in SOURCES}
    pitch: dict[str, list[float]] = {s: [] for s in SOURCES}
    roll: dict[str, list[float]] = {s: [] for s in SOURCES}
    for r in rows:
        s = r["source"]
        yaw[s].append(r["yaw_deg"] % 360.0)
        p = r.get("pitch_deg")
        if p is None:  # s001: intent なし → sixd 測定で補完
            q = qa.get(r["image"])
            if q is not None:
                pitch[s].append(float(q["sixd_pitch"]))
        else:
            pitch[s].append(float(p))
        q = qa.get(r["image"])
        if q is not None:
            roll[s].append(float(q["sixd_roll"]))

    yaw_a = {s: np.array(v) for s, v in yaw.items()}
    pitch_a = {s: np.array(v) for s, v in pitch.items()}
    roll_a = {s: np.array(v) for s, v in roll.items()}

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=150)
    stacked_hist(axes[0], yaw_a, np.arange(0, 361, 10),
                 "yaw (labels_fixed)", "yaw (deg)  [0 = frontal, +90 = facing viewer-left]")
    axes[0].set_xlim(0, 360)
    axes[0].set_xticks(np.arange(0, 361, 30))
    stacked_hist(axes[1], pitch_a, np.arange(-120, 121, 5),
                 "pitch (intent for s002-004, sixd for s001)", "pitch (deg)  [+ = up]")
    axes[1].set_xlim(-120, 120)
    stacked_hist(axes[2], roll_a, np.arange(-60, 61, 5),
                 "roll (sixd measurement; intent is 0 for all sources)", "roll (deg)")
    axes[2].set_xlim(-60, 60)
    # 凡例はバーと重なるためプロット領域の上(タイトルとの間)に横一列で出す
    axes[0].legend(ncol=len(SOURCES), loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10,
                   columnspacing=1.2, handlelength=1.4)
    axes[0].set_title(axes[0].get_title(), pad=34)
    fig.suptitle(f"yawpose yaw / pitch / roll distributions (n={len(rows):,})",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = OUT / "ypr_hist.png"
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


if __name__ == "__main__":
    main()
