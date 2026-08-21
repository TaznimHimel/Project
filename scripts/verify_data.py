"""Check the unified build and quantify the TN3K -> DDTI domain gap.

This is the Week 1-2 deliverable of the proposal: prove the masks are correct by eye,
and put a number on how far apart the two hospitals' images are before any model is
trained. It also answers the risk in section 9 -- whether the gap is caused by
appearance (which style augmentation can fix) or by nodule size (which it cannot).

Run:
    python scripts/verify_data.py
Outputs land in outputs/figures/data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
FIGDIR = ROOT / "outputs" / "figures" / "data"
N_OVERLAY = 12


def load_gray(rel_path: str) -> np.ndarray:
    return cv2.imread(str(ROOT / rel_path), cv2.IMREAD_GRAYSCALE)


# ------------------------------------------------------------------ sanity pass

def check_integrity(df: pd.DataFrame) -> None:
    print("=== integrity ===")
    problems = 0

    missing = [p for p in df["image"] if not (ROOT / p).exists()]
    missing += [p for p in df["mask"] if not (ROOT / p).exists()]
    if missing:
        problems += len(missing)
        print(f"missing files: {len(missing)} (e.g. {missing[:3]})")

    empty = df[df["empty_mask"] == 1]
    if len(empty):
        problems += len(empty)
        print(f"empty masks after resize: {len(empty)}")
        print("  " + ", ".join(empty["sample_id"].head(10)))

    tiny = df[(df["empty_mask"] == 0) & (df["nodule_frac"] < 0.0005)]
    if len(tiny):
        print(f"very small nodules (<0.05% of image): {len(tiny)} "
              f"-- check these by eye, they may be annotation errors")

    # a mask must be strictly binary
    for rel in df["mask"].sample(min(200, len(df)), random_state=0):
        m = load_gray(rel)
        if m is None:
            continue
        vals = np.unique(m)
        if not set(vals.tolist()) <= {0, 255}:
            problems += 1
            print(f"non-binary mask: {rel} values={vals[:6]}")
            break

    print("no problems found" if problems == 0 else f"{problems} problem(s) above")
    print()


# --------------------------------------------------------------------- overlays

def figure_overlays(df: pd.DataFrame, domain: str) -> None:
    sub = df[df["domain"] == domain].sample(
        min(N_OVERLAY, (df["domain"] == domain).sum()), random_state=1)
    cols, rows = 4, int(np.ceil(len(sub) / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, (_, r) in zip(np.ravel(axes), sub.iterrows()):
        img, mask = load_gray(r["image"]), load_gray(r["mask"])
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
        ax.imshow(rgb)
        ax.set_title(f"{r['sample_id']}\n{r['nodule_frac']*100:.1f}% area", fontsize=7)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.suptitle(f"{domain}: image with ground-truth nodule contour", fontsize=11)
    fig.tight_layout()
    out = FIGDIR / f"overlays_{domain}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- the gap plots

def per_image_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean brightness, contrast and speckle-like high-frequency energy per image."""
    recs = []
    for _, r in df.iterrows():
        img = load_gray(r["image"])
        if img is None:
            continue
        x = img.astype(np.float32) / 255.0
        blur = cv2.GaussianBlur(x, (0, 0), 2.0)
        recs.append({
            "domain": r["domain"],
            "mean": float(x.mean()),
            "std": float(x.std()),
            "hf_energy": float(np.abs(x - blur).mean()),   # texture / speckle proxy
            "nodule_frac": r["nodule_frac"],
        })
    return pd.DataFrame(recs)


def figure_domain_gap(stats_df: pd.DataFrame, domains: list[str]) -> None:
    metrics = [("mean", "mean intensity"), ("std", "intensity contrast (std)"),
               ("hf_energy", "high-frequency energy (texture)"),
               ("nodule_frac", "nodule area / image area")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, (key, label) in zip(axes, metrics):
        for d in domains:
            vals = stats_df.loc[stats_df["domain"] == d, key].values
            ax.hist(vals, bins=40, alpha=0.55, density=True, label=d)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Domain gap before any training: appearance statistics and nodule size",
                 fontsize=12)
    fig.tight_layout()
    out = FIGDIR / "domain_gap_histograms.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def table_domain_gap(stats_df: pd.DataFrame, source: str, targets: list[str]) -> None:
    """A two-sample KS test per statistic: one number per row for the report."""
    rows = []
    for target in targets:
        a = stats_df[stats_df["domain"] == source]
        b = stats_df[stats_df["domain"] == target]
        for key in ["mean", "std", "hf_energy", "nodule_frac"]:
            ks = stats.ks_2samp(a[key], b[key])
            rows.append({
                "statistic": key,
                "comparison": f"{source} vs {target}",
                f"{source}_mean": round(float(a[key].mean()), 4),
                "target_mean": round(float(b[key].mean()), 4),
                "ks_distance": round(float(ks.statistic), 4),
                "p_value": f"{ks.pvalue:.2e}",
            })
    table = pd.DataFrame(rows)
    out = FIGDIR / "domain_gap_stats.csv"
    table.to_csv(out, index=False)
    print(f"\n=== domain gap (KS distance: 0 = identical, 1 = no overlap) ===")
    print(table.to_string(index=False))
    print(f"\nwrote {out.relative_to(ROOT)}")

    appearance = table[table["statistic"] != "nodule_frac"]["ks_distance"].max()
    content = table[table["statistic"] == "nodule_frac"]["ks_distance"].max()
    print(f"\nlargest appearance shift: {appearance:.3f}   nodule-size shift: {content:.3f}")
    if content > 0.25:
        print("Nodule sizes also differ, so part of the gap is content, not appearance. "
              "Add scale augmentation and report the two causes separately (proposal, "
              "section 9).")
    else:
        print("Nodule-size distributions are close, so the gap is mainly appearance. "
              "Fourier style randomisation is the right tool.")


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found. Run: python -m src.data.prepare")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)

    print(f"manifest: {len(df)} samples\n")
    print(df.groupby(["domain", "split"]).size().to_string(), "\n")

    check_integrity(df)

    domains = sorted(df["domain"].unique())
    for d in domains:
        figure_overlays(df, d)

    seg_domains = [d for d in domains if d != "tg3k"]   # tg3k is gland, not nodule
    stats_df = per_image_stats(df[df["domain"].isin(seg_domains)])
    figure_domain_gap(stats_df, seg_domains)

    if "tn3k" in seg_domains:
        targets = [d for d in seg_domains if d != "tn3k"]
        if targets:
            table_domain_gap(stats_df, "tn3k", targets)

    print("\nNow open the overlay figures and look at every one. If a contour does not "
          "sit on the nodule, fix the build before training anything.")


if __name__ == "__main__":
    main()
