import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

# More permissive than the old parser:
# It supports folders such as:
#   vgg_8obj_rotation
#   vgg_8obj_rotation_uniform
#   vgg_8obj_rotation_foveated
#   vgg_8obj_rotation_compare/uniform/...
#   vgg_8obj_rotation_compare/foveated/...
#   conv_gist_6obj_blur
#   conv_gist_mlp_8obj_noise
#   vgg_gist_imagenet64_64_8obj_rotation
#   vgg_gist_imagenet64_224_8obj_rotation
RUN_RE = re.compile(
    r"(?P<model>vgg_gist_imagenet64_64|vgg_gist_imagenet64_224|vgg_gist_pretrained|conv_gist_mlp|conv_gist|vgg)"
    r"_(?P<nobj>6|8)obj_"
    r"(?P<transform>baseline|rotation|scale|skew_x|skew_y|skewx|skewy|noise|blur)"
    r"(?P<suffix>.*)$"
)

MODEL_ORDER = [
    "vgg",
    "vgg_gist_pretrained",
    "vgg_gist_imagenet64_64",
    "vgg_gist_imagenet64_224",
    "conv_gist",
    "conv_gist_mlp",
]

TRANSFORM_ORDER = ["rotation", "scale", "skew_x", "skew_y", "noise", "blur"]
VISION_ORDER = ["uniform", "foveated", "unknown"]

MODEL_LABELS = {
    "vgg": "IVSN / VGG",
    "vgg_gist_pretrained": "VGG-GIST",
    "vgg_gist_imagenet64_64": "VGG-GIST-64 (input 64)",
    "vgg_gist_imagenet64_224": "VGG-GIST-64 (input 224)",
    "conv_gist": "ConvGist",
    "conv_gist_mlp": "ConvGistMLP",
}

TRANSFORM_LABELS = {
    "rotation": "Rotation",
    "scale": "Scale",
    "skew_x": "Skew X",
    "skew_y": "Skew Y",
    "noise": "Noise",
    "blur": "Blur",
    "baseline": "Baseline",
}

X_LABELS = {
    "rotation": "Rotation (deg)",
    "scale": "Scale factor",
    "skew_x": "Skew X (deg)",
    "skew_y": "Skew Y (deg)",
    "noise": "Noise std",
    "blur": "Blur radius",
    "baseline": "Condition value",
}

VISION_LABELS = {
    "uniform": "Uniform",
    "foveated": "Foveated",
    "unknown": "Unknown",
}

METRICS = {
    "mean_fixations_all": {
        "label": "Mean fixations",
        "delta_label": "Δ Mean fixations",
        "kind": "higher_worse",
        "fixed_ylim": None,
    },
    "found_within_3_fixations_rate_all": {
        "label": "Found ≤3 (%)",
        "delta_label": "Δ Found ≤3 (%)",
        "kind": "higher_better",
        "fixed_ylim": (0, 100),
    },
    "top1_rate_all": {
        "label": "Top-1 (%)",
        "delta_label": "Δ Top-1 (%)",
        "kind": "higher_better",
        "fixed_ylim": (0, 100),
    },
    "mean_score_target_all": {
        "label": "Target similarity",
        "delta_label": "Δ Target similarity",
        "kind": "higher_better",
        "fixed_ylim": None,
    },
    "mean_score_max_distractor_all": {
        "label": "Max distractor similarity",
        "delta_label": "Δ Max distractor similarity",
        "kind": "higher_worse",
        "fixed_ylim": None,
    },
    "mean_p_target_all": {
        "label": "Target softmax prob. (%)",
        "delta_label": "Δ Target softmax prob. (%)",
        "kind": "higher_better",
        "fixed_ylim": (0, 100),
    },
}

MAIN_METRICS = [
    "mean_fixations_all",
    "found_within_3_fixations_rate_all",
    "top1_rate_all",
]

COLORS_MODEL = {
    "vgg": "#1f77b4",
    "vgg_gist_pretrained": "#ff7f0e",
    "vgg_gist_imagenet64_64": "#9467bd",
    "vgg_gist_imagenet64_224": "#8c564b",
    "conv_gist": "#2ca02c",
    "conv_gist_mlp": "#d62728",
}

COLORS_VISION = {
    "uniform": "#1f77b4",
    "foveated": "#d62728",
    "unknown": "#7f7f7f",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="plots_foveated")
    parser.add_argument(
        "--recursive-depth",
        type=int,
        default=5,
        help="How deep to search for grouped_summary.csv files below results-root.",
    )
    return parser.parse_args()


# ============================================================
# LOADING
# ============================================================

def normalize_transform_name(name: str) -> str:
    if name == "skewx":
        return "skew_x"
    if name == "skewy":
        return "skew_y"
    return name


def infer_vision_type(path: Path, df: pd.DataFrame) -> str:
    # Prefer explicit column from the new foveated script.
    if "foveated_search" in df.columns:
        vals = df["foveated_search"].dropna().astype(str).str.lower().unique().tolist()
        if any(v in ["true", "1", "yes"] for v in vals):
            return "foveated"
        if any(v in ["false", "0", "no"] for v in vals):
            return "uniform"

    # Otherwise infer from path names.
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)

    if "foveated" in joined or "fovea" in joined:
        return "foveated"
    if "uniform" in joined or "baseline" in joined or "nofovea" in joined or "no_fovea" in joined:
        return "uniform"

    return "unknown"


def parse_run_identity(csv_path: Path) -> Optional[Tuple[str, int, str]]:
    # Try from all parent folder names, starting nearest to the csv and moving upward.
    for folder in [csv_path.parent] + list(csv_path.parents):
        m = RUN_RE.match(folder.name)
        if m:
            model = m.group("model")
            nobj = int(m.group("nobj"))
            transform = normalize_transform_name(m.group("transform"))
            return model, nobj, transform

    return None


def candidate_grouped_summaries(results_root: Path, max_depth: int) -> List[Path]:
    # Path.rglob is simple and robust. max_depth avoids accidentally scanning huge unrelated trees.
    root_depth = len(results_root.parts)
    paths = []
    for p in results_root.rglob("grouped_summary.csv"):
        depth = len(p.parts) - root_depth
        if depth <= max_depth + 1:
            paths.append(p)
    return sorted(paths)


def load_runs(results_root: Path, max_depth: int) -> Dict[Tuple[str, int, str, str], pd.DataFrame]:
    runs: Dict[Tuple[str, int, str, str], pd.DataFrame] = {}
    seen = []

    for csv_path in candidate_grouped_summaries(results_root, max_depth=max_depth):
        ident = parse_run_identity(csv_path)
        if ident is None:
            print(f"Skipping {csv_path}: could not infer model/n_objects/transform from path")
            continue

        model, nobj, transform = ident

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path}: failed to read CSV: {e}")
            continue

        if "condition_value" not in df.columns:
            print(f"Skipping {csv_path}: no condition_value column")
            continue

        vision = infer_vision_type(csv_path, df)

        df = df.sort_values("condition_value").reset_index(drop=True)
        df["model"] = model
        df["model_label"] = MODEL_LABELS.get(model, model)
        df["n_objects"] = nobj
        df["transform"] = transform
        df["transform_label"] = TRANSFORM_LABELS.get(transform, transform)
        df["vision"] = vision
        df["vision_label"] = VISION_LABELS.get(vision, vision)
        df["csv_path"] = str(csv_path)
        df["run_folder"] = csv_path.parent.name

        key = (model, nobj, transform, vision)

        # If duplicate keys exist, keep the one with more rows; if tied, keep the shorter path.
        if key in runs:
            old = runs[key]
            replace = len(df) > len(old) or (len(df) == len(old) and len(str(csv_path)) < len(str(old["csv_path"].iloc[0])))
            if replace:
                print(f"Replacing duplicate run for {key}: {old['csv_path'].iloc[0]} -> {csv_path}")
                runs[key] = df
            else:
                print(f"Ignoring duplicate run for {key}: {csv_path}")
        else:
            runs[key] = df

        seen.append({
            "model": model,
            "n_objects": nobj,
            "transform": transform,
            "vision": vision,
            "csv_path": str(csv_path),
            "n_rows": len(df),
        })

    return runs


def existing_models(runs) -> List[str]:
    found = sorted({k[0] for k in runs.keys()}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 999)
    return found


def existing_n_objects(runs) -> List[int]:
    return sorted({k[1] for k in runs.keys()})


def existing_transforms(runs) -> List[str]:
    found = sorted({k[2] for k in runs.keys()}, key=lambda t: TRANSFORM_ORDER.index(t) if t in TRANSFORM_ORDER else 999)
    return found


def existing_visions(runs) -> List[str]:
    found = sorted({k[3] for k in runs.keys()}, key=lambda v: VISION_ORDER.index(v) if v in VISION_ORDER else 999)
    return found


# ============================================================
# METRIC HELPERS
# ============================================================

def is_percent_metric(metric: str) -> bool:
    return "rate" in metric or "top1" in metric or metric == "mean_p_target_all"


def metric_values_for_plot(metric: str, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if is_percent_metric(metric):
        return 100.0 * values
    return values


def get_metric_label(metric: str, delta: bool = False) -> str:
    if metric not in METRICS:
        return metric
    return METRICS[metric]["delta_label" if delta else "label"]


def available_metrics(runs) -> List[str]:
    cols = set()
    for df in runs.values():
        cols.update(df.columns)
    return [m for m in METRICS.keys() if m in cols]


def infer_baseline_x(transform: str) -> float:
    if transform == "scale":
        return 1.0
    return 0.0


def get_baseline_value(df: pd.DataFrame, transform: str, metric: str) -> float:
    baseline_x = infer_baseline_x(transform)
    idx = (df["condition_value"] - baseline_x).abs().idxmin()
    return float(df.loc[idx, metric])


def compute_ylim_for_metric(runs, metric: str) -> Optional[Tuple[float, float]]:
    vals = []
    for df in runs.values():
        if metric in df.columns:
            vals.extend(metric_values_for_plot(metric, df[metric].dropna().to_numpy(dtype=float)).tolist())

    if not vals:
        return None

    fixed = METRICS.get(metric, {}).get("fixed_ylim", None)
    if fixed is not None:
        return fixed

    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(mn) or not np.isfinite(mx):
        return None
    if mn == mx:
        pad = 0.1 * (abs(mx) + 1e-6)
    else:
        pad = 0.08 * (mx - mn)
    return mn - pad, mx + pad


# ============================================================
# PLOTS: DETAILED CURVES
# ============================================================

def plot_transform_metric_by_model(
    runs,
    n_objects: int,
    transform: str,
    vision: str,
    metric: str,
    out_path: Path,
):
    plt.figure(figsize=(8.8, 4.8))
    any_line = False

    for model in MODEL_ORDER:
        key = (model, n_objects, transform, vision)
        if key not in runs:
            continue

        df = runs[key]
        if metric not in df.columns:
            continue

        x = df["condition_value"].to_numpy(dtype=float)
        y = metric_values_for_plot(metric, df[metric].to_numpy(dtype=float))

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            label=MODEL_LABELS.get(model, model),
            color=COLORS_MODEL.get(model, None),
        )
        any_line = True

    if not any_line:
        plt.close()
        return

    plt.title(f"{TRANSFORM_LABELS.get(transform, transform)} — {n_objects} objects — {VISION_LABELS.get(vision, vision)}")
    plt.xlabel(X_LABELS.get(transform, "Condition value"))
    plt.ylabel(get_metric_label(metric))
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    ylim = compute_ylim_for_metric(runs, metric)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_uniform_vs_foveated(
    runs,
    model: str,
    n_objects: int,
    transform: str,
    metric: str,
    out_path: Path,
):
    key_u = (model, n_objects, transform, "uniform")
    key_f = (model, n_objects, transform, "foveated")

    if key_u not in runs or key_f not in runs:
        return

    plt.figure(figsize=(7.8, 4.6))

    for vision, marker, linestyle in [("uniform", "o", "-"), ("foveated", "s", "--")]:
        df = runs[(model, n_objects, transform, vision)]
        if metric not in df.columns:
            continue

        x = df["condition_value"].to_numpy(dtype=float)
        y = metric_values_for_plot(metric, df[metric].to_numpy(dtype=float))

        plt.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            color=COLORS_VISION[vision],
            label=VISION_LABELS[vision],
        )

    plt.title(f"{MODEL_LABELS.get(model, model)} — {TRANSFORM_LABELS.get(transform, transform)} — {n_objects} objects")
    plt.xlabel(X_LABELS.get(transform, "Condition value"))
    plt.ylabel(get_metric_label(metric))
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    ylim = compute_ylim_for_metric({key_u: runs[key_u], key_f: runs[key_f]}, metric)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_foveation_delta(
    runs,
    model: str,
    n_objects: int,
    transform: str,
    metric: str,
    out_path: Path,
):
    key_u = (model, n_objects, transform, "uniform")
    key_f = (model, n_objects, transform, "foveated")

    if key_u not in runs or key_f not in runs:
        return

    df_u = runs[key_u]
    df_f = runs[key_f]

    if metric not in df_u.columns or metric not in df_f.columns:
        return

    merged = pd.merge(
        df_u[["condition_value", metric]],
        df_f[["condition_value", metric]],
        on="condition_value",
        suffixes=("_uniform", "_foveated"),
    ).sort_values("condition_value")

    if merged.empty:
        return

    x = merged["condition_value"].to_numpy(dtype=float)
    delta = merged[f"{metric}_foveated"].to_numpy(dtype=float) - merged[f"{metric}_uniform"].to_numpy(dtype=float)
    delta = metric_values_for_plot(metric, delta)

    plt.figure(figsize=(7.8, 4.6))
    plt.axhline(0, color="black", linewidth=1.2, linestyle=":")

    width = (x[1] - x[0]) * 0.6 if len(x) > 1 else 0.25
    plt.bar(x, delta, width=width, color="#d62728", alpha=0.85)

    plt.title(f"Foveation cost — {MODEL_LABELS.get(model, model)} — {TRANSFORM_LABELS.get(transform, transform)} — {n_objects} objects")
    plt.xlabel(X_LABELS.get(transform, "Condition value"))
    plt.ylabel(get_metric_label(metric, delta=True))
    plt.grid(True, axis="y", alpha=0.3)

    if len(delta) > 0:
        mx = float(np.nanmax(np.abs(delta)))
        if np.isfinite(mx) and mx > 0:
            plt.ylim(-1.15 * mx, 1.15 * mx)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# SUMMARY TABLES / HEATMAPS
# ============================================================

def summarize_runs(runs, metrics: List[str]) -> pd.DataFrame:
    rows = []

    for (model, nobj, transform, vision), df in runs.items():
        row = {
            "model": model,
            "model_label": MODEL_LABELS.get(model, model),
            "n_objects": nobj,
            "transform": transform,
            "transform_label": TRANSFORM_LABELS.get(transform, transform),
            "vision": vision,
            "vision_label": VISION_LABELS.get(vision, vision),
            "csv_path": df["csv_path"].iloc[0],
        }

        for metric in metrics:
            if metric not in df.columns:
                continue

            vals = df[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue

            row[f"{metric}_avg"] = float(np.nanmean(vals))
            row[f"{metric}_baseline"] = get_baseline_value(df, transform, metric)

            kind = METRICS[metric]["kind"]
            if kind == "higher_worse":
                row[f"{metric}_worst"] = float(np.nanmax(vals))
                row[f"{metric}_degradation"] = row[f"{metric}_worst"] - row[f"{metric}_baseline"]
            elif kind == "higher_better":
                row[f"{metric}_worst"] = float(np.nanmin(vals))
                row[f"{metric}_degradation"] = row[f"{metric}_baseline"] - row[f"{metric}_worst"]

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_foveation_costs(runs, metrics: List[str]) -> pd.DataFrame:
    rows = []

    for model in MODEL_ORDER:
        for nobj in [6, 8]:
            for transform in TRANSFORM_ORDER:
                key_u = (model, nobj, transform, "uniform")
                key_f = (model, nobj, transform, "foveated")
                if key_u not in runs or key_f not in runs:
                    continue

                df_u = runs[key_u]
                df_f = runs[key_f]

                for metric in metrics:
                    if metric not in df_u.columns or metric not in df_f.columns:
                        continue

                    merged = pd.merge(
                        df_u[["condition_value", metric]],
                        df_f[["condition_value", metric]],
                        on="condition_value",
                        suffixes=("_uniform", "_foveated"),
                    )
                    if merged.empty:
                        continue

                    delta = merged[f"{metric}_foveated"].to_numpy(dtype=float) - merged[f"{metric}_uniform"].to_numpy(dtype=float)

                    idx_abs = int(np.nanargmax(np.abs(delta)))
                    rows.append({
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "n_objects": nobj,
                        "transform": transform,
                        "transform_label": TRANSFORM_LABELS.get(transform, transform),
                        "metric": metric,
                        "metric_label": get_metric_label(metric),
                        "mean_delta_foveated_minus_uniform": float(np.nanmean(delta)),
                        "worst_abs_delta_foveated_minus_uniform": float(delta[idx_abs]),
                        "condition_value_at_worst_abs_delta": float(merged["condition_value"].iloc[idx_abs]),
                    })

    return pd.DataFrame(rows)


def ordered_labels(values: List[str], order: List[str], labels: Dict[str, str]) -> List[str]:
    ordered = [v for v in order if v in values]
    ordered += [v for v in values if v not in ordered]
    return [labels.get(v, v) for v in ordered]


def plot_summary_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    n_objects: int,
    vision: str,
    value_type: str,
    out_path: Path,
):
    sub = summary_df[(summary_df["n_objects"] == n_objects) & (summary_df["vision"] == vision)].copy()
    col = f"{metric}_{value_type}"
    if sub.empty or col not in sub.columns:
        return

    models_present = [m for m in MODEL_ORDER if m in set(sub["model"])]
    transforms_present = [t for t in TRANSFORM_ORDER if t in set(sub["transform"])]

    if not models_present or not transforms_present:
        return

    pivot = sub.pivot_table(index="model_label", columns="transform_label", values=col, aggfunc="mean")
    pivot = pivot.reindex(index=[MODEL_LABELS.get(m, m) for m in models_present])
    pivot = pivot.reindex(columns=[TRANSFORM_LABELS.get(t, t) for t in transforms_present])

    arr = pivot.to_numpy(dtype=float)
    if is_percent_metric(metric):
        arr_plot = 100.0 * arr
    else:
        arr_plot = arr

    plt.figure(figsize=(max(7.0, 1.2 * len(transforms_present) + 3), max(3.2, 0.45 * len(models_present) + 2)))
    im = plt.imshow(arr_plot, aspect="auto")

    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"{value_type.capitalize()} — {get_metric_label(metric)} — {n_objects} objects — {VISION_LABELS.get(vision, vision)}")
    cbar = plt.colorbar(im)
    cbar.set_label(get_metric_label(metric))

    for i in range(arr_plot.shape[0]):
        for j in range(arr_plot.shape[1]):
            val = arr_plot[i, j]
            if not np.isfinite(val):
                continue
            txt = f"{val:.1f}" if is_percent_metric(metric) else f"{val:.2f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_foveation_cost_heatmap(
    cost_df: pd.DataFrame,
    metric: str,
    n_objects: int,
    out_path: Path,
    value_col: str = "mean_delta_foveated_minus_uniform",
):
    sub = cost_df[(cost_df["metric"] == metric) & (cost_df["n_objects"] == n_objects)].copy()
    if sub.empty:
        return

    models_present = [m for m in MODEL_ORDER if m in set(sub["model"])]
    transforms_present = [t for t in TRANSFORM_ORDER if t in set(sub["transform"])]

    pivot = sub.pivot_table(index="model_label", columns="transform_label", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=[MODEL_LABELS.get(m, m) for m in models_present])
    pivot = pivot.reindex(columns=[TRANSFORM_LABELS.get(t, t) for t in transforms_present])

    arr = pivot.to_numpy(dtype=float)
    if is_percent_metric(metric):
        arr_plot = 100.0 * arr
    else:
        arr_plot = arr

    if arr_plot.size == 0 or np.all(~np.isfinite(arr_plot)):
        return

    vmax = float(np.nanmax(np.abs(arr_plot)))
    vmin = -vmax

    plt.figure(figsize=(max(7.0, 1.2 * len(transforms_present) + 3), max(3.2, 0.45 * len(models_present) + 2)))
    im = plt.imshow(arr_plot, aspect="auto", vmin=vmin, vmax=vmax)

    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"Foveated - Uniform — {get_metric_label(metric, delta=True)} — {n_objects} objects")
    cbar = plt.colorbar(im)
    cbar.set_label(get_metric_label(metric, delta=True))

    for i in range(arr_plot.shape[0]):
        for j in range(arr_plot.shape[1]):
            val = arr_plot[i, j]
            if not np.isfinite(val):
                continue
            txt = f"{val:.1f}" if is_percent_metric(metric) else f"{val:.2f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# 6 VS 8 COMPARISON
# ============================================================

def plot_compare_6_vs_8(
    runs,
    model: str,
    transform: str,
    vision: str,
    metric: str,
    out_path: Path,
):
    key6 = (model, 6, transform, vision)
    key8 = (model, 8, transform, vision)
    if key6 not in runs or key8 not in runs:
        return

    df6 = runs[key6]
    df8 = runs[key8]
    if metric not in df6.columns or metric not in df8.columns:
        return

    plt.figure(figsize=(7.8, 4.6))

    color = COLORS_MODEL.get(model, None)
    for nobj, df, marker, linestyle in [(6, df6, "o", "-"), (8, df8, "s", "--")]:
        x = df["condition_value"].to_numpy(dtype=float)
        y = metric_values_for_plot(metric, df[metric].to_numpy(dtype=float))
        plt.plot(x, y, marker=marker, linestyle=linestyle, linewidth=2, color=color, label=f"{nobj} objects")

    plt.title(f"{MODEL_LABELS.get(model, model)} — {TRANSFORM_LABELS.get(transform, transform)} — {VISION_LABELS.get(vision, vision)}")
    plt.xlabel(X_LABELS.get(transform, "Condition value"))
    plt.ylabel(get_metric_label(metric))
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    ylim = compute_ylim_for_metric({key6: df6, key8: df8}, metric)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    results_root = Path(args.results_root)
    out_root = Path(args.out_dir)

    detailed_dir = out_root / "detailed"
    uvf_dir = out_root / "uniform_vs_foveated"
    summary_dir = out_root / "summary"
    heatmap_dir = out_root / "heatmaps"
    compare_dir = out_root / "compare_6_vs_8"
    tables_dir = out_root / "tables"

    for d in [detailed_dir, uvf_dir, summary_dir, heatmap_dir, compare_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_root, max_depth=args.recursive_depth)
    if not runs:
        raise RuntimeError(f"No grouped_summary.csv runs found below {results_root}")

    metrics = available_metrics(runs)
    if not metrics:
        raise RuntimeError("No supported metric columns found in loaded grouped_summary.csv files")

    inventory_rows = []
    for key, df in sorted(runs.items()):
        model, nobj, transform, vision = key
        inventory_rows.append({
            "model": model,
            "n_objects": nobj,
            "transform": transform,
            "vision": vision,
            "n_rows": len(df),
            "csv_path": df["csv_path"].iloc[0],
        })
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(tables_dir / "loaded_runs_inventory.csv", index=False)

    print("Loaded runs:")
    print(inventory_df.to_string(index=False))
    print(f"\nAvailable metrics: {metrics}")

    models = existing_models(runs)
    n_objects_list = existing_n_objects(runs)
    transforms = existing_transforms(runs)
    visions = existing_visions(runs)

    # Detailed model-comparison curves for each available vision condition.
    for vision in visions:
        for nobj in n_objects_list:
            for transform in transforms:
                for metric in metrics:
                    out_path = detailed_dir / f"{vision}_{nobj}obj_{transform}_{metric}.png"
                    plot_transform_metric_by_model(
                        runs=runs,
                        n_objects=nobj,
                        transform=transform,
                        vision=vision,
                        metric=metric,
                        out_path=out_path,
                    )

    # Uniform-vs-foveated comparisons, only where both exist.
    for model in models:
        model_dir = uvf_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for nobj in n_objects_list:
            for transform in transforms:
                pair_exists = (model, nobj, transform, "uniform") in runs and (model, nobj, transform, "foveated") in runs
                if not pair_exists:
                    continue

                transform_dir = model_dir / f"{nobj}obj_{transform}"
                transform_dir.mkdir(parents=True, exist_ok=True)

                for metric in metrics:
                    plot_uniform_vs_foveated(
                        runs=runs,
                        model=model,
                        n_objects=nobj,
                        transform=transform,
                        metric=metric,
                        out_path=transform_dir / f"{metric}_uniform_vs_foveated.png",
                    )
                    plot_foveation_delta(
                        runs=runs,
                        model=model,
                        n_objects=nobj,
                        transform=transform,
                        metric=metric,
                        out_path=transform_dir / f"{metric}_delta_foveated_minus_uniform.png",
                    )

    # Summary tables.
    summary_df = summarize_runs(runs, metrics)
    summary_df.to_csv(tables_dir / "summary_all_runs.csv", index=False)

    cost_df = summarize_foveation_costs(runs, metrics)
    if not cost_df.empty:
        cost_df.to_csv(tables_dir / "foveation_costs.csv", index=False)

    # Heatmaps for raw average/worst values by vision.
    for vision in visions:
        for nobj in n_objects_list:
            for metric in MAIN_METRICS:
                if metric not in metrics:
                    continue
                plot_summary_heatmap(
                    summary_df=summary_df,
                    metric=metric,
                    n_objects=nobj,
                    vision=vision,
                    value_type="avg",
                    out_path=heatmap_dir / f"{vision}_{nobj}obj_{metric}_avg_heatmap.png",
                )
                plot_summary_heatmap(
                    summary_df=summary_df,
                    metric=metric,
                    n_objects=nobj,
                    vision=vision,
                    value_type="worst",
                    out_path=heatmap_dir / f"{vision}_{nobj}obj_{metric}_worst_heatmap.png",
                )

    # Heatmaps for foveation cost.
    if not cost_df.empty:
        for nobj in n_objects_list:
            for metric in MAIN_METRICS:
                if metric not in metrics:
                    continue
                plot_foveation_cost_heatmap(
                    cost_df=cost_df,
                    metric=metric,
                    n_objects=nobj,
                    out_path=heatmap_dir / f"{nobj}obj_{metric}_foveation_cost_heatmap.png",
                    value_col="mean_delta_foveated_minus_uniform",
                )

    # 6 vs 8 comparison where available.
    for model in models:
        for vision in visions:
            model_vision_dir = compare_dir / model / vision
            model_vision_dir.mkdir(parents=True, exist_ok=True)
            for transform in transforms:
                transform_dir = model_vision_dir / transform
                transform_dir.mkdir(parents=True, exist_ok=True)
                for metric in MAIN_METRICS:
                    if metric not in metrics:
                        continue
                    plot_compare_6_vs_8(
                        runs=runs,
                        model=model,
                        transform=transform,
                        vision=vision,
                        metric=metric,
                        out_path=transform_dir / f"{metric}_6_vs_8.png",
                    )

    print(f"\nSaved plots/tables to: {out_root.resolve()}")
    print("Important outputs:")
    print(f"  - Inventory: {tables_dir / 'loaded_runs_inventory.csv'}")
    print(f"  - Summary:   {tables_dir / 'summary_all_runs.csv'}")
    if not cost_df.empty:
        print(f"  - Foveation costs: {tables_dir / 'foveation_costs.csv'}")
    print(f"  - Uniform vs foveated plots: {uvf_dir}")
    print(f"  - Heatmaps: {heatmap_dir}")


if __name__ == "__main__":
    main()
