#!/usr/bin/env python3
"""Run Task 19 LightGBM comparisons and bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover - depends on Pod packages
    lgb = None

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:  # pragma: no cover - depends on Pod packages
    pq = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_snuh_task19_baseline_models import FEATURE_SETS, binary_metrics
from scripts.run_snuh_task19_prediction_ci import metric_value


POD_ROOT = Path("/home/khdp-user/workspace/fermat-data")
DEFAULT_LABEL_DIR = POD_ROOT / "task19" / "outputs" / "patient_phenotype_labels_wide"
DEFAULT_FEATURE_DIR = POD_ROOT / "task19" / "outputs" / "baseline_features"
DEFAULT_EMBEDDING_DIR = POD_ROOT / "task19" / "outputs" / "fermat_embeddings_2018_5y_all"
DEFAULT_OUTPUT_DIR = POD_ROOT / "task19" / "outputs" / "lightgbm_ci_2018_5y"
DEFAULT_INDEX_DATE = "2018-01-01"
DEFAULT_HORIZON = "5y"

COUNT_NUMERIC = FEATURE_SETS["age_sex_counts"]["numeric"]
COUNT_CATEGORICAL = FEATURE_SETS["age_sex_counts"]["categorical"]
MODEL_SETS = [
    "lgbm_age_sex_counts",
    "lgbm_fermat_embedding",
    "lgbm_fermat_embedding_counts",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-date", default=DEFAULT_INDEX_DATE)
    parser.add_argument("--horizon", default=DEFAULT_HORIZON)
    parser.add_argument("--phenotypes", nargs="*", default=None)
    parser.add_argument(
        "--model-sets",
        nargs="+",
        choices=MODEL_SETS,
        default=MODEL_SETS,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def require_lightgbm():
    if lgb is None:
        raise RuntimeError(
            "lightgbm is required for this comparison. On the Pod, first run: "
            "python - <<'PY'\nimport lightgbm; print(lightgbm.__version__)\nPY"
        )


def prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} exists and is not empty; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)


def safe_date(index_date: str):
    return index_date.replace("-", "")


def parquet_columns(path: Path):
    if pq is not None:
        return pq.read_schema(path).names
    return pd.read_parquet(path).columns.tolist()


def resolve_embedding_file(args):
    if args.embedding_file is not None:
        path = args.embedding_file.expanduser().resolve()
    else:
        path = (
            args.embedding_dir.expanduser().resolve()
            / f"fermat_embeddings_{safe_date(args.index_date)}_{args.horizon}_last.parquet"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_inputs(args):
    date = safe_date(args.index_date)
    label_path = args.label_dir / f"patient_phenotype_labels_wide_{date}.parquet"
    feature_path = args.feature_dir / f"baseline_features_{date}.parquet"
    embedding_path = resolve_embedding_file(args)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    label_columns = parquet_columns(label_path)
    phenotypes = sorted(
        {
            column.split("__", 1)[1]
            for column in label_columns
            if column.startswith(f"eligible_{args.horizon}__")
        }
    )
    if args.phenotypes:
        requested = set(args.phenotypes)
        missing = sorted(requested - set(phenotypes))
        if missing:
            raise ValueError(f"Unknown phenotypes requested: {missing}")
        phenotypes = [p for p in phenotypes if p in requested]

    label_cols = ["person_id", "split"]
    for phenotype in phenotypes:
        label_cols.append(f"eligible_{args.horizon}__{phenotype}")
        label_cols.append(f"label_{args.horizon}__{phenotype}")
    labels = pd.read_parquet(label_path, columns=label_cols)

    embedding_columns = parquet_columns(embedding_path)
    emb_cols = [column for column in embedding_columns if column.startswith("emb_")]
    embedding_meta = [
        "person_id",
        "split",
        "has_embedding_sequence",
        "sequence_length_pre_index",
    ]
    embeddings = pd.read_parquet(embedding_path, columns=embedding_meta + emb_cols)
    embeddings = embeddings.loc[embeddings["has_embedding_sequence"].astype(bool)].copy()

    count_cols = ["person_id", "split"] + COUNT_NUMERIC + COUNT_CATEGORICAL
    features = pd.read_parquet(feature_path, columns=count_cols)
    data = labels.merge(embeddings, on=["person_id", "split"], how="inner")
    data = data.merge(features, on=["person_id", "split"], how="left")
    return data, phenotypes, emb_cols, str(label_path), str(feature_path), str(embedding_path)


def feature_columns(model_set: str, emb_cols: list[str]):
    if model_set == "lgbm_age_sex_counts":
        return list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)
    if model_set == "lgbm_fermat_embedding":
        return list(emb_cols), []
    if model_set == "lgbm_fermat_embedding_counts":
        return list(emb_cols) + list(COUNT_NUMERIC), list(COUNT_CATEGORICAL)
    raise ValueError(model_set)


def fit_transformer(train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]):
    transformer = {"numeric": {}, "categorical": {}}
    for column in numeric_cols:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=np.float32)
        median = np.nanmedian(values)
        if not np.isfinite(median):
            median = 0.0
        transformer["numeric"][column] = float(median)
    for column in categorical_cols:
        values = train[column].astype("string").fillna("__MISSING__")
        transformer["categorical"][column] = sorted(values.unique().tolist())
    return transformer


def transform_features(frame: pd.DataFrame, transformer: dict):
    data = {}
    categorical_feature = []
    for column, median in transformer["numeric"].items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        data[column] = np.where(np.isnan(values), median, values).astype(np.float32)
    for column, categories in transformer["categorical"].items():
        values = frame[column].astype("string").fillna("__MISSING__")
        values = values.where(values.isin(categories), "__MISSING__")
        data[column] = pd.Categorical(values, categories=categories)
        categorical_feature.append(column)
    x = pd.DataFrame(data, index=frame.index)
    return x, categorical_feature


def lightgbm_params(args):
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "seed": args.random_seed,
        "feature_fraction_seed": args.random_seed,
        "bagging_seed": args.random_seed,
        "data_random_seed": args.random_seed,
        "num_threads": args.n_jobs,
        "verbosity": -1,
    }


def train_predict_model(task, model_set, emb_cols, args):
    train = task.loc[task["split"].eq("train")]
    val = task.loc[task["split"].eq("val")]
    test = task.loc[task["split"].eq("test")].copy()
    numeric_cols, categorical_cols = feature_columns(model_set, emb_cols)
    transformer = fit_transformer(train, numeric_cols, categorical_cols)
    train_x, categorical_feature = transform_features(train, transformer)
    val_x, _ = transform_features(val, transformer)
    test_x, _ = transform_features(test, transformer)
    train_y = train["label"].astype(int).to_numpy()
    val_y = val["label"].astype(int).to_numpy()

    train_set = lgb.Dataset(
        train_x,
        label=train_y,
        categorical_feature=categorical_feature or "auto",
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    valid_names = None
    if len(np.unique(val_y)) == 2 and args.early_stopping_rounds > 0:
        val_set = lgb.Dataset(
            val_x,
            label=val_y,
            categorical_feature=categorical_feature or "auto",
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [val_set]
        valid_names = ["val"]
        callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=False))

    model = lgb.train(
        lightgbm_params(args),
        train_set,
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    best_iteration = model.best_iteration or args.n_estimators
    scores = model.predict(test_x, num_iteration=best_iteration)
    return test, scores, train_x.shape[1], best_iteration


def bootstrap_delta_ci(predictions: pd.DataFrame, args):
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    metrics = ["auroc", "auprc", "brier", "top5_enrichment"]
    baseline_name = "lgbm_age_sex_counts"
    compare_names = ["lgbm_fermat_embedding", "lgbm_fermat_embedding_counts"]
    for phenotype, sub in predictions.groupby("phenotype", sort=False):
        wide = sub.pivot(index="person_id", columns="model_set", values="y_score")
        labels = (
            sub.drop_duplicates("person_id")
            .set_index("person_id")
            .loc[wide.index, "y_true"]
            .to_numpy(dtype=np.int8)
        )
        if baseline_name not in wide.columns:
            continue
        baseline = wide[baseline_name].to_numpy(dtype=np.float64)
        n = len(labels)
        for model_set in compare_names:
            if model_set not in wide.columns:
                continue
            scores = wide[model_set].to_numpy(dtype=np.float64)
            for metric in metrics:
                observed_baseline = metric_value(labels, baseline, metric)
                observed_model = metric_value(labels, scores, metric)
                boot = np.empty(args.bootstrap_samples, dtype=np.float64)
                for sample in range(args.bootstrap_samples):
                    index = rng.integers(0, n, size=n)
                    boot[sample] = (
                        metric_value(labels[index], scores[index], metric)
                        - metric_value(labels[index], baseline[index], metric)
                    )
                rows.append({
                    "phenotype": phenotype,
                    "model_set": model_set,
                    "baseline_model_set": baseline_name,
                    "metric": metric,
                    "n": int(n),
                    "positives": int(labels.sum()),
                    "baseline": observed_baseline,
                    "model": observed_model,
                    "delta": observed_model - observed_baseline,
                    "ci95_lower": float(np.nanpercentile(boot, 2.5)),
                    "ci95_upper": float(np.nanpercentile(boot, 97.5)),
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_seed": args.bootstrap_seed,
                })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.label_dir = args.label_dir.expanduser().resolve()
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.embedding_dir = args.embedding_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    prepare_output(args.output_dir, args.overwrite)
    require_lightgbm()

    started_all = time.time()
    data, phenotypes, emb_cols, label_path, feature_path, embedding_path = load_inputs(args)
    log(f"rows={len(data):,}")
    log(f"phenotypes={len(phenotypes):,}")
    log(f"embedding_features={len(emb_cols):,}")
    log(f"horizon={args.horizon}")
    log(f"lightgbm_version={lgb.__version__}")

    metric_rows = []
    prediction_frames = []
    for phenotype in phenotypes:
        eligible_col = f"eligible_{args.horizon}__{phenotype}"
        label_col = f"label_{args.horizon}__{phenotype}"
        task = data.loc[data[eligible_col].astype(bool)].copy()
        task["label"] = task[label_col].astype(int)
        if task["label"].nunique() < 2:
            log(f"[SKIP] {phenotype}: only one class")
            continue
        train = task.loc[task["split"].eq("train")]
        if train["label"].nunique() < 2:
            log(f"[SKIP] {phenotype}: train one class")
            continue
        split_counts = task.groupby("split")["label"].agg(["size", "sum"]).to_dict()
        for model_set in args.model_sets:
            started = time.time()
            test, scores, n_features, best_iteration = train_predict_model(
                task,
                model_set,
                emb_cols,
                args,
            )
            y_true = test["label"].astype(int).to_numpy()
            row = {
                "index_date": args.index_date,
                "horizon": args.horizon,
                "phenotype": phenotype,
                "model_set": model_set,
                "fit_seconds": time.time() - started,
                "train_rows": int(split_counts["size"].get("train", 0)),
                "train_positives": int(split_counts["sum"].get("train", 0)),
                "val_rows": int(split_counts["size"].get("val", 0)),
                "val_positives": int(split_counts["sum"].get("val", 0)),
                "test_rows": int(split_counts["size"].get("test", 0)),
                "test_positives": int(split_counts["sum"].get("test", 0)),
                "n_model_features": int(n_features),
                "best_iteration": int(best_iteration or args.n_estimators),
            }
            row.update(binary_metrics(y_true, scores, prefix="test_"))
            metric_rows.append(row)
            prediction_frames.append(pd.DataFrame({
                "index_date": args.index_date,
                "horizon": args.horizon,
                "phenotype": phenotype,
                "model_set": model_set,
                "person_id": test["person_id"].to_numpy(),
                "split": "test",
                "y_true": y_true,
                "y_score": scores.astype(np.float32),
                "sequence_length_pre_index": test["sequence_length_pre_index"].to_numpy(),
            }))
            log(
                "[DONE] "
                f"{phenotype} {model_set} "
                f"test_auroc={row['test_auroc']:.4f} "
                f"test_auprc={row['test_auprc']:.4f} "
                f"best_iter={row['best_iteration']} "
                f"{row['fit_seconds']:.1f}s"
            )

    if not prediction_frames:
        raise RuntimeError("No predictions were generated")
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    ci = bootstrap_delta_ci(predictions, args)

    metrics_path = args.output_dir / "lightgbm_metrics.csv"
    predictions_path = args.output_dir / "lightgbm_test_predictions.parquet"
    ci_path = args.output_dir / "lightgbm_bootstrap_delta_ci.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    ci.to_csv(ci_path, index=False)

    manifest = {
        "label_path": label_path,
        "feature_path": feature_path,
        "embedding_path": embedding_path,
        "output_dir": str(args.output_dir),
        "index_date": args.index_date,
        "horizon": args.horizon,
        "phenotypes": phenotypes,
        "model_sets": args.model_sets,
        "baseline_model_set": "lgbm_age_sex_counts",
        "lightgbm_version": lgb.__version__,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "lightgbm_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "subsample_freq": args.subsample_freq,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
            "n_jobs": args.n_jobs,
            "random_seed": args.random_seed,
        },
        "elapsed_seconds": time.time() - started_all,
        "outputs": {
            "metrics": str(metrics_path),
            "test_predictions": str(predictions_path),
            "bootstrap_delta_ci": str(ci_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
