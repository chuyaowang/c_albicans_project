"""Summarize cross-validation TensorBoard logs into a per-fold metrics table.

Layout expected (matches `n_fold_validation` in `gnn_train.py`):

    <root>/<repeat>/fold_<k>/events.out.tfevents...

For each fold, reads `EarlyStopping/Best_Epoch` from the run, then samples the
epoch-indexed scalars at that epoch:
  - AUC/Test, F1/Test, PR_AUC/Test
  - Diag/Pred_Mean_Test, Diag/Pred_Std_Test

Usage:
    python summarize_cv_logs.py <log_root>
or import `summarize_cv_logs(log_root)` and use the returned DataFrame.
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

EPOCH_TAGS = [
    ("AUC/Test", "auc"),
    ("F1/Test", "f1"),
    ("PR_AUC/Test", "pr_auc"),
    ("Diag/Pred_Mean", "pred_mean_train"),
    ("Diag/Pred_Std", "pred_std_train"),
    ("Diag/Pred_Mean_Test", "pred_mean_test"),
    ("Diag/Pred_Std_Test", "pred_std_test"),
    # Node-type head. Absent from edge-only runs, and the per-class entries are absent from
    # any fold whose held-out image lacks that class -- `_value_at_step` returns None for a
    # missing tag, so those become empty cells rather than a crash or a fabricated zero.
    # Each F1 is paired with its support: an F1 read without its denominator is exactly the
    # misleading number this table exists to prevent (background F1 rests on 4-21 nodes).
    ("NodeType/Accuracy_Test", "node_accuracy"),
    ("NodeType/F1_background_Test", "node_f1_background"),
    ("NodeType/F1_epithelial_Test", "node_f1_epithelial"),
    ("NodeType/F1_hyphal_Test", "node_f1_hyphal"),
    ("NodeType/Support_background_Test", "node_support_background"),
    ("NodeType/Support_epithelial_Test", "node_support_epithelial"),
    ("NodeType/Support_hyphal_Test", "node_support_hyphal"),
]


def _load_run(run_dir: Path) -> EventAccumulator:
    acc = EventAccumulator(str(run_dir), size_guidance={'scalars': 0})
    acc.Reload()
    return acc


def _value_at_step(acc: EventAccumulator, tag: str, step: int) -> Optional[float]:
    if tag not in acc.Tags().get('scalars', []):
        return None
    events = acc.Scalars(tag)
    for ev in events:
        if ev.step == step:
            return float(ev.value)
    return None


_THRESH_RE = re.compile(r'Thresh[: ]\s*([0-9.eE+-]+)')
_SUMMARY_TAGS = ('Fold Summary', 'Overfit Test Summary')


def _read_threshold(run_dir: Path) -> Optional[float]:
    for event_file in sorted(run_dir.glob('events.out.tfevents*')):
        loader = EventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.HasField('summary'):
                continue
            for value in event.summary.value:
                if not any(tag in value.tag for tag in _SUMMARY_TAGS):
                    continue
                if not value.HasField('tensor'):
                    continue
                text = b''.join(value.tensor.string_val).decode('utf-8', errors='ignore')
                m = _THRESH_RE.search(text)
                if m is not None:
                    return float(m.group(1))
    return None


def _summarize_fold(run_dir: Path) -> Optional[dict]:
    acc = _load_run(run_dir)
    scalar_tags = acc.Tags().get('scalars', [])

    if 'EarlyStopping/Best_Epoch' not in scalar_tags:
        return None

    best_epoch = int(acc.Scalars('EarlyStopping/Best_Epoch')[0].value)

    row = {'best_epoch': best_epoch}
    for tag, col in EPOCH_TAGS:
        row[col] = _value_at_step(acc, tag, best_epoch)
    row['threshold'] = _read_threshold(run_dir)
    return row


def summarize_cv_logs(log_root: Path) -> pd.DataFrame:
    log_root = Path(log_root)
    rows = []
    fold_re = re.compile(r'fold_(\d+)$')

    for repeat_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for fold_dir in sorted(p for p in repeat_dir.iterdir() if p.is_dir()):
            m = fold_re.match(fold_dir.name)
            if m is None:
                continue
            row = _summarize_fold(fold_dir)
            if row is None:
                continue
            row['repeat'] = repeat_dir.name
            row['fold'] = int(m.group(1))
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Column order, DERIVED from EPOCH_TAGS rather than restated. This list was hardcoded
    # and silently dropped anything missing from it, so a tag added above never reached the
    # CSV -- which is exactly how the node-type metrics came to be logged to TensorBoard and
    # absent from the table for the whole node-type experiment. Deriving it means the two
    # cannot drift again.
    cols = (['repeat', 'fold', 'best_epoch']
            + [col for _, col in EPOCH_TAGS]
            + ['threshold'])
    return df[[c for c in cols if c in df.columns]].sort_values(['repeat', 'fold']).reset_index(drop=True)


def pivot_by_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the per-(repeat, fold) table so each row is one fold and every
    repeat's metrics live side-by-side as columns.

    Columns are flattened to ``<repeat>_<metric>`` (e.g. ``repeat_0_auc``,
    ``repeat_1_auc``, ``repeat_2_auc``, ``repeat_0_pred_std_test``, ...).
    Repeats and metrics are kept in the order they appear in the input frame,
    so the same metric for repeats 0/1/2 stays grouped together.
    """
    if df.empty:
        return df
    if not {'repeat', 'fold'}.issubset(df.columns):
        raise ValueError("pivot_by_fold expects 'repeat' and 'fold' columns")

    metric_cols = [c for c in df.columns if c not in ('repeat', 'fold')]
    repeats = list(dict.fromkeys(df['repeat']))

    pivoted = df.pivot(index='fold', columns='repeat', values=metric_cols)
    # Reorder columns to (metric, repeat) so each metric's repeats sit together
    pivoted = pivoted.reindex(columns=pd.MultiIndex.from_product([metric_cols, repeats]))
    pivoted.columns = [f"{repeat}_{metric}" for metric, repeat in pivoted.columns]
    return pivoted.reset_index()


def wrangle_csv(input_csv: Path, output_csv: Optional[Path] = None) -> pd.DataFrame:
    """Load a long-form CSV (as produced by `summarize_cv_logs --csv`) and
    return the pivoted, one-row-per-fold table. Optionally write it to disk.
    """
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)
    wide = pivot_by_fold(df)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(output_csv, index=False)
    return wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_root', type=Path, nargs='?', default=None,
                        help='Root experiment dir containing <repeat>/fold_<k> subdirs.')
    parser.add_argument('--from-csv', type=Path, default=None,
                        help='Skip event scraping; load this long-form CSV instead.')
    parser.add_argument('--csv', type=Path, default=None,
                        help='Optional path to write the (possibly pivoted) table as CSV.')
    parser.add_argument('--by-fold', action='store_true',
                        help='Pivot to one row per fold (repeats side-by-side). '
                             'Forced True when --from-csv is given.')
    args = parser.parse_args()

    if args.from_csv is not None:
        df = pd.read_csv(args.from_csv)
        out_df = pivot_by_fold(df)
    else:
        if args.log_root is None:
            parser.error('Either log_root or --from-csv must be given.')
        df = summarize_cv_logs(args.log_root)
        if df.empty:
            print(f"No folds with EarlyStopping/Best_Epoch found under {args.log_root}")
            return
        out_df = pivot_by_fold(df) if args.by_fold else df

    with pd.option_context('display.float_format', '{:.4f}'.format,
                           'display.max_columns', None,
                           'display.width', 200):
        print(out_df.to_string(index=False))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == '__main__':
    main()