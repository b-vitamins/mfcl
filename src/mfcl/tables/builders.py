# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import pandas as pd


# -------------------------- utilities --------------------------


def _ensure_list(x: Optional[Iterable]) -> List:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _round_cols(df: pd.DataFrame, prec: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    for col, p in prec.items():
        if col in df.columns:
            df[col] = df[col].astype(float).round(p)
    return df


def _bold_best(
    df: pd.DataFrame, cols: Sequence[str], groupby: Sequence[str], mode: str = "min"
) -> pd.DataFrame:
    df = df.copy()
    if not cols:
        return df
    grouped = df.groupby(list(groupby), dropna=False)
    best_idx: List[int] = []
    for _, g in grouped:
        if g.empty:
            continue
        if mode == "min":
            target = g[cols[0]] if len(cols) == 1 else g[cols].sum(axis=1)
            sel = target.idxmin()
        else:
            target = g[cols[0]] if len(cols) == 1 else g[cols].sum(axis=1)
            sel = target.idxmax()
        best_idx.append(int(sel))
    mask = df.index.isin(best_idx)
    for c in cols:
        if c in df.columns:
            df.loc[mask, c] = df.loc[mask, c].map(lambda v: f"\\textbf{{{v}}}")
    return df


def _latex(df: pd.DataFrame, caption: str, label: str, index: bool = False) -> str:
    return df.to_latex(escape=False, index=index, caption=caption, label=label)


def _write(
    df: pd.DataFrame,
    out_csv: Path,
    out_tex: Path,
    caption: str,
    label: str,
    index: bool = False,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=index)
    out_tex.write_text(
        _latex(df, caption=caption, label=label, index=index), encoding="utf-8"
    )


# -------------------------- table builders --------------------------


@dataclass
class TableSpec:
    name: str
    caption: str
    label: str
    filters: Optional[Dict[str, Iterable]] = None
    columns: Optional[Dict[str, str]] = None
    precision: Optional[Dict[str, int]] = None
    groupby: Optional[List[str]] = None
    select_by: str = "mae"
    select_mode: str = "min"
    bold_cols: Optional[List[str]] = None
    sort_by: Optional[List[Tuple[str, bool]]] = None


def table_fidelity_summary(csv_path: str | Path, spec: TableSpec) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(csv_path))
    df0 = cast(pd.DataFrame, df.copy())

    if spec.filters:
        for k, vals in spec.filters.items():
            df0 = cast(pd.DataFrame, df0[df0[k].isin(list(vals))])

    group_keys = spec.groupby
    if not group_keys:
        raise ValueError("TableSpec.groupby must be provided")

    df0["_rank"] = df0.groupby(group_keys)[spec.select_by].rank(
        method="first", ascending=(spec.select_mode == "min")
    )
    best = df0[df0["_rank"] == 1.0].copy()

    columns_map = spec.columns or {}
    if columns_map:
        best = best[list(columns_map.keys())]
        best = cast(pd.DataFrame, best.rename(columns=columns_map))

    if spec.precision:
        best = _round_cols(best, spec.precision)

    if spec.sort_by:
        cols_sort = [c for c, _ in spec.sort_by]
        asc = [a for _, a in spec.sort_by]
        best = best.sort_values(cols_sort, ascending=asc)

    if spec.bold_cols:
        gb_keys = [columns_map.get(k, k) for k in group_keys]
        bold_cols = [columns_map.get(c, c) for c in spec.bold_cols]
        best = _bold_best(best, bold_cols, gb_keys, mode=spec.select_mode)

    return best


def table_gradient_summary(
    csv_path: str | Path,
    *,
    filters: Dict[str, Iterable],
    columns: Dict[str, str],
    precision: Dict[str, int],
    groupby: List[str],
    sort_by: Optional[List[Tuple[str, bool]]] = None,
    bold_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(csv_path))
    df0 = cast(pd.DataFrame, df.copy())
    for k, vals in filters.items():
        df0 = cast(pd.DataFrame, df0[df0[k].isin(list(vals))])

    agg = cast(
        pd.DataFrame,
        df0.groupby(groupby)
        .agg(
            {
                "cos_all": "mean",
                "rel_norm_err_all": "mean",
                "cos_X": "mean",
                "cos_Y": "mean",
                "sign_agree_topk_X": "mean",
                "sign_agree_topk_Y": "mean",
            }
        )
        .reset_index(),
    )

    out = cast(pd.DataFrame, agg[list(columns.keys())].rename(columns=columns))
    out = _round_cols(out, precision or {})

    if sort_by:
        cols_sort = [c for c, _ in sort_by]
        asc = [a for _, a in sort_by]
        out = out.sort_values(cols_sort, ascending=asc)

    if bold_cols:
        out = _bold_best(
            out, bold_cols, groupby=[columns.get(k, k) for k in groupby], mode="max"
        )

    return out


def table_efficiency_summary(
    csv_path: str | Path,
    *,
    filters: Dict[str, Iterable],
    columns: Dict[str, str],
    precision: Dict[str, int],
    groupby: List[str],
    sort_by: Optional[List[Tuple[str, bool]]] = None,
    bold_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = cast(pd.DataFrame, pd.read_csv(csv_path))
    df0 = cast(pd.DataFrame, df.copy())
    for k, vals in filters.items():
        df0 = cast(pd.DataFrame, df0[df0[k].isin(list(vals))])

    agg = cast(
        pd.DataFrame,
        df0.groupby(groupby)
        .agg(
            {
                "ms_median": "mean",
                "peak_mem_gb": "mean",
                "bytes_all_gather_theoretical": "mean",
                "bytes_all_reduce_theoretical": "mean",
            }
        )
        .reset_index(),
    )

    out = cast(pd.DataFrame, agg[list(columns.keys())].rename(columns=columns))
    out = _round_cols(out, precision or {})

    if sort_by:
        cols_sort = [c for c, _ in sort_by]
        asc = [a for _, a in sort_by]
        out = out.sort_values(cols_sort, ascending=asc)

    if bold_cols:
        out = _bold_best(
            out, bold_cols, groupby=[columns.get(k, k) for k in groupby], mode="min"
        )

    return out
