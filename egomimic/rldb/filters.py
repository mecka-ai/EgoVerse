from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from egomimic.utils.scale_utils import build_df_from_tasks, get_completed_tasks


class DatasetFilter:
    def __init__(self, filter_lambdas: Sequence[str] | None = None) -> None:
        self.filter_lambdas = list(filter_lambdas or [])
        self.filters = []
        for expr in self.filter_lambdas:
            try:
                predicate = eval(expr)
            except Exception as exc:
                print(f"Invalid filter: {expr}", file=sys.stderr)
                raise ValueError(f"Invalid filter: {expr}") from exc
            if not callable(predicate):
                print(f"Invalid filter: {expr}", file=sys.stderr)
                raise ValueError(f"Invalid filter: {expr}")
            self.filters.append(predicate)

    def __repr__(self) -> str:
        return f"DatasetFilter(filter_lambdas={self.filter_lambdas!r})"

    def is_empty(self) -> bool:
        return not self.filter_lambdas

    def filter_df(self, df: Any) -> Any:
        """Apply any DataFrame-level lambdas (those that accept and return a DataFrame)."""
        for predicate in self.filters:
            try:
                result = predicate(df)
                if hasattr(result, "iterrows"):  # it's a DataFrame
                    df = result
            except Exception:
                pass  # per-row predicate — skip here, applied in matches()
        return df

    def matches(self, row: Mapping[str, Any]) -> bool:
        row = dict(row)
        if row.get("is_deleted", False):
            return False
        for expr, predicate in zip(self.filter_lambdas, self.filters, strict=True):
            try:
                result = predicate(row)
            except (TypeError, KeyError, AttributeError):
                # DataFrame-level predicate — already applied in filter_df(), skip
                continue
            if not isinstance(result, bool):
                raise TypeError(f"Filter must return bool: {expr}")
            if not result:
                return False
        return True


class ScaleAnnotationDatasetFilter(DatasetFilter):
    def __init__(
        self, project_name: str, filter_lambdas: Sequence[str] | None = None
    ) -> None:
        self.project_name = project_name
        self.api_key = os.environ["SCALE_API_KEY"]
        self.tasks = get_completed_tasks(self.project_name, self.api_key)
        self.df = build_df_from_tasks(self.tasks)
        self.completed_episode_hashes = set(self.df["SEQUENCE_ID"].unique().tolist())
        super().__init__(filter_lambdas)

    def is_empty(self) -> bool:
        return False

    def matches(self, row: Mapping[str, Any]) -> bool:
        if row.get("episode_hash") not in self.completed_episode_hashes:
            return False
        return super().matches(row)
