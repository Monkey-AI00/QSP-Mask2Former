"""Lightweight config validation hooks."""

from __future__ import annotations


def validate_args(args) -> None:
    """
    轻量参数校验占位。
    目前保持与 legacy 完全一致，不新增行为约束。
    """
    _ = args


__all__ = ["validate_args"]

