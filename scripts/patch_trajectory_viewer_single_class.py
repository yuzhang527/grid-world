#!/usr/bin/env python3
"""
Patch trajectory_probe_viewer_v3.py so single-class training folds use a
constant decoder instead of crashing LogisticRegression.

This handles indented/local imports such as:
        from sklearn.linear_model import LogisticRegression
"""
from __future__ import annotations

import datetime as dt
import py_compile
import re
import shutil
import sys
from pathlib import Path


MARKER = "SINGLE_CLASS_SAFE_LOGREG_V2"


def indent_block(text: str, indent: str) -> str:
    return "\n".join(indent + line if line else "" for line in text.splitlines())


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    target = repo / "scripts" / "trajectory_probe_viewer_v3.py"
    if not target.is_file():
        raise SystemExit(f"Missing viewer source: {target}")

    source = target.read_text(encoding="utf-8")
    original = source

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_single_class_v2_{timestamp}")
    shutil.copy2(target, backup)
    print(f"[patch] backup={backup}")

    if MARKER not in source:
        import_pattern = re.compile(
            r"^(?P<indent>[ \t]*)from sklearn\.linear_model import LogisticRegression[ \t]*$",
            re.MULTILINE,
        )
        match = import_pattern.search(source)
        if not match:
            raise SystemExit(
                "[patch] could not locate any LogisticRegression import.\n"
                "Run: grep -n \"sklearn.linear_model\" "
                "scripts/trajectory_probe_viewer_v3.py"
            )

        indent = match.group("indent")
        wrapper = f"""from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression
import numpy as _single_class_np

# {MARKER}
class LogisticRegression:
    # Drop-in local wrapper supporting a one-class viewer fold.

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._model = None
        self._constant = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        y_array = _single_class_np.asarray(y)
        classes = _single_class_np.unique(y_array)
        if classes.size == 0:
            raise ValueError("Cannot fit a decoder with zero training labels")
        self.classes_ = classes
        if classes.size == 1:
            self._constant = classes[0]
            self._model = None
            return self

        self._constant = None
        self._model = _SklearnLogisticRegression(*self._args, **self._kwargs)
        if sample_weight is None:
            self._model.fit(X, y)
        else:
            self._model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        if self._constant is not None:
            return _single_class_np.full(len(X), self._constant)
        return self._model.predict(X)

    def predict_proba(self, X):
        if self._constant is not None:
            return _single_class_np.ones((len(X), 1), dtype=float)
        return self._model.predict_proba(X)

    def decision_function(self, X):
        if self._constant is not None:
            return _single_class_np.zeros(len(X), dtype=float)
        return self._model.decision_function(X)

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        correct = prediction == _single_class_np.asarray(y)
        if sample_weight is None:
            return float(correct.mean())
        weights = _single_class_np.asarray(sample_weight, dtype=float)
        return float((correct * weights).sum() / weights.sum())
"""
        source = source[: match.start()] + indent_block(wrapper, indent) + source[match.end() :]
        import_line = original[:match.start()].count("\n") + 1
        print(f"[patch] wrapped LogisticRegression import at original line {import_line}")
    else:
        print("[patch] safe LogisticRegression wrapper already present")

    raise_pattern = re.compile(
        r'^(?P<indent>[ \t]*)raise RuntimeError\('
        r'f"Task \{task\} has fewer than two train classes: \{classes\}"'
        r'\)[ \t]*$',
        re.MULTILINE,
    )

    replacements = 0

    def replace_raise(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        indent = match.group("indent")
        return (
            indent
            + 'print(f"[viewer/v3] task={task} has one train class {classes}; "'
            + '"using constant decoder (structural baseline, not a learned probe).")'
        )

    source = raise_pattern.sub(replace_raise, source)

    if replacements == 0:
        if "has fewer than two train classes" in source:
            raise SystemExit(
                "[patch] found the single-class error text but could not safely "
                "replace its raise statement. Inspect with:\n"
                "sed -n '260,310p' scripts/trajectory_probe_viewer_v3.py"
            )
        print("[patch] single-class raise already absent")
    else:
        print(f"[patch] replaced_single_class_raises={replacements}")

    target.write_text(source, encoding="utf-8")
    py_compile.compile(str(target), doraise=True)

    final = target.read_text(encoding="utf-8")
    if MARKER not in final:
        raise SystemExit("[patch] verification failed: wrapper marker missing")
    if re.search(
        r'raise RuntimeError\(f"Task \{task\} has fewer than two train classes',
        final,
    ):
        raise SystemExit("[patch] verification failed: crashing raise remains")

    print(f"[patch] installed={target}")
    print("[patch] py_compile=PASS")


if __name__ == "__main__":
    main()

