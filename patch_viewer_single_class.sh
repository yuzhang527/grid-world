#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-$PWD}"
VIEWER="$REPO/scripts/trajectory_probe_viewer_v3.py"

if [[ ! -f "$VIEWER" ]]; then
  echo "[patch] missing viewer: $VIEWER" >&2
  exit 1
fi

BACKUP="${VIEWER}.bak_single_class_$(date +%Y%m%d_%H%M%S)"
cp "$VIEWER" "$BACKUP"
echo "[patch] backup=$BACKUP"

VIEWER="$VIEWER" python - <<'PY'
from __future__ import annotations

import os
import re
from pathlib import Path

path = Path(os.environ["VIEWER"])
src = path.read_text(encoding="utf-8")

wrapper_marker = "class _SingleClassSafeLogisticRegression:"

wrapper = r'''
class _SingleClassSafeLogisticRegression:
    # LogisticRegression-compatible wrapper with a constant-class fallback.
    # The fallback keeps the viewer operational without pretending that a
    # learned decision boundary exists.

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._model = None
        self._constant = None
        self.is_constant_ = False

    def fit(self, X, y, sample_weight=None):
        import numpy as _np

        classes = _np.unique(_np.asarray(y, dtype=object))
        if classes.size == 0:
            raise ValueError("Cannot fit decoder with zero training labels")

        self.classes_ = classes
        self.n_features_in_ = int(X.shape[1])

        if classes.size == 1:
            self._constant = classes[0]
            self.is_constant_ = True
            return self

        self._model = _SklearnLogisticRegression(*self._args, **self._kwargs)
        if sample_weight is None:
            self._model.fit(X, y)
        else:
            self._model.fit(X, y, sample_weight=sample_weight)

        self.classes_ = self._model.classes_
        self.n_features_in_ = self._model.n_features_in_
        return self

    def predict(self, X):
        import numpy as _np

        if self.is_constant_:
            return _np.full(X.shape[0], self._constant, dtype=object)
        return self._model.predict(X)

    def predict_proba(self, X):
        import numpy as _np

        if self.is_constant_:
            return _np.ones((X.shape[0], 1), dtype=float)
        return self._model.predict_proba(X)

    def decision_function(self, X):
        import numpy as _np

        if self.is_constant_:
            return _np.zeros(X.shape[0], dtype=float)
        return self._model.decision_function(X)

    def __getattr__(self, name):
        model = self.__dict__.get("_model")
        if model is not None:
            return getattr(model, name)
        raise AttributeError(name)


def LogisticRegression(*args, **kwargs):
    return _SingleClassSafeLogisticRegression(*args, **kwargs)
'''

if wrapper_marker not in src:
    import_patterns = [
        r"^from sklearn\.linear_model import LogisticRegression\s*$",
        r"^from sklearn\.linear_model import LogisticRegression as LogisticRegression\s*$",
    ]
    replaced = False
    for pattern in import_patterns:
        match = re.search(pattern, src, flags=re.MULTILINE)
        if match:
            replacement = (
                "from sklearn.linear_model import "
                "LogisticRegression as _SklearnLogisticRegression\n\n"
                + wrapper.strip("\n")
            )
            src = src[:match.start()] + replacement + src[match.end():]
            replaced = True
            break
    if not replaced:
        raise SystemExit(
            "[patch] could not locate the LogisticRegression import. "
            "Run: grep -n \"sklearn.linear_model\" scripts/trajectory_probe_viewer_v3.py"
        )
else:
    print("[patch] safe LogisticRegression wrapper already installed")

raise_pattern = re.compile(
    r'(?P<indent>^[ \t]*)if len\(classes\) < 2:\s*\n'
    r'(?P=indent)[ \t]+raise RuntimeError\('
    r'f(?P<quote>["\'])Task \{task\} has fewer than two train classes: '
    r'\{classes\}(?P=quote)\)',
    flags=re.MULTILINE,
)

match = raise_pattern.search(src)
if match:
    indent = match.group("indent")
    replacement = (
        f"{indent}if len(classes) < 2:\n"
        f"{indent}    print(\n"
        f"{indent}        f\"[viewer/v3] task={{task}} has one train class "
        f"{{classes}}; using a constant OOF decoder \"\n"
        f"{indent}        \"(structural baseline, not a learned probe).\"\n"
        f"{indent}    )"
    )
    src = src[:match.start()] + replacement + src[match.end():]
elif "has fewer than two train classes" in src:
    raise SystemExit(
        "[patch] found the single-class error text but its surrounding code "
        "did not match the expected form; refusing an ambiguous edit"
    )
else:
    print("[patch] single-class raise already removed or changed")

path.write_text(src, encoding="utf-8")
print(f"[patch] updated={path}")
PY

python -m py_compile "$VIEWER"
echo "[patch] syntax_check=PASS"

grep -n \
  -e "_SingleClassSafeLogisticRegression" \
  -e "constant OOF decoder" \
  "$VIEWER" \
  | head -20

echo
echo "[patch] Remove old viewer cache before rerunning:"
echo "  rm -rf \"\$RUN/trajectory_viewer_cache_v3\""
