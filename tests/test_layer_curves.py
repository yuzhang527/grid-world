from pathlib import Path

import pandas as pd

import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_layer_curves.py"
SPEC = importlib.util.spec_from_file_location("plot_layer_curves", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
aggregate_layers = MODULE.aggregate_layers


def test_layer_aggregation_uses_fixed_tasks():
    frame = pd.DataFrame(
        [
            {
                "task_group": "cells",
                "task": "a",
                "position": "prompt_last",
                "layer": 0,
                "macro_f1_mean": 0.4,
                "macro_f1_std": 0.1,
                "majority_macro_f1_mean": 0.2,
                "num_test_mean": 10,
            },
            {
                "task_group": "cells",
                "task": "b",
                "position": "prompt_last",
                "layer": 0,
                "macro_f1_mean": 0.6,
                "macro_f1_std": 0.1,
                "majority_macro_f1_mean": 0.2,
                "num_test_mean": 10,
            },
            {
                "task_group": "cells",
                "task": "a",
                "position": "prompt_last",
                "layer": 1,
                "macro_f1_mean": 0.5,
                "macro_f1_std": 0.1,
                "majority_macro_f1_mean": 0.2,
                "num_test_mean": 10,
            },
            {
                "task_group": "cells",
                "task": "b",
                "position": "prompt_last",
                "layer": 1,
                "macro_f1_mean": 0.7,
                "macro_f1_std": 0.1,
                "majority_macro_f1_mean": 0.2,
                "num_test_mean": 10,
            },
        ]
    )
    summary = aggregate_layers(frame)
    layer0 = summary[summary["layer"] == 0].iloc[0]
    layer1 = summary[summary["layer"] == 1].iloc[0]
    assert layer0["tasks"] == 2
    assert layer0["mean_macro_f1"] == 0.5
    assert layer1["mean_macro_f1"] == 0.6
