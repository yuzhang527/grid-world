# Full-layer and no-grid experiments

## Experiment 2: all layers on existing explicit-belief trajectories

This experiment does not regenerate trajectories. It creates an analysis view
that links to the existing 1000-episode run and writes new all-layer
activations and probes to a separate directory.

Primary positions:

- `prompt_last`: before any response token; cleanest cross-condition state test.
- `pre_action_token`: immediately before the generated action value.
- `mean_current_belief_grid`: while reading the explicit map in the prompt.

Run:

```bash
SOURCE_RUN=runs/qwen25_7b_diverse5x5_1000 \
VIEW_RUN=runs/qwen25_7b_diverse5x5_1000_explicit_all_layers \
DEVICE=cuda:0 \
bash scripts/run_explicit_all_layers.sh
```

Outputs:

```text
runs/qwen25_7b_diverse5x5_1000_explicit_all_layers/
├── activations/
├── probes/
└── layer_curves/
```

The activation artifact contains hidden-state indices 0 through 28. Index 0 is
the embedding output and index 28 is the final normalized representation.

## No-grid condition

The no-grid prompt:

- includes size, start, goal, current position, latest feedback, legal actions,
  and the full observation/action history;
- does not include a current belief grid;
- does not include required map updates;
- requires an action-only JSON response;
- discards any extra map or reasoning fields during parsing.

It uses exactly the same 1000 map specifications as the explicit condition.

Run:

```bash
RUN=runs/qwen25_7b_no_grid_diverse5x5_1000 \
GPUS=0,1,2,3 \
DEVICE=cuda:0 \
bash scripts/run_no_grid_full.sh
```

## Extended map targets

The target builder now creates:

- `cells`: the correct observable map, using F/O/U;
- `memory`: whether each location has been observed;
- `explicit_cells`: the map written in the model's current explicit response;
- `true_cells`: the complete true map, using F/O;
- `true_cells_observed`: true map labels only for observed locations;
- `true_cells_unobserved`: true map labels only for unobserved locations.

`explicit_cells` is unavailable in the no-grid condition by design.

## Important comparison detail

`prompt_last` is the primary fair representation comparison. Both conditions
are measured after reading their prompts but before generating a response.

`pre_action_token` remains useful for action selection. In the explicit
condition, however, the response may already contain the generated belief grid
before the action field. Therefore pre-action map decodability can partly
reflect the model's own generated map text.
