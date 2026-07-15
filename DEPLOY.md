# Deployment

## Archive deployment

```bash
unzip grid-world.zip
cd grid-world
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[activation,dev]"
```

Install a vLLM build matching the server CUDA environment, then:

```bash
pip install -e ".[generation]"
```

Validate:

```bash
pytest
bash scripts/smoke_test.sh
```

## Cat-friendly deployment

Copy `install_grid_world.sh` to the server and run:

```bash
cat install_grid_world.sh | bash -s -- /workspace/luoyuzhang
cd /workspace/luoyuzhang/grid-world
```
