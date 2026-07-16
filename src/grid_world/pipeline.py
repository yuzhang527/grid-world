from __future__ import annotations
from grid_world.activations.extract import extract_activations
from grid_world.config import load_yaml
from grid_world.env.maps import generate_maps
from grid_world.evaluation.validate import validate_run
from grid_world.generation.launcher import generate_trajectories
from grid_world.probes.report import build_report
from grid_world.probes.train import train_probes
from grid_world.targets.build import build_targets

def run_pipeline(config_path, stages):
    cfg=load_yaml(config_path)
    selected=[x.strip() for x in stages.split(",") if x.strip()]
    mcfg,gcfg,acfg,pcfg=cfg.get("maps",{}),cfg.get("generation",{}),cfg.get("activations",{}),cfg.get("probes",{})
    run=gcfg.get("run")
    for stage in selected:
        print(f"[pipeline] stage={stage}",flush=True)
        if stage=="maps": generate_maps(mcfg["config"],mcfg["output"])
        elif stage=="generate":
            generate_trajectories(config_path=gcfg["config"],maps_path=gcfg["maps"],
                                  run_dir=gcfg["run"],gpus=str(gcfg.get("gpus","0")),
                                  parallel_mode=str(gcfg.get("parallel_mode","data")),
                                  resume=bool(gcfg.get("resume",True)))
        elif stage=="validate": validate_run(run)
        elif stage=="targets": build_targets(run)
        elif stage=="activations":
            extract_activations(run_dir=run,model_name=acfg["model"],
                                layers=str(acfg.get("layers","auto")),
                                positions=str(acfg.get("positions","default")),
                                device=str(acfg.get("device","cuda:0")),
                                dtype=str(acfg.get("dtype","auto")),
                                batch_size=int(acfg.get("batch_size",1)))
        elif stage=="probes":
            train_probes(run_dir=run,groups=str(pcfg.get("groups","local,cells,planning")),
                         positions=str(pcfg.get("positions","auto")),
                         layers=str(pcfg.get("layers","all")),
                         backend=str(pcfg.get("backend","torch")),
                         device=str(pcfg.get("device","cuda:0")),
                         splits=int(pcfg.get("splits",20)),
                         epochs=int(pcfg.get("epochs",60)))
        elif stage=="report": build_report(run)
        else: raise ValueError(f"Unknown stage: {stage}")
