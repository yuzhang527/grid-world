from __future__ import annotations
from pathlib import Path
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from grid_world.activations.extract import extract_activations
from grid_world.env.maps import generate_maps,show_map,validate_maps,summarize_maps
from grid_world.evaluation.behavior import summarize_run
from grid_world.evaluation.validate import validate_run
from grid_world.generation.launcher import generate_trajectories
from grid_world.migrate import migrate_legacy_run
from grid_world.pipeline import run_pipeline
from grid_world.probes.report import build_report
from grid_world.probes.train import train_probes
from grid_world.targets.build import build_targets

app=typer.Typer(no_args_is_help=True,help="Grid-world LLM representation pipeline.")
maps_app=typer.Typer(no_args_is_help=True); trajectories_app=typer.Typer(no_args_is_help=True)
targets_app=typer.Typer(no_args_is_help=True); activations_app=typer.Typer(no_args_is_help=True)
probes_app=typer.Typer(no_args_is_help=True); pipeline_app=typer.Typer(no_args_is_help=True)
migrate_app=typer.Typer(no_args_is_help=True)
for name,sub in [("maps",maps_app),("trajectories",trajectories_app),("targets",targets_app),
                 ("activations",activations_app),("probes",probes_app),
                 ("pipeline",pipeline_app),("migrate",migrate_app)]:
    app.add_typer(sub,name=name)

@maps_app.command("generate")
def maps_generate(config:Path=typer.Option(...,exists=True),output:Path=typer.Option(...)):
    rows=generate_maps(config,output); print(f"[green]Generated {len(rows)} maps:[/green] {output}")

@maps_app.command("validate")
def maps_validate(maps:Path=typer.Option(...,exists=True)): print(validate_maps(maps))

@maps_app.command("summarize")
def maps_summarize(maps: Path = typer.Option(..., exists=True)):
    print(summarize_maps(maps))


@maps_app.command("show")
def maps_show(maps:Path=typer.Option(...,exists=True),
              episode:str|None=typer.Option(None),index:int=typer.Option(0)):
    print(show_map(maps,episode,index))

@trajectories_app.command("generate")
def trajectories_generate(config:Path=typer.Option(...,exists=True),
                          maps:Path=typer.Option(...,exists=True),run:Path=typer.Option(...),
                          gpus:str=typer.Option("0"),parallel_mode:str=typer.Option("data"),
                          resume:bool=typer.Option(True,"--resume/--no-resume")):
    print(generate_trajectories(config_path=config,maps_path=maps,run_dir=run,
                                gpus=gpus,parallel_mode=parallel_mode,resume=resume))

@trajectories_app.command("validate")
def trajectories_validate(run:Path=typer.Option(...,exists=True)): print(validate_run(run))

@trajectories_app.command("summarize")
def trajectories_summarize(run:Path=typer.Option(...,exists=True)):
    summary=summarize_run(run); table=Table(title="Trajectory summary")
    table.add_column("Metric"); table.add_column("Value")
    for key,value in summary.items(): table.add_row(str(key),str(value))
    Console().print(table)

@targets_app.command("build")
def targets_build(run:Path=typer.Option(...,exists=True)):
    rows=build_targets(run); print(f"[green]Built {len(rows)} target rows.[/green]")

@activations_app.command("extract")
def activations_extract(run:Path=typer.Option(...,exists=True),model:str=typer.Option(...),
                        layers:str=typer.Option("auto"),positions:str=typer.Option("default"),
                        device:str=typer.Option("cuda:0"),dtype:str=typer.Option("auto"),
                        batch_size:int=typer.Option(1,min=1),
                        include_repaired:bool=typer.Option(False),
                        include_parse_errors:bool=typer.Option(False),
                        max_rows:int|None=typer.Option(None),
                        trust_remote_code:bool=typer.Option(False)):
    print(extract_activations(run_dir=run,model_name=model,layers=layers,positions=positions,
                              device=device,dtype=dtype,batch_size=batch_size,
                              include_repaired=include_repaired,
                              include_parse_errors=include_parse_errors,max_rows=max_rows,
                              trust_remote_code=trust_remote_code))

@probes_app.command("train")
def probes_train(run:Path=typer.Option(...,exists=True),
                 groups:str=typer.Option("local,cells,planning"),
                 positions:str=typer.Option("auto"),layers:str=typer.Option("all"),
                 backend:str=typer.Option("torch"),device:str=typer.Option("cuda:0"),
                 splits:int=typer.Option(20),test_size:float=typer.Option(0.2),
                 seed:int=typer.Option(123),min_class_count:int=typer.Option(5),
                 epochs:int=typer.Option(60),learning_rate:float=typer.Option(0.03),
                 weight_decay:float=typer.Option(1e-4)):
    frame=train_probes(run_dir=run,groups=groups,positions=positions,layers=layers,
                       backend=backend,device=device,splits=splits,test_size=test_size,
                       seed=seed,min_class_count=min_class_count,epochs=epochs,
                       learning_rate=learning_rate,weight_decay=weight_decay)
    print(f"[green]Saved {len(frame)} aggregate rows.[/green]")

@probes_app.command("report")
def probes_report(run:Path=typer.Option(...,exists=True)): print(build_report(run))

@pipeline_app.command("run")
def pipeline_run(config:Path=typer.Option(...,exists=True),
                 stages:str=typer.Option("maps,generate,validate,targets,activations,probes,report")):
    run_pipeline(config,stages)

@migrate_app.command("legacy-run")
def migrate_legacy(source:Path=typer.Option(...,exists=True),run:Path=typer.Option(...)):
    print(migrate_legacy_run(source,run))
