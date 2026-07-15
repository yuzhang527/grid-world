from __future__ import annotations
from pathlib import Path
from grid_world.env.grid import GridSpec
from grid_world.env.planning import shortest_path_length
from grid_world.generation.runner import summarize_episode_rows
from grid_world.utils.io import read_json, read_jsonl, write_json, write_jsonl
from grid_world.utils.manifest import write_manifest

def _map_obstacles(value):
    if not isinstance(value,list) or not value:
        return None
    size=len(value); obstacles=[]
    for row_index,row in enumerate(value):
        if not isinstance(row,list) or len(row)!=size:
            return None
        y=size-1-row_index
        for x,cell in enumerate(row):
            if str(cell).upper() in {"O","#","1","BLOCKED"}:
                obstacles.append([x,y])
    return size,obstacles

def migrate_legacy_run(source, run_dir):
    source,run=Path(source),Path(run_dir); run.mkdir(parents=True,exist_ok=True)
    steps=read_jsonl(source/"steps.jsonl"); normalized=[]
    for row in steps:
        item=dict(row)
        if "current_pos" not in item and "position" in item: item["current_pos"]=item["position"]
        if "next_pos" not in item and "next_position" in item: item["next_pos"]=item["next_position"]
        if "raw_response" not in item and "response" in item: item["raw_response"]=item["response"]
        if "prompt_text" not in item and "prompt" in item: item["prompt_text"]=item["prompt"]
        if "parsed_belief_grid" not in item:
            parsed=item.get("parsed_response") or {}
            item["parsed_belief_grid"]=parsed.get("belief_grid") if isinstance(parsed,dict) else None
        item.setdefault("schema_version","1.0"); normalized.append(item)
    write_jsonl(run/"steps.jsonl",normalized)
    raw=read_json(source/"summary.json") if (source/"summary.json").exists() else []
    legacy=raw.get("episodes",[]) if isinstance(raw,dict) and isinstance(raw.get("episodes"),list) else (
        raw if isinstance(raw,list) else [])
    summary_by_id={str(x.get("episode_id")):x for x in legacy if isinstance(x,dict) and x.get("episode_id") is not None}
    grouped={}
    for step in normalized: grouped.setdefault(str(step["episode_id"]),[]).append(step)
    maps=[]; episodes=[]
    for episode_id,episode_steps in sorted(grouped.items()):
        episode_steps.sort(key=lambda x:int(x.get("step_id",0)))
        old=summary_by_id.get(episode_id,{})
        first,last=episode_steps[0],episode_steps[-1]
        size=int(old.get("size") or first.get("size") or 5)
        obstacles=old.get("obstacles") or first.get("obstacles")
        if obstacles is None:
            parsed=_map_obstacles(old.get("true_map") or old.get("obstacle_map"))
            if parsed: size,obstacles=parsed
        obstacles=obstacles or []
        start=old.get("start") or first.get("start") or [0,0]
        goal=old.get("goal") or first.get("goal") or [size-1,size-1]
        seed=int(old.get("seed") or first.get("seed") or 0)
        spec=GridSpec(episode_id,seed,size,(int(start[0]),int(start[1])),
                      (int(goal[0]),int(goal[1])),
                      frozenset((int(x[0]),int(x[1])) for x in obstacles))
        distance=shortest_path_length(spec)
        maps.append(GridSpec(episode_id,seed,size,spec.start,spec.goal,spec.obstacles,distance).to_dict())
        success=bool(old.get("success",last.get("reached_goal",False)))
        nsteps=int(old.get("steps",len(episode_steps)))
        episodes.append({"schema_version":"1.0","episode_id":episode_id,"seed":seed,
                         "size":size,"start":list(spec.start),"goal":list(spec.goal),
                         "obstacles":[list(x) for x in sorted(spec.obstacles)],
                         "success":success,"steps":nsteps,"shortest_path_length":distance,
                         "optimality_gap":nsteps-distance if success and distance is not None else None,
                         "final_position":last.get("next_pos"),"trajectory":old.get("trajectory",[])})
    write_jsonl(run/"maps.jsonl",maps); write_jsonl(run/"episodes.jsonl",episodes)
    write_json(run/"summary.json",summarize_episode_rows(episodes,normalized))
    write_manifest(run/"manifest.json",stage="migration",config={"source":str(source)},
                   counts={"episodes":len(episodes),"steps":len(normalized),"maps":len(maps)})
    return {"episodes":len(episodes),"steps":len(normalized),"maps":len(maps)}
