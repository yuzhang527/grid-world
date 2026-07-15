from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from grid_world.probes.tasks import load_catalog, select_tasks
from grid_world.utils.io import read_json, read_jsonl
from grid_world.utils.manifest import write_manifest

def _names(value, available, defaults):
    if value == "all": return list(available)
    if value == "auto":
        selected = [x for x in defaults if x in available]
        return selected or list(available)
    selected = [x.strip() for x in value.split(",") if x.strip()]
    missing = set(selected)-set(available)
    if missing: raise ValueError(f"Unavailable positions: {missing}")
    return selected

def _layers(value, available):
    if value in {"all","auto"}: return list(available)
    selected = [int(x.strip()) for x in value.split(",") if x.strip()]
    missing = set(selected)-set(available)
    if missing: raise ValueError(f"Unavailable layers: {missing}")
    return selected

def _metrics(y_true, y_pred, majority):
    baseline = np.full(len(y_true), majority)
    return {
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "majority_accuracy":float(accuracy_score(y_true,baseline)),
        "majority_macro_f1":float(f1_score(y_true,baseline,average="macro",zero_division=0)),
    }

def _data(run, tasks):
    meta = read_jsonl(run / "activations" / "meta.jsonl")
    targets = {(str(x["episode_id"]),int(x["step_id"])):x
               for x in read_jsonl(run / "targets" / "targets.jsonl")}
    indices, groups, aligned = [], [], []
    for item in meta:
        key = (str(item["episode_id"]),int(item["step_id"]))
        if key in targets:
            indices.append(int(item["row_index"])); groups.append(key[0]); aligned.append(targets[key])
    task_data = {}
    for task in tasks:
        values = [row.get(task) for row in aligned]
        classes = sorted({str(x) for x in values if x is not None})
        if len(classes) < 2: continue
        lookup = {x:i for i,x in enumerate(classes)}
        y = np.full(len(values),-1,dtype=np.int64)
        for i,value in enumerate(values):
            if value is not None: y[i] = lookup[str(value)]
        task_data[task] = {"classes":classes,"y":y}
    if not task_data: raise ValueError("No trainable tasks")
    return np.asarray(indices),np.asarray(groups),task_data

def _aggregate(frame):
    metrics = ["accuracy","balanced_accuracy","macro_f1",
               "majority_accuracy","majority_macro_f1"]
    keys = ["task","task_group","position","layer","backend","classes"]
    agg = {x:["mean","std"] for x in metrics}; agg["num_test"] = "mean"
    out = frame.groupby(keys,dropna=False).agg(agg).reset_index()
    out.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col,tuple) else str(col)
                   for col in out.columns]
    return out

def _sklearn(X,mask,indices,groups,task_data,catalog,positions,pmap,layers,lmap,
             splits,test_size,seed,min_class_count):
    rows=[]
    split_list=list(GroupShuffleSplit(n_splits=splits,test_size=test_size,
                                      random_state=seed).split(np.zeros(len(indices)),groups=groups))
    total=len(task_data)*len(positions)*len(layers); progress=0
    for task,info in task_data.items():
        for position in positions:
            for layer in layers:
                progress += 1
                print(f"[probe/sklearn] {progress}/{total} {task} {position} L{layer}",flush=True)
                valid=mask[indices,pmap[position]]
                for sid,(train,test) in enumerate(split_list):
                    train=train[(info["y"][train]>=0)&valid[train]]
                    test=test[(info["y"][test]>=0)&valid[test]]
                    if not len(train) or not len(test): continue
                    counts=Counter(info["y"][train].tolist())
                    if len(counts)<2 or min(counts.values())<min_class_count: continue
                    scaler=StandardScaler()
                    xtr=scaler.fit_transform(np.asarray(X[indices[train],pmap[position],lmap[layer]],
                                                       dtype=np.float32))
                    xte=scaler.transform(np.asarray(X[indices[test],pmap[position],lmap[layer]],
                                                   dtype=np.float32))
                    model=LogisticRegression(max_iter=1000,class_weight="balanced",solver="lbfgs")
                    model.fit(xtr,info["y"][train]); pred=model.predict(xte)
                    rows.append({"task":task,"task_group":catalog[task]["group"],
                                 "position":position,"layer":layer,"backend":"sklearn",
                                 "split":sid,"classes":json.dumps(info["classes"]),
                                 "num_train":len(train),"num_test":len(test),
                                 **_metrics(info["y"][test],pred,counts.most_common(1)[0][0])})
    return rows

def _torch(X,mask,indices,groups,task_data,catalog,positions,pmap,layers,lmap,
           splits,test_size,seed,min_class_count,device,epochs,lr,weight_decay):
    import torch
    torch.manual_seed(seed)
    names=list(task_data); offsets={}; total_outputs=0
    for task in names:
        n=len(task_data[task]["classes"]); offsets[task]=(total_outputs,total_outputs+n); total_outputs+=n
    split_list=list(GroupShuffleSplit(n_splits=splits,test_size=test_size,
                                      random_state=seed).split(np.zeros(len(indices)),groups=groups))
    rows=[]; total=len(positions)*len(layers)*splits; progress=0
    for position in positions:
        valid_position=mask[indices,pmap[position]]
        for layer in layers:
            base=np.asarray(X[indices,pmap[position],lmap[layer]],dtype=np.float32)
            for sid,(train_base,test_base) in enumerate(split_list):
                progress+=1
                print(f"[probe/torch] {progress}/{total} {position} L{layer} split={sid}",flush=True)
                train_idx=train_base[valid_position[train_base]]
                test_idx=test_base[valid_position[test_base]]
                if not len(train_idx) or not len(test_idx): continue
                mean=base[train_idx].mean(0); std=base[train_idx].std(0); std[std<1e-6]=1
                xtr=torch.from_numpy((base[train_idx]-mean)/std).to(device)
                xte=torch.from_numpy((base[test_idx]-mean)/std).to(device)
                valid_tasks=[]; train_labels={}; test_labels={}; weights={}
                for task in names:
                    y=task_data[task]["y"]; ytr=y[train_idx]; yte=y[test_idx]
                    trmask=ytr>=0; temask=yte>=0; counts=Counter(ytr[trmask].tolist())
                    if len(counts)<2 or not temask.any() or min(counts.values())<min_class_count:
                        continue
                    valid_tasks.append(task)
                    train_labels[task]=(torch.from_numpy(ytr).long().to(device),
                                        torch.from_numpy(trmask).bool().to(device))
                    test_labels[task]=(yte,temask,counts)
                    nclasses=len(task_data[task]["classes"]); w=np.ones(nclasses,dtype=np.float32)
                    total_count=sum(counts.values())
                    for cid in range(nclasses):
                        w[cid]=total_count/(nclasses*max(1,counts.get(cid,0)))
                    weights[task]=torch.from_numpy(w).to(device)
                if not valid_tasks: continue
                model=torch.nn.Linear(base.shape[1],total_outputs).to(device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
                model.train()
                for _ in range(epochs):
                    optimizer.zero_grad(set_to_none=True); logits=model(xtr); losses=[]
                    for task in valid_tasks:
                        start,end=offsets[task]; labels,tmask=train_labels[task]
                        losses.append(torch.nn.functional.cross_entropy(
                            logits[tmask,start:end],labels[tmask],weight=weights[task]))
                    torch.stack(losses).mean().backward(); optimizer.step()
                model.eval()
                with torch.inference_mode(): logits=model(xte)
                for task in valid_tasks:
                    start,end=offsets[task]; yte,temask,counts=test_labels[task]
                    pred=logits[:,start:end].argmax(-1).cpu().numpy()[temask]
                    rows.append({"task":task,"task_group":catalog[task]["group"],
                                 "position":position,"layer":layer,"backend":"torch",
                                 "split":sid,"classes":json.dumps(task_data[task]["classes"]),
                                 "num_train":sum(counts.values()),"num_test":int(temask.sum()),
                                 **_metrics(yte[temask],pred,counts.most_common(1)[0][0])})
                del model,optimizer,xtr,xte
                if str(device).startswith("cuda"): torch.cuda.empty_cache()
    return rows

def train_probes(*,run_dir,groups="local,cells,planning",positions="auto",layers="all",
                 backend="torch",device="cuda:0",splits=20,test_size=0.2,seed=123,
                 min_class_count=5,epochs=60,learning_rate=0.03,weight_decay=1e-4):
    run=Path(run_dir)
    X=np.load(run/"activations"/"X.npy",mmap_mode="r")
    mask=np.load(run/"activations"/"position_mask.npy",mmap_mode="r")
    available_positions=read_json(run/"activations"/"positions.json")
    available_layers=np.load(run/"activations"/"layers.npy").astype(int).tolist()
    position_names=_names(positions,available_positions,
                          ["mean_last_feedback","mean_current_belief_grid","pre_action_token","prompt_last"])
    layer_names=_layers(layers,available_layers)
    pmap={x:available_positions.index(x) for x in position_names}
    lmap={x:available_layers.index(x) for x in layer_names}
    catalog=load_catalog(run); tasks=select_tasks(catalog,groups)
    indices,episode_groups,task_data=_data(run,tasks)
    print(f"[probe] X={X.shape} rows={len(indices)} tasks={len(task_data)} "
          f"positions={position_names} layers={layer_names}",flush=True)
    common=dict(X=X,mask=mask,indices=indices,groups=episode_groups,task_data=task_data,
                catalog=catalog,positions=position_names,pmap=pmap,layers=layer_names,lmap=lmap,
                splits=splits,test_size=test_size,seed=seed,min_class_count=min_class_count)
    if backend=="sklearn":
        rows=_sklearn(**common)
    elif backend=="torch":
        rows=_torch(**common,device=device,epochs=epochs,lr=learning_rate,weight_decay=weight_decay)
    else:
        raise ValueError("backend must be torch or sklearn")
    if not rows: raise RuntimeError("No probe results produced")
    out=run/"probes"; out.mkdir(parents=True,exist_ok=True)
    split_frame=pd.DataFrame(rows); split_frame.to_csv(out/"probe_results_splits.csv",index=False)
    aggregate=_aggregate(split_frame); aggregate.to_csv(out/"probe_results.csv",index=False)
    write_manifest(out/"manifest.json",stage="probes",
                   config={"groups":groups,"positions":positions,"layers":layers,
                           "backend":backend,"device":device,"splits":splits,"epochs":epochs},
                   counts={"split_rows":len(split_frame),"aggregate_rows":len(aggregate)},
                   upstream={"activations":str(run/"activations"),"targets":str(run/"targets")})
    return aggregate
