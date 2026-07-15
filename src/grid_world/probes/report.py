from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def build_report(run_dir):
    run=Path(run_dir); out=run/"probes"
    frame=pd.read_csv(out/"probe_results.csv")
    score,std="macro_f1_mean","macro_f1_std"
    best=frame.loc[frame.groupby("task")[score].idxmax()].sort_values(["task_group","task"])
    best.to_csv(out/"best_by_task.csv",index=False)
    figures=out/"figures"; figures.mkdir(parents=True,exist_ok=True)
    paths=[]
    for group,gf in frame.groupby("task_group"):
        pivot=gf.pivot_table(index="layer",columns="position",values=score,aggfunc="mean").sort_index()
        fig,ax=plt.subplots(figsize=(max(7,len(pivot.columns)*1.8),5))
        image=ax.imshow(pivot.values,aspect="auto")
        ax.set_title(f"{group}: mean macro-F1")
        ax.set_xlabel("Representation position"); ax.set_ylabel("Hidden-state layer")
        ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha="right")
        ax.set_yticks(range(len(pivot.index)),pivot.index)
        fig.colorbar(image,ax=ax,label="Macro-F1"); fig.tight_layout()
        path=figures/f"{group}_heatmap.png"; fig.savefig(path,dpi=160); plt.close(fig); paths.append(path)
    summary=(best.groupby("task_group").agg(
        tasks=("task","count"),mean_best_macro_f1=(score,"mean"),
        median_best_macro_f1=(score,"median"),mean_split_std=(std,"mean")).reset_index())
    summary.to_csv(out/"group_summary.csv",index=False)
    lines=["# Probe report","","## Best score by task group","",
           summary.to_markdown(index=False),"","## Best position/layer per task","",
           best[["task_group","task","position","layer",score,std,
                 "majority_macro_f1_mean"]].to_markdown(index=False),"",
           "## Interpretation notes","",
           "- Compare macro-F1 with the majority macro-F1 baseline.",
           "- A high feedback-span score can reflect information directly present in the prompt.",
           "- Probe decodability does not establish causal use.",""]
    (out/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    return {"best_by_task":str(out/"best_by_task.csv"),"summary":str(out/"summary.md"),
            "figures":[str(x) for x in paths]}
