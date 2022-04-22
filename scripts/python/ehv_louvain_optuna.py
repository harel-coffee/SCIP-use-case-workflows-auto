import scanpy
import anndata
import optuna
from pathlib import Path
import pickle
import os

data_dir = Path(os.environ["VSC_DATA_VO_USER"]) / "datasets/weizmann/EhV/v2/results/scip/202202071958/"

def objective(trial):
    ad = anndata.read(data_dir / "anndata/cleaned.hda5")
    
    n_comps = trial.suggest_int("n_comps", low=10, high=100, step=5)
    n_neighbors = trial.suggest_int("n_neighbors", low=5, high=50, step=5)
    method = trial.suggest_categorical("method", choices=["umap", "gauss"])
    resolution = trial.suggest_float("resolution", low=0.5, high=5, step=0.1)
    do_scale = trial.suggest_categorical("do_scale", choices=[True, False])
    
    if do_scale:
        scanpy.pp.scale(ad)
    
    if n_comps > 0:
        scanpy.pp.pca(ad, n_comps=n_comps)
        
    scanpy.pp.neighbors(ad, n_neighbors=n_neighbors, knn=True, method=method, n_pcs=n_comps)
    scanpy.tl.louvain(ad, resolution=resolution, random_state=0)
    
    return -(ad.obs.groupby(["louvain", "meta_label"]).size().groupby("louvain").max() / ad.obs.groupby("louvain").size()).mean()

if __name__ == "__main__":
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
        study_name="purity"
        #storage="sqlite:///" + str(data_dir / "optuna/louvain.sqlite")
    )
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        pass
    
    with open(data_dir / "optuna/purity.pickle", "wb") as fh:
        pickle.dump(study, fh)
