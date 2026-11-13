"""
Optuna search over imitation pretraining settings (src/imitation/hpo.py), one study per representation.

Config: configs/rllib/pretrain_hpo_graph36.yaml (GCN with per-substation readout on IEEE-36; search space
and pruner in `hpo`, objective: mean validation regret of top-20 simulation), or
configs/rllib/pretrain_hpo_graph36_mlp.yaml for the MLP baseline. Each worker
joins the study `experiment.name` in `imitation.out_dir/study.log` and runs trials until the study
has `hpo.n_trials`. Run once with `hpo.convert_only=true` first, so the workers find the converted
observations in the cache instead of all converting them at once.

Writes to `imitation.out_dir` (results/junior/<study>/):
    study.log           Optuna journal
    config.yaml         base config of the study (env config for do-nothing rollouts)
    trial_<n>/          run directory of every trial (src/imitation/pretrain_run.py)
Summarize with experiments/summarize_hpo.py.

Example:
    PYTHONPATH=$(pwd)/src python experiments/pretrain_hpo.py obs_space=graph_comparison/substation \\
        experiment.name=graph36_substation imitation.dataset=data/imitation/case36_k208_greedy_hard.npz \\
        imitation.device=cuda hpo.worker_id=0
"""
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from core.constants import set_seed
from core.rllib_config import build_env_config, register_custom_models
from imitation.hpo import load_study, make_objective, run_study, study_sampler_seed
from imitation.pretrain_run import CONFIG_FILE, build_datasets, check_pretraining_config, converter_observation_space

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/rllib", config_name="pretrain_hpo_graph36")
def main(cfg: DictConfig) -> None:
    env_config = build_env_config(cfg, "train")
    check_pretraining_config(cfg, env_config)
    set_seed(cfg.experiment.seed)
    register_custom_models()
    study_dir = Path(cfg.imitation.out_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, study_dir / CONFIG_FILE, resolve=True)

    sets = build_datasets(cfg, env_config)
    if cfg.hpo.convert_only:
        logger.info("Observations converted and cached (hpo.convert_only=true)")
        return
    sampler_seed = study_sampler_seed(cfg.experiment.name, cfg.hpo.worker_id, cfg.hpo.sampler_seed)
    study = load_study(study_dir, cfg.experiment.name, sampler_seed=sampler_seed,
                       pruner=OmegaConf.to_container(cfg.hpo.pruner))
    study.set_user_attr("dataset", str(cfg.imitation.dataset))
    run_study(study, make_objective(cfg, sets, converter_observation_space(env_config), study_dir), cfg.hpo.n_trials)


if __name__ == "__main__":
    logging.getLogger("pandapower").setLevel(logging.WARNING)
    main()
