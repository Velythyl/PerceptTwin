import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, filter_overrides
from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

from typing import Any, Sequence

from hydra.core.utils import JobReturn
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher


class PackedSubmititLauncher(BaseSubmititLauncher):
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    def launch_batch(
        self,
        sweep_overrides: List[List[str]],
        job_dir_key: List[str],
        job_num: List[int],
        job_id: List[str],
        singleton_state: List[Dict[type, Singleton]],
    ) -> JobReturn:
        import submitit

        log.info(self.config)
        log.info(os.environ)

        task_id = submitit.JobEnvironment().global_rank
        return self(
            sweep_overrides[task_id],
            job_dir_key[task_id],
            job_num[task_id],
            job_id[task_id],
            singleton_state[task_id],
        )

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        import submitit

        assert self.config is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0
        params = self.params
        # build executor
        init_params = {"folder": self.params["submitit_folder"]}
        specific_init_keys = {"max_num_timeout"}

        init_params.update(
            **{
                f"{self._EXECUTOR}_{x}": y
                for x, y in params.items()
                if x in specific_init_keys
            }
        )
        init_keys = specific_init_keys | {"submitit_folder"}
        executor = submitit.AutoExecutor(cluster=self._EXECUTOR, **init_params)

        # specify resources/parameters
        baseparams = set(OmegaConf.structured(BaseQueueConf).keys())
        params = {
            x if x in baseparams else f"{self._EXECUTOR}_{x}": y
            for x, y in params.items()
            if x not in init_keys
        }
        executor.update_parameters(**params)

        log.info(
            f"Submitit '{self._EXECUTOR}' sweep output dir : "
            f"{self.config.hydra.sweep.dir}"
        )
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        if "mode" in self.config.hydra.sweep:
            mode = int(str(self.config.hydra.sweep.mode), 8)
            os.chmod(sweep_dir, mode=mode)

        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    idx,
                    f"job_id_for_{idx}",
                    Singleton.get_state(),
                )
            )
        jobs = executor.map_array(
            self.launch_batch,
            *list(
                batch(jps, self.params["tasks_per_node"]) for jps in zip(*job_params)
            ),
        )
        return [res for j in jobs for res in j.results()]


def batch(x, bs):
    return [x[i : i + bs] for i in range(0, len(x), bs)]


class LocalLauncher(PackedSubmititLauncher):
    _EXECUTOR = "local"


class SlurmLauncher(PackedSubmititLauncher):
    _EXECUTOR = "slurm"