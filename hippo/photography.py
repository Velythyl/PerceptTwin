import os
import tempfile
from llmqueries.llm import set_api_key

from hippo.utils.subproc import run_subproc
from hippo.utils.file_utils import get_tmp_file

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pathlib import Path
from time import sleep

from ai2holodeck.constants import OBJATHOR_ASSETS_DIR

from hippo.conceptgraph.conceptgraph_to_hippo import get_hippos




import hydra

HIPPO = None

from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    from hippo.reconstruction.composer import SceneComposer

    target_dir = "/Users/charlie/Projects/Holodeck/hippo/sampled_scenes/realscenes/mugtmp/tmp"
    composer = SceneComposer(
        cfg,
        asset_lookup=None,
        target_dir=target_dir,
        objectplans=None,
        roomplan=None,
        consider_walls_and_floors_in_scene_count=False
    )

    print("Taking photos...")
    composer.take_photos()
    print("Bye, have a nice day!")
    os._exit(0)

if __name__ == '__main__':
    import socket

    if False and "pop-os" in socket.gethostname():
        run_subproc(f'Xvfb :99 -screen 10 180x180x24', shell=True, immediately_return=True)
        os.environ["DISPLAY"] = f":99"

    main()


"""

PYTHONPATH=..:$PYTHONPATH xvfb-run -a -s "-screen 0 1400x900x24" python3 main.py



PYTHONPATH=..:$PYTHONPATH python3 main.py  --multirun hydra/launcher=sbatch +hydra/sweep=sbatch hydra.launcher.timeout_min=30  hydra.launcher.gres=gpu:0 hydra.launcher.cpus_per_task=6 hydra.launcher.mem_gb=32 hydra.launcher.array_parallelism=60 hydra.launcher.partition=long-cpu  secrets=secrets_cluster  paths.conceptgraphs_dir="./datasets/concept-nodes" assetfitting=rot_and_axisscale,rot_and_aspect_fill,rot_and_aspect_fit  paths.scene_id='replica_office2_cg-detector_2025-06-10-20-34-08.445148,replica_room0_cg-detector_2025-06-10-20-16-50.243828,replica_office0_cg-detector_2025-06-10-20-25-14.795112,replica_office3_cg-detector_2025-06-10-20-40-27.070705,replica_room1_cg-detector_2025-06-10-20-04-54.511655,replica_office1_cg-detector_2025-06-10-20-29-28.683537,replica_office4_cg-detector_2025-06-10-20-45-18.828598,replica_room2_cg-detector_2025-06-10-20-21-16.981182'  paths.out_dir="./sampled_scenes/firstrun"

"""