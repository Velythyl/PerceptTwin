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





import hydra

HIPPO = None

from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    if "HYDRA_SPOOF" in os.environ and os.environ["HYDRA_SPOOF"] == "toplevel":
        print("Detected HYDRA_SPOOF environment variable. Launching bottom-level code now...")
        from hydra.core.hydra_config import HydraConfig 
        hc = HydraConfig.get()
        overrides = hc.overrides.task
        print(overrides)
        return run_subproc(f'cd /home/mila/c/charlie.gauthier/Holodeck/hippo && source ../venv/bin/activate && CUDA_VISIBLE_DEVICES=-1 HYDRA_SPOOF="bottomlevel" PYTHONPATH=..:$PYTHONPATH python3 main.py {" ".join(overrides)}', shell=True)

    
    print("Running HIPPO scene composition...")
    print("Loading CG...")
    set_api_key(cfg.secrets.openai_key)
    from hippo.conceptgraph.conceptgraph_to_hippo import get_hippos
    hipporoom, objects = get_hippos(cfg, Path(cfg.paths.scene_dir).resolve(), pad=cfg.scene.pad)

    print("CG loaded, number of objects:", len(objects))
    print("Loading AssetLookup...")
    from hippo.reconstruction.assetlookup.CLIPLookup import CLIPLookup
    from hippo.reconstruction.assetlookup.TRELLISLookup import TRELLISLookup
    from hippo.reconstruction.composer import SceneComposer
    global HIPPO
    trellis_proc = None
    if HIPPO is None:
        if cfg.assetlookup.method == "CLIP":
            HIPPO = CLIPLookup(cfg, OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, consider_size=True)
        elif cfg.assetlookup.method == "TRELLIS":

            if  "pop-os" not in socket.gethostname():
                print("Finding free port for TRELLIS server...")
                def find_free_port_in_range(start=1024, end=65535):
                    for port in range(start, end):
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            try:
                                s.bind(('', port))
                                return port
                            except OSError:
                                continue
                    raise RuntimeError("No free ports found in range")
                def find_free_port_in_range(start=1024, end=65535):
                    import socket
                    sock = socket.socket()
                    sock.bind(('', 0))
                    return sock.getsockname()[-1]
                free_port = find_free_port_in_range()
                print(f"Free port found: {free_port}")
                print("Hopefully, someone doesn't steal the port in the meantime...")

                print("Starting TRELLIS server...")
                # Ensure the TRELLIS server is started in a separate process
                trellis_proc = run_subproc(
                    f"cd /home/mila/c/charlie.gauthier/TRELLIS && source venv2/bin/activate && CUDA_VISIBLE_DEVICES=0 HF_HOME=/network/scratch/c/charlie.gauthier/hfcache python3 flaskserver.py --port={free_port}",
                    shell=True, immediately_return=True)
                TOTAL_WAIT_TIME = 0
                while True:
                    print("Waiting for TRELLIS server to start...")
                    sleep(10)
                    TOTAL_WAIT_TIME += 10
                    if TOTAL_WAIT_TIME > 600:
                        raise RuntimeError("TRELLIS server did not start in time! Check the logs for errors.")
                    if "WARNING: This is a development server." in trellis_proc.stdout_stderr.getvalue():
                        break
                    if "* Running on all addresses" in trellis_proc.stdout_stderr.getvalue():
                        break

                sleep(10)
                print("TRELLIS server started successfully.")
                cfg.assetlookup.client_port = free_port

            HIPPO = TRELLISLookup(cfg, OBJATHOR_ASSETS_DIR, do_weighted_random_selection=True, consider_size=True)
    print("AssetLookup loaded.")
    print("Looking up assets...")
    #os.makedirs(cfg.paths.out_scene_dir, exist_ok=True)
    composer = SceneComposer.create(
        cfg,
        asset_lookup=HIPPO,
        target_dir=cfg.paths.out_scene_dir,
        objectplans=objects,
        roomplan=hipporoom
    )
    print("Assets looked up.")

    print("Writing down compositions...")
    if cfg.scene.scene_generation_method == "random":
        composer.write_random_compositions(cfg.scene.num_scenes_to_generate)
    elif cfg.scene.scene_generation_method == "in_order":
        composer.write_compositions_in_order(cfg.scene.num_scenes_to_generate)
    elif cfg.scene.scene_generation_method == "most_likely":
        composer.write_most_likely_composition(cfg.scene.num_scenes_to_generate)

    #composer.write_compositions_in_order(1)

    print("Taking topdown view...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # this IS necessary, but WHY??? We already set it to -1 above! somehow it flips back to 0!? And this only happens on a multirun, not on `mila code`!
    composer.take_photos()
    print("Done with scene composition and topdown view.")

    if trellis_proc is not None:
        print("Killing TRELLIS server...")
        trellis_proc.process.kill()
        print("TRELLIS server killed.")

    print("Bye, have a nice day!")
    os._exit(0)

if __name__ == '__main__':


    import socket
    from hippo.utils.subproc import run_subproc
    os.environ["XDG_RUNTIME_DIR"] = "/tmp"
    os.makedirs("/tmp/.X11-unix", exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if True: #True:# and "pop-os" in socket.gethostname():
        print("Running in Xvfb...")
        run_subproc(f'Xvfb :99 -screen 10 180x180x24', shell=True, immediately_return=True)
        os.environ["DISPLAY"] = f":99"
    print("launching main...")
    main()
