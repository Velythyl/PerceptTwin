import os
import subprocess
import uuid

from hippo.reconstruction.assetlookup.TRELLISUtils.cache import TRELLIS_GEN_DIR
from hippo.utils.subproc import run_subproc


def convert(input_folder, target_uuid, target_folder):

    retcode = run_subproc(f"mv {input_folder}/model.glb {input_folder}/{target_uuid}.glb", shell=True)
    assert retcode.success

    user_uid = os.getuid()
    try:
        # need to first create the singularity container...
        singularity_path = f"{TRELLIS_GEN_DIR}/objathorconvert.sif"
        if not os.path.exists(singularity_path):
            run_subproc(
                f"singularity build {singularity_path} docker://velythyl/objathorconvert",
                shell=True,
                timeout=60 * 5,
                raise_timeout_exception=True,
            )

        def timeout_cleanup():
            print("Conversion timed out! Make sure the timeout was long enough!")
            run_subproc(f'pkill singularity', shell=True) # fixme a bit messy, should get the PID instead

        import tempfile
        tmpdir1 = tempfile.mkdtemp()
        tmpdir2 = tempfile.mkdtemp()
        tmpdir3 = tempfile.mkdtemp()

        retcode = run_subproc( # bro come on...
            f'singularity run \
                  --bind {input_folder}:/input \
                  --bind {target_folder}:/output \
                  --bind {tmpdir1}:/objathor/objathor/asset_conversion/colliders/linux \
                  --bind {tmpdir2}:/tmp \
                  --bind {tmpdir3}:/objathor/objathor/asset_conversion/colliders/__MACOSX \
                  --env UID={user_uid} \
                  {singularity_path} \
                  --uids={target_uuid} \
                  --glb_paths=/input/{target_uuid}.glb',
            shell=True,
            timeout=60 * 3,
            raise_timeout_exception=True,
            timeout_cleanup_func=timeout_cleanup
            )



        """
        def timeout_cleanup():
            print("Conversion timed out! Make sure the timeout was long enough!")
            run_subproc(f'podman kill $(podman ps -q --filter "name={CONTAINER_NAME}")', shell=True)

        run_subproc(f'podman pull velythyl/objathorconvert',
                              shell=True,
                              timeout=60 * 3,
                              raise_timeout_exception=True,
                              timeout_cleanup_func=timeout_cleanup
                              )

        retcode = run_subproc(f'podman run --name {CONTAINER_NAME} -v {input_folder}:/input -v {target_folder}:/output -u {user_uid}  velythyl/objathorconvert --uids={target_uuid} --glb_paths=/input/{target_uuid}.glb',
                              shell=True,
                              timeout=60 * 3,
                              raise_timeout_exception=True,
                              timeout_cleanup_func=timeout_cleanup
                              )"""
        assert retcode.success
        return True
    except subprocess.TimeoutExpired:
        return False
