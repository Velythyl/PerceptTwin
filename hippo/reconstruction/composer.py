import copy
import dataclasses
import itertools
import json
import os
import random
from typing import Tuple, Dict, List, Union

import numpy as np
from tqdm import tqdm

from ai2holodeck.generation.utils import get_top_down_frame, get_hippo_room_images, get_replica_pov, \
    get_top_down_seg_frame
from hippo.ai2thor_hippo_controller import get_hippo_controller, get_sim
from hippo.reconstruction.assetlookup._AssetLookup import AssetLookup
from hippo.reconstruction.llm_annotation import LLM_annotate
from hippo.reconstruction.scenedata import HippoObject, HippoRoomPlan
from hippo.utils.selfdataclass import SelfDataclass
import multiprocessing as mp
from omegaconf import DictConfig, OmegaConf, ListConfig

with open("../ai2holodeck/generation/empty_house.json", "r") as f:
    DEFAULT_SCENE = json.load(f)



@dataclasses.dataclass
class SceneComposer(SelfDataclass):
    cfg: DictConfig
    asset_lookup: AssetLookup
    target_dir: str
    objectplans: Tuple[HippoObject]
    roomplan: HippoRoomPlan
    consider_walls_and_floors_in_scene_count: bool
    #scene: Dict

    @classmethod
    def create(cls, cfg, asset_lookup: AssetLookup, target_dir: str, objectplans: Tuple[HippoObject], roomplan: HippoRoomPlan, consider_walls_and_floors_in_scene_count=False):
        looked_up_objectplans = []
        for obj in tqdm(objectplans, desc="Looking up viable assets..."):
            looked_up_obj = asset_lookup.lookup_assets(obj) #[:KEEP_TOP_K]

            if cfg.skillprediction.method == None:
                looked_up_obj2 = looked_up_obj
            else:
                looked_up_obj2 = LLM_annotate(cfg, looked_up_obj)

            looked_up_objectplans.append(looked_up_obj2)



        #objectplans = tuple([asset_lookup.lookup_assets(x,asset_lookup.objaverse_asset_dir)[:KEEP_TOP_K] for x in objectplans])


        ret = cls(
            cfg=cfg,
            asset_lookup=asset_lookup,
            target_dir=target_dir,
            objectplans=looked_up_objectplans,
            roomplan=roomplan,
            consider_walls_and_floors_in_scene_count=consider_walls_and_floors_in_scene_count
            #scene=scene
        )

        print("Number of scenes in composer:", ret.number_of_scenes)
        return ret

    @property
    def number_of_scenes(self):
        from math import factorial
        from functools import reduce

        def multinomial(counts):
            total = sum(counts)
            denominator = reduce(lambda x, y: x * factorial(y), counts, 1)
            return factorial(total) // denominator

        objs = [len(obj) for obj in self.objectplans]
        for num, obj in zip(objs, self.objectplans):
            print(f'We found {num} assets for object "{obj.object_name}"')

        if self.consider_walls_and_floors_in_scene_count:
            if isinstance(self.roomplan.wall_type, list) or isinstance(self.roomplan.wall_type, ListConfig):
                objs += [len(self.roomplan.wall_type)]
            if isinstance(self.roomplan.floor_type, list) or isinstance(self.roomplan.floor_type, ListConfig):
                objs += [len(self.roomplan.floor_type)]

        return multinomial(objs)

    @property
    def done_paths(self):
        all_paths_in_target_dir = list(map(lambda x: f"{self.target_dir}/{x}", os.listdir(self.target_dir)))

        ret = []
        for p in all_paths_in_target_dir:
            works = True
            for n in ["scene.json", "cfg.yaml", "concrete_assets"]:
                if not os.path.exists(f"{p}/{n}"):
                    works = False
                    break
            if works:
                ret.append(p)

        return ret

    @property
    def asset_dir(self):
        return self.asset_lookup.objaverse_asset_dir

    @property
    def _object_indices(self):
        ret = []
        for obj in self.objectplans:
            ret.append(list(np.arange(len(obj))))
        return ret

    @property
    def _object_indices_prod(self):
        product = (itertools.product(*self._object_indices))
        #print(f"Found {len(product)} valid scenes.")
        return product

    def __len__(self):
        return len(self._object_indices_prod)

    def generate_compositions_in_order(self, prod=None, MAX_NUM=10):
        if prod is None:
            prod = self._object_indices_prod

        COUNTER = 0
        for possible_scene in prod:
            new_objectplans = []
            for i, obj_i in enumerate(possible_scene):
                new_objectplans.append(self.objectplans[i][obj_i])
            yield self.replace(objectplans=tuple(new_objectplans))
            COUNTER += 1
            if COUNTER >= MAX_NUM:
                break

    def _write_compositions(self, generator, generationname, prefetch=32):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        import json
        from omegaconf import OmegaConf
        from tqdm import tqdm
        import itertools
        def process_one(index, selves):
            WRITE_DIR = os.path.join(self.target_dir, f"{generationname}{index}")
            os.makedirs(WRITE_DIR, exist_ok=False)

            CONCRETIZATION_DIR = os.path.join(WRITE_DIR, "concrete_assets")
            os.makedirs(CONCRETIZATION_DIR, exist_ok=False)

            scene = selves.get_scene(CONCRETIZATION_DIR)

            with open(os.path.join(WRITE_DIR, "scene.json"), "w") as f:
                json.dump(scene, f, indent=4)

            with open(os.path.join(WRITE_DIR, "cfg.yaml"), "w") as f:
                OmegaConf.save(config=self.cfg, f=f)

        counter = 0
        for self in generator:
            process_one(counter, self)
            counter += 1
        return
        #exit()
        #process_one(0, next(generator))

        with ThreadPoolExecutor(max_workers=self.cfg.parallelism.composer_selves_max_workers) as executor:
            futures = {}
            index_gen = itertools.count()
            gen_iter = (generator)

            pbar = tqdm(desc="Writing compositions", unit="scene")

            # Prefill the first N tasks
            try:
                for _ in range(prefetch):
                    i = next(index_gen)
                    selves = next(gen_iter)
                    futures[executor.submit(process_one, i, selves)] = i
            except StopIteration:
                pass  # generator had fewer than `prefetch` items

            # As each future completes, submit another
            while futures:
                for future in as_completed(futures):
                    i = futures.pop(future)
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in composition {i}: {e}")
                    pbar.update(1)

                    try:
                        j = next(index_gen)
                        selves = next(gen_iter)
                        futures[executor.submit(process_one, j, selves)] = j
                    except StopIteration:
                        break  # done submitting new work

    def write_compositions_in_order(self, MAX_NUM=10):
        self._write_compositions(self.generate_compositions_in_order(MAX_NUM=MAX_NUM), "in_order_")

    def generate_random_compositions(self, MAX_NUM=10):
        if max([len(x) for x in self._object_indices]) <= 10:
            prod = list(self._object_indices_prod)
            random.shuffle(prod)
            yield from self.generate_compositions_in_order(prod, MAX_NUM)
            return

        def generator():
            USED = set()

            def gen_one():
                to_use = []
                for obj in self._object_indices:
                    to_use.append(random.choice(obj))
                to_use = tuple(to_use)
                return to_use

            def gen_unique(callcount=0):
                to_use = gen_one()
                if to_use in USED:
                    if callcount >= MAX_NUM:
                        raise Exception("Too many calls to generate unique compositions. Perhaps you requested too many compositions for a too-small set of assets? If not, you might need to implement reservoir sampling here.")
                    return gen_unique(callcount=callcount+1)

                USED.add(to_use)
                return to_use

            for _ in range(MAX_NUM):
                unique = gen_unique()
                ret_self = next(self.generate_compositions_in_order([unique]))
                yield ret_self

        yield from generator()

    def write_random_compositions(self, MAX_NUM=10):
        generator = self.generate_random_compositions(MAX_NUM=MAX_NUM)
        return self._write_compositions(generator, "random_")

    def generate_most_likely_composition(self):
        objs = []
        for obj in self.objectplans:
            i = np.argmax(np.array(obj._found_scores))
            objs.append(obj[i])

        return self.replace(objectplans=tuple(objs))

    def write_most_likely_composition(self):
        return self._write_compositions(self.generate_most_likely_composition(), "most_likely_")

    def get_scene(self, concrete_asset_dir):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import copy

        scene = self.asset_lookup.generate_rooms(copy.deepcopy(DEFAULT_SCENE), hipporoom=self.roomplan)

        #print(self.objectplans[0]._position, self.objectplans[1]._position)
        scene = copy.deepcopy(scene)
        if 'objects' not in scene:
            scene['objects'] = []

        def process_object(obj):
            if len(obj) > 1:
                obj = obj[0]
            obj.concretize(self.cfg, concrete_asset_dir)
            return obj.as_holodeckdict()

        for obj in self.objectplans:
            new_object = process_object(obj)
            scene['objects'].append(new_object)
        return scene

        with ThreadPoolExecutor(max_workers=self.cfg.parallelism.composer_concretizer_max_workers) as executor:
            futures = [executor.submit(process_object, obj) for obj in self.objectplans]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Concretizing objects..."):
                new_object = future.result()
                scene['objects'].append(new_object)

        return scene

    def take_photos(self, paths: Union[List[str],str]=None):
        if paths is None:
            paths = self.done_paths
        if isinstance(paths, str):
            paths = [paths]

        todo_paths = []
        for path in paths:
            if path.endswith(".json"):
                todo_paths.append(path)
            else:
                todo_paths.append(path + "/scene.json")

        photo_funcs = [("replica_pov.png", get_replica_pov), ("topdown.png", get_top_down_frame), ("topdown_seg.png", get_top_down_seg_frame)]#, ("room_image.png", get_hippo_room_images)]

        def process_path(path):
            print(f"Processing path: {path}")
            print("With cuda visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
            try:
                controller = get_sim(path, width=2048, height=2048, renderInstanceSegmentation=True).controller
            except Exception as e:
                print(f"Failed to create controller for {path}: {e}")
                raise e
            print(f"Controller created for {path}")
            with open(path) as f:
                scene_dict = json.load(f)

            for name, photo_func in photo_funcs:
                img = photo_func(controller, cfg=self.cfg, scene=scene_dict)
                if not isinstance(img, list):
                    savepath = os.path.join(os.path.dirname(path), name)
                    img.save(savepath)
                else:
                    for i, im in enumerate(img):
                        p = os.path.join(os.path.dirname(path), name).replace(".png", f"{i}.png")
                        im.save(p)
            print(f"Photos taken for {path}")
            controller.stop()
            print(f"Controller stopped for {path}")

        for p in todo_paths:
            process_path(p)
        return

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import copy
        with ThreadPoolExecutor(max_workers=1) as executor: # fixme to augment this, would need to remove the unlink to the build dir found in hippo controller
            futures = {executor.submit(process_path, path): path for path in todo_paths}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating top-down images"):
                path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to process {path}: {e}")
