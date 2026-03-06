import json
import os
import shutil
import uuid

import numpy as np
from PIL import Image

from hippo.reconstruction.assetlookup.CLIPLookup import CLIPLookup
from hippo.reconstruction.assetlookup.TRELLISUtils import convert_to_ai2thor, cache
from hippo.reconstruction.assetlookup.TRELLISUtils.flaskclient import Trellis3DClient
from hippo.reconstruction.scenedata import HippoRoomPlan, HippoObject

DEFAULT_THOR_METADATA = {       # doesn't actually matter much; gets overwritten when we use LLM skill annotations
    "assetMetadata": {
        "primaryProperty": "CanPickup",
        "secondaryProperties": [
            "Receptacle"
        ]
    }
}


class TRELLISLookup:

    def __init__(self, cfg, objaverse_asset_dir: str, do_weighted_random_selection: bool, consider_size: bool):
        self.clip_lookup = CLIPLookup(cfg, objaverse_asset_dir, do_weighted_random_selection, consider_size)
        self._TRELLIS_client = None
        self.cfg = cfg

    @property
    def TRELLIS_client(self):
        if self._TRELLIS_client is None:
            self._TRELLIS_client = Trellis3DClient(port=self.cfg.assetlookup.client_port)
        return self._TRELLIS_client


    def generate_rooms(self, scene, plan: str=None, hipporoom: HippoRoomPlan=None):
        return self.clip_lookup.generate_rooms(scene, plan, hipporoom)

    def lookup_assets(self, obj: HippoObject, size_comparison_tresh=0.1):
        image_dir = obj._cg_paths["rgb"]
        masks_dir = obj._cg_paths["mask"]

        def path_contains_symlink(path: str) -> bool:
            """
            Check if the given path or any of its components is a symlink.

            Args:
                path (str): The filesystem path to check.

            Returns:
                bool: True if the path or any part of it is a symlink, False otherwise.
            """
            if isinstance(path, list) or isinstance(path, tuple):
                return any(path_contains_symlink(p) for p in path)

            # Normalize to absolute path
            path = os.path.abspath(path)

            # Walk upward through all components
            while path != os.path.dirname(path):  # until reaching root
                if os.path.islink(path):
                    return True
                path = os.path.dirname(path)

            return False
        if path_contains_symlink(image_dir) or path_contains_symlink(masks_dir):
            raise ValueError(f"Path {image_dir} or {masks_dir} contains a symlink, which is not supported by the TRELLIS lookup due to interactions with Singularity containers.")


        if not isinstance(image_dir, str):
            if len(image_dir) == 1:
                assert len(masks_dir) == 1
                image_dir = image_dir[0]
                masks_dir = masks_dir[0]
            else:

                assert not isinstance(masks_dir, str)
                assert isinstance(image_dir, list) or isinstance(image_dir, tuple)

                import hashlib

                def hash_string(s):
                    return hashlib.sha1(s.encode()).hexdigest()

                hashed_image_dirs = (list(map(hash_string, image_dir)))
                hashed_masks_dirs = (list(map(hash_string, masks_dir)))

                composed_image_dir = "/tmp/" + "-".join(hashed_image_dirs)
                composed_masks_dir = "/tmp/" + "-".join(hashed_masks_dirs)

                os.makedirs(composed_image_dir, exist_ok=True)
                os.makedirs(composed_masks_dir, exist_ok=True)

                for (im_dir, hash_im_dir), (ma_dir, hash_ma_dir) in zip(zip(image_dir, hashed_image_dirs), zip(masks_dir,hashed_masks_dirs)):
                    for f in os.listdir(im_dir):
                        src_path = os.path.join(im_dir, f)
                        dst_path = os.path.join(composed_image_dir, hash_im_dir+ f)
                        shutil.copy2(src_path, dst_path)

                    for f in os.listdir(ma_dir):
                        src_path = os.path.join(ma_dir, f)
                        dst_path = os.path.join(composed_masks_dir, hash_im_dir + f)    # note: using the same im dir! image and mask file names are paired
                        shutil.copy2(src_path, dst_path)

                image_dir = composed_image_dir
                masks_dir = composed_masks_dir

        if self.cfg.assetlookup.use_masks:
            masked_dir = image_dir.replace("rgb", "masked")
            os.makedirs(masked_dir, exist_ok=True)
            # Get list of RGB images (case-insensitive PNG check)
            rgb_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]

            for rgb_file in rgb_files:
                # Construct paths
                rgb_path = os.path.join(image_dir, rgb_file)
                
                # Load images
                rgb_img = Image.open(rgb_path)

                mask_path = os.path.join(masks_dir, rgb_file)  # Assuming same filename
                try:
                    mask_img = Image.open(mask_path).convert('L')  # Convert mask to grayscale
                except:
                    mask_img = np.load(mask_path.replace(".png", ".npy")).astype(np.uint8)  # Load mask from .npy if .png not found
                    mask_img = (mask_img > 0) * 255  # Convert to binary mask

                # Convert images to numpy arrays
                rgb_array = np.array(rgb_img)
                mask_array = np.array(mask_img)

                # Create RGBA image
                if rgb_array.shape[2] == 4:
                    # If original has alpha channel, preserve it and combine with mask
                    rgba_array = rgb_array.copy()
                    rgba_array[:, :, 3] = np.minimum(rgba_array[:, :, 3], (mask_array > 127) * 255)
                else:
                    # If no alpha channel, create new RGBA array
                    rgba_array = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 4), dtype=np.uint8)
                    rgba_array[:, :, :3] = rgb_array  # Copy RGB channels
                    rgba_array[:, :, 3] = (mask_array > 127) * 255

                # Set fully transparent pixels to (0,0,0,0)
                rgba_array[mask_array <= 127] = [0, 0, 0, 0]

                # Convert to PIL Image
                result_img = Image.fromarray(rgba_array, 'RGBA')

                # --- Crop out transparent border ---
                bbox = result_img.getbbox()  # finds bounding box of non-zero alpha pixels
                if bbox:
                    result_img = result_img.crop(bbox)

                # Save
                output_path = os.path.join(masked_dir, rgb_file)
                result_img.save(output_path)
                print(f"Processed {rgb_file}")


            image_dir = masked_dir

        if self.cfg.assetlookup.clear_TRELLIS_cache_for_this_scene:
            print(f"Clearing TRELLIS cache for asset <{image_dir}>...")
            cache.clear_cache(image_dir)
            print("Cache cleared.")

        # cache lookup
        IS_FOUND_IN_CACHE = cache.is_in_cache(image_dir)
        try:
            if IS_FOUND_IN_CACHE:
                metadata_folder = cache.path_in_cache_for_metadata(image_dir)
                with open(f"{metadata_folder}/bbox.json", "r") as f:
                    meta = json.load(f)

                    image_sequence_stride = meta["image_sequence_stride"]
                    image_sequence_start = meta["image_sequence_start"]
                    image_sequence_end = meta["image_sequence_end"]
                    use_largest_mask_instead = meta["use_largest_mask_instead"]
                    
                    if image_sequence_stride != self.cfg.assetlookup.image_sequence_stride or \
                       image_sequence_start != self.cfg.assetlookup.image_sequence_start or \
                       image_sequence_end != self.cfg.assetlookup.image_sequence_end or \
                       use_largest_mask_instead != self.cfg.assetlookup.use_largest_mask_instead:
                        raise ValueError(
                            f"Image sequence settings does not match config ({self.cfg.assetlookup.num_images_to_use})."
                        )
        except:
            cache.clear_cache(image_dir)
            IS_FOUND_IN_CACHE = False

        if cache.cache_says_trellis_fails(image_dir):
            print(f"TRELLIS pipeline can't handle <{image_dir}>, falling back to OBJAVERSE...")
            return self.clip_lookup.lookup_assets(obj)

        

        

        def mark_as_infeasible(reason):
            print(f"Marking as infeasible: <{image_dir}>")

            trellisfailure_folder = cache.path_in_cache_for_trellisfailure(image_dir)
            os.makedirs(trellisfailure_folder, exist_ok=True)

            with open(f"{trellisfailure_folder}/trellis_failure.txt", "w") as f:
                f.write(
                    "If the directory <trellis_failure> exists, the creation of this asset failed. Fall back to OBJAVERSE.")
                f.write("\n")
                f.write(reason)
            return

        def obtain_and_put_in_cache():
            cache.clear_cache(image_dir)

            raw_folder = cache.path_in_cache_for_raw(image_dir)
            os.makedirs(raw_folder, exist_ok=True)
            
            uuid_str = str(uuid.uuid4())
            temp_image_dir = f"/tmp/{uuid_str}/{image_dir}".replace("//", "/")
            os.makedirs(temp_image_dir, exist_ok=True)

            if self.cfg.assetlookup.image_sequence_end == -1:
                image_sequence_end = len([f for f in os.listdir(image_dir) if f.endswith(".png")]) 
            else:
                image_sequence_end = min(self.cfg.assetlookup.image_sequence_end, len([f for f in os.listdir(image_dir) if f.endswith(".png")]) - 1)

            image_sequence_start = self.cfg.assetlookup.image_sequence_start
            if image_sequence_start > len([f for f in os.listdir(image_dir) if f.endswith(".png")]):
                image_sequence_start = len([f for f in os.listdir(image_dir) if f.endswith(".png")]) - 1


            if self.cfg.assetlookup.use_largest_mask_instead:
                # Use the largest mask in the sequence
                masks = []
                #masks = []
                for idx in range(self.cfg.assetlookup.image_sequence_start, image_sequence_end + 1, self.cfg.assetlookup.image_sequence_stride):
                    if os.path.exists(os.path.join(masks_dir, f"{idx:03d}.png")):
                        masks.append(os.path.join(masks_dir, f"{idx:03d}.png") )
                largest_mask = max(masks, key=lambda x: (np.array(Image.open(x)) > 0).sum())
                largest_mask_name = os.path.basename(largest_mask)
                #shutil.copy2(largest_mask, os.path.join(temp_image_dir, "000.png"))

            # Loop over selected indices
            NUM_IMAGES_TO_USE = 0
            for idx in range(image_sequence_start, image_sequence_end+1, self.cfg.assetlookup.image_sequence_stride):
                if self.cfg.assetlookup.use_largest_mask_instead and f"{idx:03d}.png" != largest_mask_name:
                    continue

                src = os.path.join(image_dir, f"{idx:03d}.png")
                dst = os.path.join(temp_image_dir, f"{NUM_IMAGES_TO_USE:03d}.png")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    NUM_IMAGES_TO_USE += 1
                else:
                    print(f"Warning: {src} not found")

            try:
                if NUM_IMAGES_TO_USE == 1:
                    raise Exception("Using single image mode, not multiimage mode.")
                print("USING MULTI IMAGE TRELLIS")
                self.TRELLIS_client.generate_and_download_from_multiple_images(
                    temp_image_dir,
                    target_dir=raw_folder,
                    params={
                        'multiimage_algo': 'stochastic',
                        'seed': 123,
                        'preprocess_image': str(not self.cfg.assetlookup.use_masks)
                    }
                )
            except Exception as e:
                try:
                    print("USING SINGLE IMAGE TRELLIS")
                    self.TRELLIS_client.generate_and_download_from_single_image(
                        temp_image_dir + "/000.png",
                        target_dir=raw_folder,
                        params={
                            'multiimage_algo': 'stochastic',
                            'seed': 123,
                            'preprocess_image': str(not self.cfg.assetlookup.use_masks)
                        }
                    )
                except Exception as e:
                    return mark_as_infeasible("Failed to apply TRELLIS to these images, for some reason.")

            convert_folder = cache.path_in_cache_for_convert(image_dir)
            os.makedirs(convert_folder, exist_ok=True)

            assetid = str(uuid.uuid4().hex)[:8]
            conversion_success = convert_to_ai2thor.convert(raw_folder, assetid, convert_folder)
            if not conversion_success:
                return mark_as_infeasible("Failed to convert to AI2THOR")

            from ai2thor.util.runtime_assets import load_existing_thor_asset_file
            obj = load_existing_thor_asset_file(convert_folder, f"{assetid}/{assetid}")
            from hippo.utils.spatial_utils import get_ai2thor_object_bbox
            bbox = get_ai2thor_object_bbox(obj)

            metadata_folder = cache.path_in_cache_for_metadata(image_dir)
            os.makedirs(metadata_folder, exist_ok=True)

            with open(f"{metadata_folder}/bbox.json", "w") as f:
                bbox = {k: float(v) for k, v in bbox.items()}

                bbox["image_sequence_stride"] = self.cfg.assetlookup.image_sequence_stride
                bbox["image_sequence_start"] = self.cfg.assetlookup.image_sequence_start
                bbox["image_sequence_end"] = self.cfg.assetlookup.image_sequence_end
                bbox["use_largest_mask_instead"] = self.cfg.assetlookup.use_largest_mask_instead

                json.dump(bbox, f, indent=2)

            with open(f"{metadata_folder}/uuid.txt", "w") as f:
                f.write(assetid)

        print(f"Looking up assets for <{image_dir}>...")
        if IS_FOUND_IN_CACHE:
            print("Cache hit! Not querying TRELLIS.")
        else:
            print("Cache miss! Querying TRELLIS, this will take a while...")
            obtain_and_put_in_cache()

        if cache.cache_says_trellis_fails(image_dir):
            print(f"TRELLIS pipeline can't handle <{image_dir}>, falling back to OBJAVERSE...")
            return self.clip_lookup.lookup_assets(obj)

        with open(f"{cache.path_in_cache_for_metadata(image_dir)}/bbox.json", "r") as f:
            bbox = json.load(f)
            size = bbox["x"], bbox["y"], bbox["z"]

        with open(f"{cache.path_in_cache_for_metadata(image_dir)}/uuid.txt", "r") as f:
            assetid = f.read().strip()

        sizes = np.array(size)[None]

        return obj.add_asset_info_([assetid], sizes, [1], [cache.path_in_cache_for_convert(image_dir)])
