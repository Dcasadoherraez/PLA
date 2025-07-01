import os
import glob
import json
import tqdm
import pickle
import nltk
import torch
import numpy as np
from functools import partial
import concurrent.futures as futures
from pcseg.datasets.truckscenes.truckscenes_dataset import TruckScenesDataset
from pcseg.utils import common_utils, caption_utils
import json
import multiprocessing
from pathlib import Path
import tqdm

# Global variable to hold the large data in each worker process
_view_caption_global = None

def _init_worker(view_caption_path: Path):
    """
    Initializer for each worker in the pool. Loads the large JSON file into a
    global variable, making it accessible to the worker function without
    needing to pass it repeatedly.
    """
    global _view_caption_global
    # Load data only once per process
    if _view_caption_global is None:
        with open(view_caption_path, 'r') as f:
            _view_caption_global = json.load(f)

def _process_single_item(item_tuple: tuple, static_processing_func, save_path: Path):
    """
    The function executed by each worker process for a single item.
    It calls the static processing logic and accesses the loaded view_caption
    data from the global scope.
    """
    info, idx = item_tuple
    # The core logic is called here, using the globally available caption data.
    static_processing_func(info, save_path, _view_caption_global)
    
    
class CaptionIdxProcessor(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.infos = self.dataset.data_list  # List of frame paths for TruckScenes

    def get_lidar(self, info, idx):
        """Load point cloud data for a given frame.

        Args:
            info: Path to the frame's .pth file.
            idx: Index of the frame in the dataset.

        Returns:
            data_dict: Dictionary with point cloud data.
            scene_name: Name of the scene.
            info: Updated info dict with scene_name and depth_image_size.
        """
        xyz, feats, label, inst_label, binary_label = self.dataset.load_data(idx)
        scene_name = info.parts[-3]  # Extract scene name from path, e.g., 'scene-0044384af3d8494e913fb8b14915239e-11'
        frame_id = info.stem  # Extract frame ID, e.g., '0'
        data_dict = {'points_xyz': xyz, 'scene_name': scene_name, 'frame_id': frame_id}
        info = {'scene_name': scene_name, 'frame_id': frame_id, 'depth_image_size': self.dataset.depth_image_scale}
        return data_dict, scene_name, info

    def merge_to_one_file(self, save_path, new_save_path):
        """Merge all individual frame indices into a single pickle file."""
        file_list = glob.glob(os.path.join(save_path, '*.pkl'))
        merged_idx = {}
        for fn in file_list:
            fn_name = fn.split('/')[-1].split('.')[0]
            data = pickle.load(open(fn, 'rb'))
            merged_idx[fn_name] = data
        with open(new_save_path, 'wb') as f:
            pickle.dump(merged_idx, f)

    def create_view_caption_idx(self, num_workers=16):
        """Generate view caption indices for all frames.

        Args:
            num_workers: Number of worker threads (unused in single-threaded version).
        """
        save_path = self.dataset.root_path / 'caption_idx' / 'truckscenes_view_matching_idx'
        save_path.mkdir(parents=True, exist_ok=True)

        create_view_caption_idx_single_scene = partial(
            self.create_view_caption_idx_single,
            save_path=save_path
        )

        for idx, info in tqdm.tqdm(enumerate(self.infos), total=len(self.infos)):
            create_view_caption_idx_single_scene((info, idx))

        if dataset_cfg.get('MERGE_IDX', False):
            new_save_path = str(save_path) + '.pkl'
            self.merge_to_one_file(save_path, new_save_path)


    # Copying and pasting results from point_camera_correspondences 
    def create_view_caption_idx_single(self, info_with_idx, save_path):
        """Generate view caption indices for a single frame.

        Args:
            info_with_idx: Tuple of (frame path, index).
            save_path: Directory to save the indices.
        """
        scene_name, idx = info_with_idx
        caption_corr_idx_path = Path(args.view_caption_corr_idx_path) / scene_name / 'view_caption_corr_idx.pkl'
        view_caption_corr_idx = pickle.load(open(caption_corr_idx_path, 'rb'))

        view_point_correspondences = {}
        for cam_with_idx in view_caption_corr_idx.keys():
            view_point_correspondences[cam_with_idx] =  view_caption_corr_idx[cam_with_idx][1]
            
        scene_caption_save_path = save_path / f"{scene_name}.pkl"
        
        with open(scene_caption_save_path, 'wb') as f:
            pickle.dump(view_point_correspondences, f)



    # def create_view_caption_idx_single(self, info_with_idx, save_path):
    #     """Generate view caption indices for a single frame.

    #     Args:
    #         info_with_idx: Tuple of (frame path, index).
    #         save_path: Directory to save the indices.
    #     """
    #     info, idx = info_with_idx
    #     scene_name = info.parts[-3]  # Extract scene name from path, e.g., 'scene-0044384af3d8494e913fb8b14915239e-11'
    #     frame_id = info.stem  # Extract frame ID, e.g., '0'
    #     # data_dict, scene_name, info = self.get_lidar(info, idx)
    #     # data_dict = self.dataset.get_image(info, data_dict)
    #     # scene_caption_idx = data_dict['point_img_idx']
    #     # for key, values in scene_caption_idx.items():
    #     #     scene_caption_idx[key] = torch.from_numpy(values).int()
    #     caption_corr_idx_path = Path(args.view_caption_corr_idx_path) / scene_name / 'view_caption_corr_idx.pkl'
    #     view_caption_corr_idx = pickle.load(open(caption_corr_idx_path, 'rb'))
        
    #     view_point_correspondences = {}
    #     for cam in ['CAMERA_LEFT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK', 'CAMERA_RIGHT_FRONT']:
    #         key = f"{cam}_{frame_id}"
    #         if key in view_caption_corr_idx:
    #             view_point_correspondences[cam] =  view_caption_corr_idx[key][1]
    #         else:
    #             view_point_correspondences[cam] = []
        
    #     scene_caption_save_path = save_path / f"{scene_name}_{frame_id}.pkl"
        
    #     with open(scene_caption_save_path, 'wb') as f:
    #         pickle.dump(view_point_correspondences, f)


    # def create_entity_caption_idx(self, num_workers=16):
    #     """Generate view caption indices for all frames.

    #     Args:
    #         num_workers: Number of worker threads (unused in single-threaded version).
    #     """
    #     save_path = self.dataset.root_path / 'caption_idx' / 'truckscenes_entity_matching_idx'
    #     save_path.mkdir(parents=True, exist_ok=True)
    #     view_caption = json.load(open(args.view_caption_path, 'r'))

    #     create_entity_caption_idx_single_scene = partial(
    #         self.create_entity_caption_idx_single,
    #         save_path=save_path,
    #         view_caption=view_caption
    #     )


    #     for idx, info in tqdm.tqdm(enumerate(self.infos), total=len(self.infos)):
    #         create_entity_caption_idx_single_scene((info, idx))

    #     if dataset_cfg.get('MERGE_IDX', False):
    #         new_save_path = str(save_path) + '.pkl'
    #         self.merge_to_one_file(save_path, new_save_path)

    def create_entity_caption_idx(self, num_workers: int = 32):
        """
        Generate view caption indices for all frames in parallel.
        """
        save_path = self.dataset.root_path / 'caption_idx' / 'truckscenes_entity_matching_idx'
        save_path.mkdir(parents=True, exist_ok=True)
        view_caption_path = Path(args.view_caption_path) # Ensure this is a Path object

        # Use functools.partial to "freeze" arguments for the worker function.
        # This creates a callable that only needs the 'item_tuple'.
        worker_func = partial(
            _process_single_item,
            static_processing_func=self.create_entity_caption_idx_single,
            save_path=save_path
        )

        tasks = [(info, idx) for idx, info in enumerate(self.infos)]

        with multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=(view_caption_path,)) as pool:
            print(f"Starting parallel processing with {num_workers} workers...")
            results = list(tqdm.tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks)))
            print("Parallel processing finished.")

        if dataset_cfg.get('MERGE_IDX', False):
            print("Merging index files...")
            new_save_path = str(save_path) + '.pkl'
            self.merge_to_one_file(save_path, new_save_path)
            print(f"Files merged into {new_save_path}")


    def create_entity_caption_idx_single(self, scene_name, save_path, view_caption, num_workers=16):
        """Generate entity caption indices for all frames.

        Args:
            num_workers: Number of worker threads (unused in single-threaded version).
        """
        scene_name = scene_name
        caption_save_path = save_path / f"{scene_name}.pkl"
        
        if os.path.exists(caption_save_path):
            return
        
        caption_corr_idx_path = Path(args.view_caption_corr_idx_path) / scene_name / 'view_caption_corr_idx.pkl'
        view_caption_corr_idx = pickle.load(open(caption_corr_idx_path, 'rb'))
        view_entity_caption = self.extract_entity(view_caption[scene_name])
        captions_entity_corr_idx = self.get_entity_caption_corr_idx(
            view_entity_caption, view_caption_corr_idx
        )
        with open(caption_save_path, 'wb') as f:
            pickle.dump(captions_entity_corr_idx, f)
        print("Saving to:", caption_save_path)



    # def create_entity_caption_idx_single(self, info, save_path, view_caption, num_workers=16):
    #     """Generate entity caption indices for all frames.

    #     Args:
    #         num_workers: Number of worker threads (unused in single-threaded version).
    #     """
    #     scene_name = info.parts[-3]  
    #     frame_id = info.stem  
    #     caption_save_path = save_path / f"{scene_name}_{frame_id}.pkl"
    #     if os.path.exists(caption_save_path):
    #         return

        
    #     caption_corr_idx_path = Path(args.view_caption_corr_idx_path) / scene_name / 'view_caption_corr_idx.pkl'
    #     view_caption_corr_idx = pickle.load(open(caption_corr_idx_path, 'rb'))
    #     view_entity_caption = self.extract_entity(view_caption[scene_name])
        
    #     captions_entity_corr_idx = self.get_entity_caption_corr_idx(
    #         view_entity_caption, view_caption_corr_idx
    #     )
        
    #     with open(caption_save_path, 'wb') as f:
    #         pickle.dump(captions_entity_corr_idx, f)
    #     print("Saving to:", caption_save_path)


    @staticmethod
    def extract_entity(view_caption):
        """Extract noun entities from captions.

        Args:
            view_caption: Dictionary of captions per scene and frame.

        Returns:
            caption_entity: Dictionary of extracted entities.
        """
        caption_entity = {}
        for frame in view_caption:
            caption = view_caption[frame]
            tokens = nltk.word_tokenize(caption)
            tagged = nltk.pos_tag(tokens)
            entities = [e[0] for e in tagged if e[1].startswith('NN')]
            caption_entity[frame] = ' '.join(entities)
        return caption_entity
    
    # @staticmethod
    # def extract_entity(view_caption):
    #     """Extract noun entities from captions.

    #     Args:
    #         view_caption: Dictionary of captions per scene and frame.

    #     Returns:
    #         caption_entity: Dictionary of extracted entities.
    #     """
    #     caption_entity = {}
    #     for scene in view_caption:
    #         caption_entity[scene] = {}
    #         for frame in view_caption[scene]:
    #             caption = view_caption[scene][frame]
    #             tokens = nltk.word_tokenize(caption)
    #             tagged = nltk.pos_tag(tokens)
    #             entities = [e[0] for e in tagged if e[1].startswith('NN')]
    #             caption_entity[scene][frame] = ' '.join(entities)
    #     return caption_entity

    @staticmethod
    def compute_intersect_and_diff(c1, c2):
        """Compute set differences and intersection.

        Args:
            c1, c2: Lists or sets to compare.

        Returns:
            old, new, intersect: Differences and intersection.
        """
        old = set(c1) - set(c2)
        new = set(c2) - set(c1)
        intersect = set(c1) & set(c2)
        return old, new, intersect

    def get_entity_caption_corr_idx(self, view_entity_caption, view_caption_corr_idx):
        """Generate entity caption indices for a single scene."""

        entity_caption_corr_idx = {}
        minpoint = 100
        ratio = args.entity_overlap_thr
        frame_keys = list(view_caption_corr_idx.keys())
        entity_num = 0

        # Preprocess entity captions once: remove 'room' and store as set
        processed_captions = {
            k: set(word for word in v.split() if word != 'room')
            for k, v in view_entity_caption.items()
        }

        # Preprocess view indices once
        processed_indices = {
            k: np.array(view_caption_corr_idx[k][1])
            for k in frame_keys
        }
        print(frame_keys)
        print( processed_captions)
        print(processed_indices)

        for ii in range(len(frame_keys) - 1):
            frame1 = frame_keys[ii]
            idx1 = processed_indices[frame1]
            c1 = processed_captions[frame1]

            for jj in range(ii + 1, len(frame_keys)):
                frame2 = frame_keys[jj]
                idx2 = processed_indices[frame2]
                c2 = processed_captions[frame2]
                
                # Compute index intersection/differences
                old, new, intersection = self.compute_intersect_and_diff(idx1, idx2)
                print(c1, c2)
                old_c, new_c, intersection_c = self.compute_intersect_and_diff(c1, c2)

                len_idx1 = len(idx1)
                len_idx2 = len(idx2)
                len_inter = len(intersection)

                # Only add valid entries
                if len_inter > minpoint and intersection_c and len_inter / min(len_idx1, len_idx2) <= ratio:
                    entity_caption_corr_idx[f'entity_{entity_num}'] = torch.IntTensor(list(intersection))
                    entity_num += 1

                if len(old) > minpoint and old_c and len(old) / len_idx1 <= ratio:
                    entity_caption_corr_idx[f'entity_{entity_num}'] = torch.IntTensor(list(old))
                    entity_num += 1

                if len(new) > minpoint and new_c and len(new) / len_idx2 <= ratio:
                    entity_caption_corr_idx[f'entity_{entity_num}'] = torch.IntTensor(list(new))
                    entity_num += 1
                print("Entity num", entity_num)

        return entity_caption_corr_idx

    # def get_entity_caption_corr_idx(self, view_entity_caption, view_caption_corr_idx):
    #     """Generate entity caption indices for a single scene.

    #     Args:
    #         view_entity_caption: Extracted entities from captions (frame_id -> caption).
    #         view_caption_corr_idx: View caption indices (frame_id -> indices).

    #     Returns:
    #         entity_caption_corr_idx: Indices for entity captions (entity_id -> indices).
    #     """
    #     entity_caption_corr_idx = {}
    #     minpoint = 100
    #     ratio = args.entity_overlap_thr
        
    #     # No need to loop over scenes; view_caption_corr_idx is for one scene
    #     frame_keys = list(view_caption_corr_idx.keys())  # Keys are frame IDs
    #     entity_num = 0
        
    #     # Compare pairs of frames within the scene
    #     for ii in range(len(frame_keys) - 1):
    #         for jj in range(ii + 1, len(frame_keys)):
    #             frame1 = frame_keys[ii]
    #             frame2 = frame_keys[jj]
    #             print(frame1, frame2)
    #             # print(view_caption_corr_idx[frame1][1], view_caption_corr_idx[frame2][1])
    #             idx1 = np.array(view_caption_corr_idx[frame1][1])
    #             idx2 = np.array(view_caption_corr_idx[frame2][1])
    #             c = view_entity_caption[frame1].split(' ')
    #             c2 = view_entity_caption[frame2].split(' ')
    #             if 'room' in c:
    #                 c.remove('room')
    #             if 'room' in c2:
    #                 c2.remove('room')
    #             old, new, intersection = self.compute_intersect_and_diff(idx1, idx2)
    #             old_c, new_c, intersection_c = self.compute_intersect_and_diff(c, c2)
                
    #             # Check intersection condition
    #             if (len(intersection) > minpoint and 
    #                 len(intersection_c) > 0 and 
    #                 len(intersection) / min(len(idx1), len(idx2)) <= ratio):
    #                 entity_caption_corr_idx[f'entity_{entity_num}'] = torch.IntTensor(list(intersection))
    #                 entity_num += 1
                
    #             # Check 'old' difference condition
    #             if (len(old) > minpoint and 
    #                 len(old_c) > 0 and 
    #                 len(old) / len(idx1) <= ratio):
    #                 entity_caption_corr_idx[f'entity_{entity_num}'] = torch.IntTensor(list(old))
    #                 entity_num += 1
                
    #             # Check 'new' difference condition
    #             if (len(new) > minpoint and 
    #                 len(new_c) > 0 and 
    #                 len(new) / len(idx2) <= ratio):
    #                 entity_caption_corr_idx[f'entity_{entity_num}'] = torch.IntTensor(list(new))
    #                 entity_num += 1
        
    #     return entity_caption_corr_idx
    
    # def get_entity_caption_corr_idx(self, view_entity_caption, view_caption_corr_idx):
    #     """Generate entity caption indices.

    #     Args:
    #         view_entity_caption: Extracted entities from captions.
    #         view_caption_corr_idx: View caption indices.

    #     Returns:
    #         entity_caption_corr_idx: Indices for entity captions.
    #     """
    #     entity_caption_corr_idx = {}
    #     minpoint = 100
    #     ratio = args.entity_overlap_thr
        
    #     print(view_caption_corr_idx.keys())
    #     exit(-1)

    #     for scene in tqdm.tqdm(view_caption_corr_idx):
    #         frame_idx = view_caption_corr_idx[scene]
    #         entity_caption_corr_idx[scene] = {}
    #         entity_num = 0
    #         frame_keys = list(frame_idx.keys())
    #         for ii in range(len(frame_keys) - 1):
    #             for jj in range(ii + 1, len(frame_keys)):
    #                 idx1 = frame_idx[frame_keys[ii]].cpu().numpy()
    #                 idx2 = frame_idx[frame_keys[jj]].cpu().numpy()
    #                 c = view_entity_caption[scene][frame_keys[ii]].split(' ')
    #                 c2 = view_entity_caption[scene][frame_keys[jj]].split(' ')
    #                 if 'room' in c:
    #                     c.remove('room')
    #                 if 'room' in c2:
    #                     c2.remove('room')
    #                 old, new, intersection = self.compute_intersect_and_diff(idx1, idx2)
    #                 old_c, new_c, intersection_c = self.compute_intersect_and_diff(c, c2)
    #                 if len(intersection) > minpoint and len(intersection_c) > 0 and \
    #                     len(intersection) / min(len(idx1), len(idx2)) <= ratio:
    #                     entity_caption_corr_idx[scene][f'entity_{entity_num}'] = torch.IntTensor(list(intersection))
    #                     entity_num += 1
    #                 if len(old) > minpoint and len(old_c) > 0 and len(old) / len(idx1) <= ratio:
    #                     entity_caption_corr_idx[scene][f'entity_{entity_num}'] = torch.IntTensor(list(old))
    #                     entity_num += 1
    #                 if len(new) > minpoint and len(new_c) > 0 and len(new) / len(idx2) <= ratio:
    #                     entity_caption_corr_idx[scene][f'entity_{entity_num}'] = torch.IntTensor(list(new))
    #                     entity_num += 1
    #     return entity_caption_corr_idx


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default='truckscenes', help='specify the dataset')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--model_cfg', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_view_caption_idx', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--workers', type=int, default=8, help='')
    parser.add_argument('--save_path', type=str, help='')
    parser.add_argument('--filter_by_image_size', action='store_true', default=False, help='')
    parser.add_argument('--filter_empty_caption', action='store_true', default=False, help='')
    parser.add_argument('--entity_overlap_thr', default=0.3, help='threshold ratio for filtering out large entity-level point set')
    parser.add_argument('--view_caption_path', default=None, help='path for view-level caption')
    parser.add_argument('--view_caption_corr_idx_path', default=None, help='path for view-level caption corresponding index')

    global args
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    dataset_cfg.VERSION = args.version

    cfg = EasyDict(yaml.safe_load(open(args.model_cfg)))

    if args.dataset == 'truckscenes':
        dataset = TruckScenesDataset(
            dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
            root_path=ROOT_DIR / 'data' / 'truckscenes',
            training=False, logger=common_utils.create_logger()
        )
    else:
        raise NotImplementedError

    processor = CaptionIdxProcessor(dataset)
    if args.func == 'create_view_caption_idx':
        processor.create_view_caption_idx(args.workers)
    elif args.func == 'create_entity_caption_idx':
        processor.create_entity_caption_idx(args.workers)
    else:
        raise NotImplementedError