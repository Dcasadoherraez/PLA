import os
import numpy as np
import torch
import glob
import pickle
from pathlib import Path
import numpy as np
import cv2
import os
import SharedArray as SA
from ..indoor_dataset import IndoorDataset
from ...utils.common_utils import sa_create
import matplotlib.pyplot as plt

import uuid
IGNORE_CLASS_INDEX = 12
TRUCKSCENES_LABELS_TO_IDX = {
    "animal": 0,
    "human.pedestrian.adult": 7,
    "human.pedestrian.child": 7,
    "human.pedestrian.construction_worker": 7,
    "human.pedestrian.personal_mobility": 7,
    "human.pedestrian.police_officer": 7,
    "human.pedestrian.stroller": IGNORE_CLASS_INDEX,
    "human.pedestrian.wheelchair": IGNORE_CLASS_INDEX,
    "movable_object.barrier": 1,
    "movable_object.debris": IGNORE_CLASS_INDEX,
    "movable_object.pushable_pullable": IGNORE_CLASS_INDEX,
    "movable_object.trafficcone": 8,
    "static_object.bicycle_rack": IGNORE_CLASS_INDEX,
    "static_object.traffic_sign": 9,
    "vehicle.bicycle": 2,
    "vehicle.bus.bendy": 3,
    "vehicle.bus.rigid": 3,
    "vehicle.car": 4,
    "vehicle.construction": 6,
    "vehicle.emergency.ambulance": IGNORE_CLASS_INDEX,
    "vehicle.emergency.police": IGNORE_CLASS_INDEX,
    "vehicle.motorcycle": 5,
    "vehicle.trailer": 10,
    "vehicle.truck": 11,
    "vehicle.train": IGNORE_CLASS_INDEX,
    "vehicle.other": IGNORE_CLASS_INDEX,
    "vehicle.ego_trailer": IGNORE_CLASS_INDEX,
    "unlabeled": IGNORE_CLASS_INDEX
}

# IGNORE_CLASS_INDEX = 3
# TRUCKSCENES_LABELS_TO_IDX = {
#     "animal": IGNORE_CLASS_INDEX,
#     "human.pedestrian.adult": IGNORE_CLASS_INDEX,
#     "human.pedestrian.child": IGNORE_CLASS_INDEX,
#     "human.pedestrian.construction_worker": IGNORE_CLASS_INDEX,
#     "human.pedestrian.personal_mobility": IGNORE_CLASS_INDEX,
#     "human.pedestrian.police_officer": IGNORE_CLASS_INDEX,
#     "human.pedestrian.stroller": IGNORE_CLASS_INDEX,
#     "human.pedestrian.wheelchair": IGNORE_CLASS_INDEX,
#     "movable_object.barrier": IGNORE_CLASS_INDEX,
#     "movable_object.debris": IGNORE_CLASS_INDEX,
#     "movable_object.pushable_pullable": IGNORE_CLASS_INDEX,
#     "movable_object.trafficcone": IGNORE_CLASS_INDEX,
#     "static_object.bicycle_rack": IGNORE_CLASS_INDEX,
#     "static_object.traffic_sign": IGNORE_CLASS_INDEX,
#     "vehicle.bicycle": IGNORE_CLASS_INDEX,
#     "vehicle.bus.bendy": IGNORE_CLASS_INDEX,
#     "vehicle.bus.rigid": IGNORE_CLASS_INDEX,
#     "vehicle.car": 0,
#     "vehicle.construction": IGNORE_CLASS_INDEX,
#     "vehicle.emergency.ambulance": IGNORE_CLASS_INDEX,
#     "vehicle.emergency.police": IGNORE_CLASS_INDEX,
#     "vehicle.motorcycle": IGNORE_CLASS_INDEX,
#     "vehicle.trailer": 1,
#     "vehicle.truck": 2,
#     "vehicle.train": IGNORE_CLASS_INDEX,
#     "vehicle.other": IGNORE_CLASS_INDEX,
#     "vehicle.ego_trailer": IGNORE_CLASS_INDEX,
#     "unlabeled": IGNORE_CLASS_INDEX
# }


        
class TruckScenesDataset(IndoorDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None):
        """Initialize the TruckScenesDataset.

        Args:
            dataset_cfg: Configuration dictionary.
            class_names: List of class names.
            training: Boolean indicating training or evaluation mode.
            root_path: Path to the dataset root directory.
            logger: Logger instance for logging information.
        """
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        # Load data list from split file (e.g., truckscenes_train.txt)
        mode = dataset_cfg.DATA_SPLIT[self.mode]
        print("TruckScenesDataset in", mode, "mode")
        # split_file = self.root_path / f"truckscenes_{mode}_single_scene.txt"
        split_file = self.root_path / f"truckscenes_no_dark_no_highway_{mode}.txt"
        with open(split_file, 'r') as f:
            self.data_list = sorted([line.strip() for line in f.readlines()])
        
        self.all_data_list = [self.root_path / "trainval" / seq / "labelled_map.pth" for seq in self.data_list]
        
        # self.data_list = [self.data_list[0]]
        
        ## Shared memory 
        self.run_id = os.environ.get("CUDA_VISIBLE_DEVICES", str(uuid.uuid4()))
        self.put_data_to_shm()

        # Load caption correspondence if training and captions are enabled
        if self.training and hasattr(self, 'caption_cfg'):# and self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
            self.scene_image_corr_infos, self.scene_image_corr_entity_infos = self.include_caption_infos()

        # Image loading parameters
        self.load_image = self.dataset_cfg.get('LOAD_IMAGE', False)
        self.depth_image_scale = self.dataset_cfg.get('DEPTH_IMAGE_SCALE', (480, 640))  # Default image size

        self.logger.info(f"Totally {len(self.data_list)} samples in {mode} set.")


    def put_data_to_shm(self):
        
        for item in self.all_data_list:
            scene_name = item.parts[-2]

            # shm_key = f"truckscenes_{scene_name}_{frame_id}"
            shm_key = f"{self.run_id}_truckscenes_{scene_name}"

            shm_xyz = f"/dev/shm/{shm_key}_xyz"
            shm_feats = f"/dev/shm/{shm_key}_feats"
            shm_label = f"/dev/shm/{shm_key}_label"
            shm_inst_label = f"/dev/shm/{shm_key}_inst_label"

            # print(f"Checking SHM for {scene_name}, frame {frame_id}")
            # print(f"Exists: xyz={os.path.exists(shm_xyz)}, feats={os.path.exists(shm_feats)}, label={os.path.exists(shm_label)}, inst_label={os.path.exists(shm_inst_label)}")
            # print("self.cache =", self.cache)

            if self.cache and not (os.path.exists(shm_xyz) and os.path.exists(shm_feats)
                                and os.path.exists(shm_label) and os.path.exists(shm_inst_label)):
                xyz, feats, label, inst_label = torch.load(item)
                xyz = xyz[:, :3]

                if np.isscalar(feats) and feats == 0:
                    feats = np.zeros_like(xyz)

                try:
                    sa_create(f"shm://{shm_key}_xyz", xyz)
                    sa_create(f"shm://{shm_key}_feats", feats)
                    sa_create(f"shm://{shm_key}_label", label)
                    sa_create(f"shm://{shm_key}_inst_label", inst_label)
                except FileExistsError:
                    continue

         
    def transform_point_cloud(self, scene_name, xyz, frame_id_1=0, frame_id_2=39):
        # Load transformation matrices for frame_id 0 and frame_id 39
        T_w_to_ego_1 = self.load_transformation_matrix(scene_name, frame_id_1)
        T_w_to_ego_2 = self.load_transformation_matrix(scene_name, frame_id_2)

        # Calculate the middle point between the two frames
        T_w_to_ego_middle = self.calculate_middle_transformation(T_w_to_ego_1, T_w_to_ego_2)

        # Transform the point cloud
        xyz_transformed = self.apply_transformation(xyz, T_w_to_ego_middle)

        return xyz_transformed

    def load_transformation_matrix(self, scene_name, frame_id):
        ego_pose_path = self.root_path / "trainval" / scene_name / "poses" / "EGO" / f"{frame_id}.txt"
        T_w_to_ego = np.loadtxt(ego_pose_path)
        return T_w_to_ego

    def calculate_middle_transformation(self, T1, T2):
        # Calculate the middle transformation matrix
        # This is a simplified approach; you might need a more sophisticated method depending on your use case
        translation_middle = (T1[:3, 3] + T2[:3, 3]) / 2
        rotation_middle = (T1[:3, :3] + T2[:3, :3]) / 2

        T_middle = np.eye(4)
        T_middle[:3, :3] = rotation_middle
        T_middle[:3, 3] = translation_middle

        return T_middle

    def apply_transformation(self, xyz, T_w_to_ego):
        # Transform xyz (N x 3) to homogeneous coordinates (N x 4)
        homo_coords = np.hstack((xyz[:, :3], np.ones((xyz.shape[0], 1), dtype=np.float64)))  # (N, 4)

        # Apply the transformation
        xyz_transformed = (np.linalg.inv(T_w_to_ego) @ homo_coords.T).T[:, :3]  # (N, 3)
        xyz_transformed = np.ascontiguousarray(xyz_transformed[:, :3])

        return xyz_transformed

       
    def load_data(self, index):
        """Load data for a specific frame from file or shared memory.

        Args:
            index: Index of the data item.

        Returns:
            xyz: 3D point coordinates.
            feats: Point features (zeros in TruckScenes).
            label: Semantic labels (mapped to integers).
            inst_label: Instance labels.
        """
        scene_name = self.data_list[index]
        scene_map_path = self.all_data_list[index]
        shm_key = f"{self.run_id}_truckscenes_{scene_name}"
        
        if self.cache:
            xyz = SA.attach(f"shm://{shm_key}_xyz").copy()
            feats = SA.attach(f"shm://{shm_key}_feats").copy()
            label = SA.attach(f"shm://{shm_key}_label").copy()
            inst_label = SA.attach(f"shm://{shm_key}_inst_label").copy()
        else:
            xyz, feats, label, inst_label = torch.load(scene_map_path)
            xyz = xyz[:, :3] # Select only xyz

            if np.isscalar(feats) and feats == 0:
                feats = np.zeros_like(xyz)
                
        # Map string labels to integers
        # if isinstance(label[0], str):
        label = np.array([TRUCKSCENES_LABELS_TO_IDX.get(l, IGNORE_CLASS_INDEX) for l in label], dtype=np.int64)
        inst_label = np.array([l if l is not None else -1 for l in label ], dtype=np.int64)
        
        if hasattr(self, 'base_class_mapper'):
            binary_label = self.binary_class_mapper[label.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(label)
        if self.class_mode == 'base':
            label = self.base_class_mapper[label.astype(np.int64)]
        elif self.class_mode == 'novel':
            label = self.novel_class_mapper[label.astype(np.int64)]
        elif self.class_mode == 'all' and hasattr(self, 'ignore_class_idx'):
            label = self.valid_class_mapper[label.astype(np.int64)]
        # inst_label_all[label_all == self.ignore_label] = self.ignore_label

        # Bring the points to zero
        xyz_transformed = self.transform_point_cloud(scene_name, xyz)
        # Transform point cloud to ego
        # ego_pose_path = self.root_path / "trainval" / scene_name / "poses" / "EGO" / f"{frame_id}.txt"
        # T_w_to_ego = np.loadtxt(ego_pose_path)

        # # Transform xyz (N x 3) to homogeneous coordinates (N x 4)
        # homo_coords = np.hstack((xyz[:, :3], np.ones((xyz.shape[0], 1), dtype=np.float64)))  # (N, 4)
        # xyz = (np.linalg.inv(T_w_to_ego) @ homo_coords.T).T[:, :3]  # (N, 3)
        # xyz = np.ascontiguousarray(xyz[:, :3])
        # plt.figure(figsize=(20,20))
        # plt.scatter(xyz_transformed[:, 0], xyz_transformed[:, 1], s=2)
        # plt.gca().set_aspect('equal')
        # plt.savefig("/home/daniel/spatial_understanding/benchmarks/PLA/.vscode/deleteme/xyz_transformed.png")
        
        # plt.figure(figsize=(20,20))
        # plt.scatter(xyz[:, 0], xyz[:, 1], s=2)
        # plt.gca().set_aspect('equal')
        # plt.savefig("/home/daniel/spatial_understanding/benchmarks/PLA/.vscode/deleteme/xyz.png")
        # exit(-1)
        # print("AAAAAAAAAAAAAAAAA", fn)
        # print(np.unique(label))
        # exit(-1)
        return xyz_transformed, feats, label, inst_label, binary_label


    def __len__(self):
        return len(self.data_list) * (self.repeat if self.training else 1)


    def __getitem__(self, item):
        """Get a data item for training or evaluation.

        Args:
            item: Index of the item (with repeat handling).

        Returns:
            data_dict: Dictionary containing points, features, labels, and optional image correspondences.
        """
        index = item % len(self.data_list)
        xyz, feats, label, inst_label, binary_label = self.load_data(index)

        # Set RGB to zeros since no color is provided
        rgb = np.zeros_like(xyz)
        pc_count = xyz.shape[0]
        origin_idx = np.arange(pc_count, dtype=np.int64)

        # Extract scene and frame info
        scene_name = self.data_list[index]
        # fn = self.data_list[index]
        # scene_name = fn.parts[-3]
        # frame_id = fn.stem
        # scene_frame = f"{scene_name}_{frame_id}"

        # Handle captions if enabled
        caption_data = None
        if self.training and hasattr(self, 'caption_cfg'): # and self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
            image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name_from_memory(scene_name, index)
            caption_data = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)
            
        data_dict = {
            'points_xyz': xyz,
            'rgb': rgb,
            'labels': label,
            'inst_label': inst_label,
            'origin_idx': origin_idx,
            'pc_count': pc_count,
            'caption_data': caption_data,
            'ids': index,
            'scene_name': scene_name,
            'binary_labels': binary_label
        }

        # Load and project images if enabled
        if self.load_image:
            raise NotImplementedError("ERROR load_image NOT IMPLEMENTED")
            info = {'scene_name': scene_name, 'frame_id': frame_id, 'depth_image_size': self.depth_image_scale}
            data_dict = self.get_image(info, data_dict)

        # Apply augmentations and processing
        # if self.training:
        #     data_dict = self.augmentor.forward(data_dict)
        #     if not data_dict['valid']:
        #         return self.__getitem__(np.random.randint(self.__len__()))
        # else:
        #     xyz_voxel_scale = xyz * self.voxel_scale
        #     xyz_voxel_scale -= xyz_voxel_scale.min(0)
        #     data_dict['points_xyz_voxel_scale'] = xyz_voxel_scale
        #     data_dict['points'] = xyz
            
        # Consistently create scaled and shifted point coordinates for both training and evaluation
        xyz_voxel_scale = xyz * self.voxel_scale
        xyz_voxel_scale -= xyz_voxel_scale.min(0)
        data_dict['points_xyz_voxel_scale'] = xyz_voxel_scale
        data_dict['points'] = xyz

        # Apply augmentations and processing
        if self.training:
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return self.__getitem__(np.random.randint(self.__len__()))
        
    

        # Prepare features for voxelization
        if self.dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            data_dict['feats'] = data_dict['rgb']
        if self.dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            data_dict['feats'] = data_dict['points_xyz'] if 'feats' not in data_dict else np.concatenate(
                (data_dict['feats'], data_dict['points_xyz']), axis=1
            )

        data_dict = self.data_processor.forward(data_dict)
        
        return data_dict

    def get_image(self, info, data_dict):
        """Project 3D points to 2D images and store correspondence indices.

        Args:
            info: Dictionary with scene_name, frame_id, and depth_image_size.
            data_dict: Data dictionary to update with image correspondences.

        Returns:
            Updated data_dict with point-to-image mappings.
        """
        scene_name = info['scene_name']
        frame_id = info['frame_id']
        depth_image_size = info['depth_image_size']
        cameras = ['CAMERA_LEFT_BACK', 'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_BACK', 'CAMERA_RIGHT_FRONT']

        data_dict['point_img_1d'] = {}
        data_dict['point_img'] = {}
        data_dict['point_img_idx'] = {}
        data_dict['image_shape'] = {}
        points_xyz = data_dict['points_xyz']

        for camera in cameras:
            image_name = f"{camera}_{frame_id}"
            pose_path = self.root_path / "trainval" / scene_name / "poses" / camera / f"{frame_id}.txt"
            intrinsic_path = self.root_path / "trainval" / scene_name / "intrinsics" / camera / f"{frame_id}.txt"
            image_path = self.root_path / "trainval" / scene_name / "images" / camera / f"{frame_id}.png"
            
            if not (pose_path.exists() and intrinsic_path.exists()):
                continue

            point_idx, image_idx_1d, image_idx, image_shape = self.project_point_to_image(
                points_xyz, pose_path, intrinsic_path, image_path, depth_image_size
            )
            data_dict['point_img_1d'][image_name.lower()] = image_idx_1d
            data_dict['point_img'][image_name.lower()] = image_idx
            data_dict['point_img_idx'][image_name.lower()] = point_idx
            data_dict['image_shape'][image_name.lower()] = image_shape

        data_dict['depth_image_size'] = depth_image_size
        return data_dict

    @staticmethod
    def project_point_to_image(points_world, pose_path, intrinsic_path, image_path, image_size):
        """Project 3D points to 2D image coordinates and overlay them on the image.

        Args:
            points_world: 3D points (N, 3).
            pose_path: Path to camera pose file.
            intrinsic_path: Path to camera intrinsic file.
            image_path: Path to image file.
            image_size: Tuple of (height, width).

        Returns:
            point_valid_idx: Indices of valid points.
            point2image_coords_1d: 1D image coordinates.
            point2image_coords_2d: 2D image coordinates (u, v).
            image_size: Image dimensions.
        """
        pose = np.loadtxt(pose_path)
        intrinsic = np.loadtxt(intrinsic_path)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        points_homo = np.hstack((points_world, np.ones((points_world.shape[0], 1))))
        points_cam = (np.linalg.inv(pose) @ points_homo.T).T  # World to camera

        u = (points_cam[:, 0] * fx) / points_cam[:, 2] + cx
        v = (points_cam[:, 1] * fy) / points_cam[:, 2] + cy
        d = points_cam[:, 2]

        mask = (d > 0) & (u >= 0) & (u < image_size[1]) & (v >= 0) & (v < image_size[0])
        point_valid_idx = np.where(mask)[0]
        u_valid = u[point_valid_idx].astype(np.int32)
        v_valid = v[point_valid_idx].astype(np.int32)
        d_valid = d[point_valid_idx]

        point2image_coords_1d = v_valid * image_size[1] + u_valid
        point2image_coords_2d = np.stack([u_valid, v_valid], axis=1)

        # Load image
        # image = cv2.imread(image_path)
        # if image is None:
        #     raise FileNotFoundError(f"Image not found at {image_path}")

        # # Normalize depth and apply colormap
        # max_depth = np.percentile(d_valid, 95)
        # min_depth = np.percentile(d_valid, 5)
        # norm_depths = np.clip((d_valid - min_depth) / (max_depth - min_depth), 0, 1)

        # for i, (x, y) in enumerate(zip(u_valid, v_valid)):
        #     color_rgb = plt.cm.viridis(norm_depths[i])[:3]  # Get RGB [0,1]
        #     color_bgr = tuple(int(255 * c) for c in color_rgb[::-1])  # Convert to BGR [0,255]
        #     cv2.circle(image, (int(x), int(y)), radius=2, color=color_bgr, thickness=-1)

        # # Save image
        # save_path = '/home/daniel/spatial_understanding/benchmarks/PLA/deleteme/projected.png'  
        # cv2.imwrite(save_path, image)
        # print(f"Projected image saved to {save_path}")

        return point_valid_idx, point2image_coords_1d, point2image_coords_2d, image_size

    def include_caption_infos(self):
        """Load caption correspondence indices per scene."""
        
        scene_image_corr_infos = None
        if self.caption_cfg.get('VIEW', None) and self.caption_cfg.VIEW.ENABLED:
            scene_image_corr_infos = {}
            for scene_name in self.data_list:
                
                if scene_name not in scene_image_corr_infos:
                    scene_image_corr_infos[scene_name] = {}
                
                corr_idx_path = self.root_path / "caption_idx" / "truckscenes_view_matching_idx" / f"{scene_name}.pkl"
                if corr_idx_path.exists():
                    with open(corr_idx_path, 'rb') as f:
                        pkl = pickle.load(f)
                        for camera_view, i in pkl.items():
                           scene_image_corr_infos[scene_name][camera_view.upper()] = torch.tensor(i)

        scene_image_corr_entity_infos = None
        if self.caption_cfg.get('ENTITY', None) and self.caption_cfg.ENTITY.ENABLED:
            scene_image_corr_entity_infos = {}
            for scene_name in self.data_list:
                
                if scene_name not in scene_image_corr_entity_infos:
                    scene_image_corr_entity_infos[scene_name] = {}
                    
                corr_idx_path = self.root_path / "caption_idx" / "truckscenes_entity_matching_idx" / f"{scene_name}.pkl"
                if corr_idx_path.exists():
                    with open(corr_idx_path, 'rb') as f:
                        pkl = pickle.load(f)
                        for camera_view, i in pkl.items():
                           scene_image_corr_entity_infos[scene_name][camera_view.upper()] = torch.tensor(i)
                    # with open(corr_idx_path, 'rb') as f:
                    #     scene_image_corr_entity_infos[scene_frame_key] = pickle.load(f)

        return scene_image_corr_infos, scene_image_corr_entity_infos