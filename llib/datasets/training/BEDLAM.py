import sys, os
sys.path.append("./")

import cv2
import json
import tqdm
import torch
import random
import joblib
import numpy as np

from configs.landmarks import verts_306, verts_512
from ..HumanPoseEstimation import HumanPoseEstimationDataset as Dataset
from ..utils.augmentation import extreme_cropping, augment_image, xyxy2cs
from ...models.utils.transform import fliplr_joints, get_affine_transform

def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img


class BEDLAMDataset(Dataset):
    def __init__(self, 
                 label_path,
                 landmark_type="verts_306",
                 target_type="keypoints",
                 condition=True,
                 **kwargs):
        super(BEDLAMDataset, self).__init__(**kwargs)

        self.landmark_type = landmark_type
        self.condition = condition
        if self.landmark_type == "verts_306":
            landmark_cfg = verts_306
        elif self.landmark_type == "verts_512":
            landmark_cfg = verts_512
        else:
            raise NotImplementedError

        self.target_type = target_type
        self.flip_pairs = landmark_cfg.flip_pairs
        joints_weight = [1] * self.max_num_joints
        self.joints_weight = np.array(joints_weight).reshape(self.max_num_joints, 1)
        
        self.body_parts_dict = landmark_cfg.body_parts_dict
        self.body_idx = landmark_cfg.body_idx
        self.parts2body_idx = np.argsort(self.body_idx)

        self.mean = 255 * np.array([0.485, 0.456, 0.406])
        self.std = 255 * np.array([0.229, 0.224, 0.225])
        self.labels = joblib.load(label_path)

        self.cond_flip_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)
        ]
    
    def __len__(self):
        return len(self.labels["imgpaths"])

    def __getitem__(self, index):
        joints_data = {}
        
        img_name = self.labels["imgpaths"][index].copy()
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if "closeup" in img_name:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        joints = self.labels["subsample_verts2d"][index].copy()
        
        # Add condition
        if self.condition:
            sparse_cond = self.labels["conditions"][index].copy()
            
        # Apply cropping augmentation
        is_random_crop = torch.rand(1).item() if self.is_train else 1.0
        if self.is_train and (is_random_crop < self.extreme_cropping_prob):
            try: 
                bbox, rescale = extreme_cropping(joints, self.body_parts_dict, image.shape[1], image.shape[0])
                c, s = xyxy2cs(bbox, self.aspect_ratio, self.pixel_std)
                s = s * rescale
            except:
                c = self.labels["centers"][index].copy()
                s = self.labels["scales"][index].copy()
                s = s / 1.2  # remove CLIFF_SCALE_FACTOR
                s = np.array([s,s])    
        else:
            c = self.labels["centers"][index].copy()
            s = self.labels["scales"][index].copy()
            s = s / 1.2  # remove CLIFF_SCALE_FACTOR
            s = np.array([s,s])    

        r = 0

        # Apply data augmentation
        f = False
        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor

            if self.scale:
                s = s * np.clip(random.random() * sf + 1, 1 - sf, 1 + sf)  # A random scale factor in [1 - sf, 1 + sf]

            if self.trans_factor > 0 and random.random() < self.trans_prob:
                # multiplying by self.pixel_std removes the 200 scale
                trans_x = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[0]
                trans_y = np.random.uniform(-self.trans_factor, self.trans_factor) * self.pixel_std * s[1]
                c[0] = c[0] + trans_x
                c[1] = c[1] + trans_y

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)  # A random rotation factor in [-2 * rf, 2 * rf]
            else:
                r = 0

            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                joints = fliplr_joints(joints, image.shape[1], self.flip_pairs)
                if self.condition:
                    sparse_cond = fliplr_joints(sparse_cond, image.shape[1], self.cond_flip_pairs)
                c[0] = image.shape[1] - c[0] - 1
                f = True
            else:
                f = False

            if torch.rand(1).item() < self.img_aug_prob:
                # Numbers taken from bedlam/core/datasets/utils.py get_example
                image = augment_image(image)

        # Apply affine transform on joints and image
        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        joints = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = joints @ trans.T
        
        if self.condition:
            sparse_cond = np.concatenate((sparse_cond, np.ones((sparse_cond.shape[0], 1))), axis=1)
            sparse_cond = sparse_cond @ trans.T

        # Convert image to tensor and normalize
        if self.transform is not None:  # I could remove this check
            # cvimg = image.copy()
            image  = convert_cvimg_to_tensor(image)  # convert from HWC to CHW
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        valid_joints_x = (np.zeros(joints.shape[0]) <= joints[:, 0]) & (joints[:, 0] < self.image_size[0])
        valid_joints_y = (np.zeros(joints.shape[0]) <= joints[:, 1]) & (joints[:, 1] < self.image_size[1])
        valid_joints = (valid_joints_x & valid_joints_y)*1
        valid_joints = valid_joints[:, None]
        target_weight = valid_joints.copy().astype(np.float32)

        if self.target_type == "keypoints":
            if joints[valid_joints[:,0]==0].shape[0] > 0:
                # normalize joints outside the image to [-1, 1]
                outside_joints = 2*(joints[valid_joints[:,0]==0]/self.image_size - 0.5)
                # calculate the distance of the joints to the center
                outside_joints = np.linalg.norm(outside_joints, axis=1)
                beta = 2. #0.25 #4
                outside_joints= np.exp(-beta*np.abs(outside_joints-1))
                target_weight[valid_joints[:,0]==0, 0] = outside_joints

            if True:
                for key, val in self.body_parts_dict.items():
                    # add 1 to the joints like hands, feet and head
                    if valid_joints[val].sum() > 0 and key in ["left_hand", "right_hand", "left_feet", "right_feet"]:
                        target_weight[val] = target_weight[val] + target_weight[val]

        if self.target_type == "keypoints":
            target = np.array([0])
        else:
            target, _ = self._generate_target(joints, valid_joints, score_invis=1.0)

        # # if "closeup" in img_name:
        # _image = cvimg.copy()
        # for xy in joints[:, :2]:
        #     cv2.circle(_image, (int(xy[0]), int(xy[1])), color=(0, 255, 0), radius=2, thickness=-1)
        # for xy in sparse_cond[:, :2]:
        #     cv2.circle(_image, (int(xy[0]), int(xy[1])), color=(0, 0, 255), radius=2, thickness=-1)
        # cv2.imwrite("test.png", _image[..., ::-1])
        
        # scale joints to [-1, 1]
        joints[:, 0] /= self.image_size[0]
        joints[:, 1] /= self.image_size[1]
        joints = joints * 2 - 1

        if self.condition:
            sparse_cond[:, 0] /= self.image_size[0]
            sparse_cond[:, 1] /= self.image_size[1]
            sparse_cond = sparse_cond * 2 - 1
            
            # Apply random 30% mask
            # TODO: Fix this hard coding
            cond_mask = np.random.rand(len(sparse_cond)) < self.cond_mask_prob

            # Add sparse condition noise
            # Triple the error for hips and shoulders
            # TODO: Fix this hard coding
            sparse_cond_noise = np.random.randn(sparse_cond.shape[0], sparse_cond.shape[1])
            sparse_cond_noise = sparse_cond_noise * 0.01
            sparse_cond_noise[[5, 6, 11, 12]] *= 3

            sparse_cond = sparse_cond + sparse_cond_noise
        else:
            sparse_cond = np.array([0])
            cond_mask = np.array([0]).astype(bool)

        if self.ldmk_prompt_prob > 0.0 and np.random.rand() < self.ldmk_prompt_prob:
            mask_bits = np.random.rand(self.num_joints) < self.ldmk_prompt_ratio
            target_weight[mask_bits] *= 0.0 # Not computing on those points
        else:
            mask_bits = np.zeros(self.num_joints).astype(bool)

        joints_data['joints'] = joints
        joints_data['cond'] = sparse_cond.astype(np.float32)
        joints_data['cond_mask'] = cond_mask
        joints_data['mask_bits'] = mask_bits
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['flip'] = f if self.is_train else False
        joints_data['image'] = image.astype(np.float32)
        joints_data['target'] = target.astype(np.float32)
        joints_data['target_weight'] = target_weight.astype(np.float32)
        return joints_data


if __name__ == "__main__":
    label_pth = "/large_experiments/3po/data/parsed_data/bedlam_2d_verts_306.pth"
    dataset = BEDLAMDataset(label_path=label_pth, condition=True)

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
    )

    import matplotlib.pyplot as plt
    import alphashape
    from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
    from shapely.ops import unary_union
    from skimage.draw import polygon
    
    is_plot = True
    count = 0
    for epoch in tqdm.tqdm(range(10)):
        for batch in dataloader:
            images = batch['image']
            joints = batch['joints']
            for i, (img, joints) in enumerate(zip(images, joints)):
                fig, ax = plt.subplots(1, 3)
                img_norm = (img.numpy() * std + mean).transpose(1, 2, 0)
                img_norm_ori = img_norm.copy()
                ax[0].imshow(img_norm_ori)
                print(img_norm.max(), img_norm.min(), img_norm_ori.max(), img_norm_ori.min())
                ax[1].imshow(img_norm)
                ax[2].imshow(img_norm)
                joints = (joints + 1) / 2
                joints[:, 0] *= img.shape[2]
                joints[:, 1] *= img.shape[1]
                joints = joints.numpy()
                
                ax[1].scatter(joints[:, 0], joints[:, 1], c="b", s=1)
                
                # Get concave hull
                # alpha = alphashape.optimizealpha(joints)
                # concave_hull = alphashape.alphashape(joints, 0.001)
                # concave_hull = alphashape.alphashape(joints, 50)
                concave_hull = alphashape.alphashape(joints, 384)
                
                mask = np.zeros((img.shape[1], img.shape[2])).astype(np.uint8)
                if isinstance(concave_hull, MultiPolygon):
                    n_hulls = len(concave_hull.geoms)
                    for i_hull in range(n_hulls):
                        xs, ys = concave_hull.geoms[i_hull].exterior.coords.xy
                        rr, cc = polygon(ys, xs, shape=(img.shape[1], img.shape[2]))
                        mask[rr, cc] = 1
                elif isinstance(concave_hull, GeometryCollection):
                    poly_list = []
                    for part in concave_hull.geoms:
                        if isinstance(concave_hull, Polygon):
                            poly_list.append(part)
                        elif isinstance(concave_hull, MultiPolygon):
                            poly_list.extend(list(part.geoms))

                    for poly in poly_list:
                        xs, ys = poly.exterior.coords.xy
                        rr, cc = polygon(ys, xs, shape=(img.shape[1], img.shape[2]))
                        mask[rr, cc] = 1
                else:
                    xs, ys = concave_hull.exterior.coords.xy
                    rr, cc = polygon(ys, xs, shape=(img.shape[1], img.shape[2]))
                    mask[rr, cc] = 1
                
                ax[2].imshow(mask, cmap="Reds")
                plt.savefig(f"outputs/vis/data_proc/bedlam/polygon/img_{count:04d}.png")
                plt.close()
                count += 1 
                import pdb; pdb.set_trace()