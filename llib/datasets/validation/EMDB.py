import sys, os
sys.path.append("./")

import cv2
import json
import tqdm
import torch
import random
import joblib
import numpy as np

from configs import constants as _C
from configs.landmarks import verts_306, verts_512
from ..HumanPoseEstimation import HumanPoseEstimationDataset as Dataset
from ..utils.augmentation import xyxy2cs
from ...models.utils.transform import get_affine_transform

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


class EMDBDataset(Dataset):
    def __init__(self, 
                 label_path,
                 landmark_type="verts_306",
                 condition=True,
                 **kwargs):
        super(EMDBDataset, self).__init__(**kwargs)

        self.landmark_type = landmark_type
        self.condition = condition
        if self.landmark_type == "verts_306":
            landmark_cfg = verts_306
        elif self.landmark_type == "verts_512":
            landmark_cfg = verts_512
        else:
            raise NotImplementedError

        self.mean = 255 * np.array([0.485, 0.456, 0.406])
        self.std = 255 * np.array([0.229, 0.224, 0.225])
        self.labels = joblib.load(label_path)

        self.smplx2smpl = joblib.load(_C.PATHS.SMPLX2SMPL_PTH)["matrix"]
        smplx2subsample = joblib.load(landmark_cfg.subsample_pth).numpy()
        smpl2subsample = smplx2subsample @ self.smplx2smpl.T
        smpl2subsample[smpl2subsample < 0.5] = 0.0
        
        # Only points that have higher overlap will be used for validation
        target_weights = np.zeros(len(smpl2subsample))
        target_weights[smpl2subsample.sum(1) > 0.5] = 1.0
        self.target_weights = target_weights
        self.subsample_idxs = np.argmax(smpl2subsample, axis=1)

        self.J_regressor = np.load(_C.PATHS.SMPL2SPARSE_PTH)


    def __len__(self):
        return len(self.labels["imgpaths"])

    def __getitem__(self, index):
        joints_data = {}

        image = cv2.imread(self.labels["imgpaths"][index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        smpl_verts = self.labels["verts2d"][index].copy()
        joints = smpl_verts[self.subsample_idxs]
        bbox = self.labels["bboxes"][index].copy()
        c, s = xyxy2cs(bbox, self.aspect_ratio, self.pixel_std)

        # Apply affine transform on joints and image
        trans = get_affine_transform(c, s, self.pixel_std, 0, self.image_size)
        
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        joints = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
        joints = joints @ trans.T

        # Convert image to tensor and normalize
        if self.transform is not None:  # I could remove this check
            # cvimg = image.copy()
            image  = convert_cvimg_to_tensor(image)  # convert from HWC to CHW
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        # _image = cvimg.copy()
        # for xy in joints[:, :2]:
        #     cv2.circle(_image, (int(xy[0]), int(xy[1])), color=(0, 255, 0), radius=2, thickness=-1)
        # cv2.imwrite("test.png", _image[..., ::-1])

        joints[:, 0] /= self.image_size[0]
        joints[:, 1] /= self.image_size[1]
        joints = joints * 2 - 1

        if self.condition:
            sparse_cond = self.J_regressor @ smpl_verts
            sparse_cond = np.concatenate((sparse_cond, np.ones((sparse_cond.shape[0], 1))), axis=1)
            sparse_cond = sparse_cond @ trans.T
            sparse_cond[:, 0] /= self.image_size[0]
            sparse_cond[:, 1] /= self.image_size[1]
            sparse_cond = sparse_cond * 2 - 1
            cond_mask = np.zeros(sparse_cond.shape[0]).astype(bool)
        else:
            sparse_cond = np.array([0])
            cond_mask = np.array([0]).astype(bool)

        mask_bits = np.zeros(self.num_joints)

        joints_data['joints'] = joints
        joints_data['cond'] = sparse_cond.astype(np.float32)
        joints_data['cond_mask'] = cond_mask
        joints_data['mask_bits'] = mask_bits
        joints_data['image'] = image.astype(np.float32)
        joints_data['target_weight'] = self.target_weights.reshape(-1, 1)
        return joints_data


if __name__ == "__main__":
    label_pth = "/large_experiments/3po/data/parsed_data/emdb1_6fps.pth"
    dataset = EMDBDataset(label_path=label_pth, condition=True)

    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    import matplotlib.pyplot as plt
    is_plot = True
    count = 0
    for epoch in tqdm.tqdm(range(10)):
        for batch in dataloader:
            pass