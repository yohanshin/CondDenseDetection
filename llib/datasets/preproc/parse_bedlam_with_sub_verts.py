import os
import sys
sys.path.append('./')

import torch
import joblib
import numpy as np
from tqdm import tqdm

from configs import constants as _C

if __name__ == '__main__':
    bedlam_part_dir = os.path.join(_C.PATHS.PARSED_DATA_DIR, "bedlam_2d_parts")
    bedlam_part_file_pths = sorted(os.listdir(bedlam_part_dir))

    # TODO: Change if you want to use new vertices
    verts_name = "verts_306"
    # verts_name = "verts_512"
    subsample_pth = f"/large_experiments/3po/model/smplx/{verts_name}.pkl"
    subsample = torch.argmax(joblib.load(subsample_pth), dim=1).numpy()
    output_pth = os.path.join(_C.PATHS.PARSED_DATA_DIR, f"bedlam_2d_{verts_name}.pth")

    J_regressor = np.load("/large_experiments/3po/model/smplx/J_regressor_coco+feet.npy")

    dataset = dict(
        centers=[],
        scales=[],
        imgpaths=[],
        subsample_verts2d=[],
        conditions=[],
    )

    for bedlam_part_file_pth in tqdm(bedlam_part_file_pths):
        bedlam_part_file = joblib.load(os.path.join(bedlam_part_dir, bedlam_part_file_pth))

        dataset["centers"].append(bedlam_part_file["centers"].copy())
        dataset["scales"].append(bedlam_part_file["scales"].copy())
        dataset["imgpaths"].append(bedlam_part_file["imgpaths"].copy())
        
        verts2d = bedlam_part_file["verts2d"]
        subsample_verts2d = verts2d[:, subsample].copy()
        dataset["subsample_verts2d"].append(subsample_verts2d)

        sparse_condition = np.matmul(J_regressor[None], verts2d)
        dataset["conditions"].append(sparse_condition)
        
        del bedlam_part_file

    for k, v in dataset.items():
        dataset[k] = np.concatenate(v)

    dataset["centers"] = dataset["centers"].astype(np.float32)
    dataset["scales"] = dataset["scales"].astype(np.float32)
    dataset["subsample_verts2d"] = dataset["subsample_verts2d"].astype(np.float32)
    dataset["conditions"] = dataset["conditions"].astype(np.float32)
    joblib.dump(dataset, output_pth)