import os
import sys
sys.path.append('./')
import argparse

import cv2
import torch
import joblib
import numpy as np
from tqdm import tqdm
from smplx import SMPLX

from configs import constants as _C

def get_smplx_vertices(poses, betas, trans, smplx_model):
    if isinstance(trans, np.ndarray):
        trans = torch.from_numpy(trans)
    model_out = smplx_model(betas=torch.tensor(betas).unsqueeze(0).float(),
                            global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                            body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                            left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
                            right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
                            jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                            leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                            reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                            transl=trans.clone().unsqueeze(0).float())
    
    return model_out.vertices[0], model_out.joints[0]


def preprocess(part, n_parts=8):
    # Divide the processing by 8 parts

    smplx = {
        # gender: SMPL(
        gender: SMPLX(
            _C.PATHS.SMPLX_MODEL_DIR, 
            gender=gender, 
            num_betas=num_betas, 
            flat_hand_mean=True,
            use_pca=False
        ) for num_betas, gender in zip([11, 10, 10], ['neutral', 'female', 'male'])
    }
    
    annot_fldr = os.path.basename(_C.PATHS.BEDLAM_ANNOT_DIR)
    image_fldr = os.path.basename(_C.PATHS.BEDLAM_IMAGE_DIR)
    sequence_list = sorted(os.listdir(_C.PATHS.BEDLAM_ANNOT_DIR))
    sequence_list = sequence_list[part::n_parts]

    dataset = dict(
        centers=[],
        scales=[],
        imgpaths=[],
        verts2d=[],
    )
    save_pth = os.path.join(_C.PATHS.PARSED_DATA_DIR, f'bedlam_2d_{part+1:02d}.pth')
    for sequence in (bar1 := tqdm(sequence_list, total=len(sequence_list), dynamic_ncols=True, leave=True)):
        sequence_dir = os.path.join(_C.PATHS.BEDLAM_ANNOT_DIR, sequence, 'png')
        bar1.set_description_str(sequence)
        
        subsequence_list = sorted(os.listdir(sequence_dir))
        for subsequence in (bar2 := tqdm(subsequence_list, total=len(subsequence_list), dynamic_ncols=True, leave=True)):
            subsequence_dir = os.path.join(sequence_dir, subsequence)
            image_dir = subsequence_dir.replace(annot_fldr, image_fldr)
            files = sorted(os.listdir(subsequence_dir))

            annot_path_list = [os.path.join(subsequence_dir, file) for file in files]
            image_path_list = [os.path.join(image_dir, file.replace(".data.pyd", ".png")) for file in files]

            for annot_path, image_path in (bar3 := tqdm(zip(annot_path_list, image_path_list), total=len(annot_path_list), leave=False)):
                annots = joblib.load(annot_path)
                n_people = len(annots)
                if n_people == 0:
                    continue
                
                intrinsic = torch.from_numpy(annots[0]['cam_int']).float()
                for annot in annots:
                    # Build SMPL GT
                    with torch.no_grad():
                        verts_cam, _ = get_smplx_vertices(annot["pose_cam"], annot["shape"], annot["trans_cam"], smplx[annot["gender"]])

                    verts_25d = torch.div(verts_cam, verts_cam[:, -1:])
                    verts_img = (intrinsic @ verts_25d.T).T[..., :2]
                    
                    if False:
                        img = cv2.imread(image_path)
                        for xy in verts_img:
                            img = cv2.circle(img, (int(xy[0]), int(xy[1])), color=(0, 255, 0), radius=2, thickness=-1)
                        cv2.imwrite('test.png', img)

                    dataset["centers"].append(torch.FloatTensor(annot['center']))
                    dataset["scales"].append(torch.FloatTensor([annot['scale']]))
                    dataset["verts2d"].append(verts_img)
                    dataset["imgpaths"].append(image_path)
            
    dataset["centers"] = torch.stack(dataset["centers"]).numpy().astype(np.float32)
    dataset["scales"] = torch.cat(dataset["scales"]).numpy().astype(np.float32)
    dataset["verts2d"] = torch.stack(dataset["verts2d"]).numpy().astype(np.float32)
    dataset["imgpaths"] = np.array(dataset["imgpaths"])
    joblib.dump(dataset, save_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--part', type=int, default=0, help='Part')
    args = parser.parse_args()

    preprocess(args.part)