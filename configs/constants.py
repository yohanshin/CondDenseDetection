import os

class PATHS:
    THREEPO_BASE_DIR = "/large_experiments/3po"
    DATA_BASE_DIR = f"{THREEPO_BASE_DIR}/data/images"
    BEDLAM_ANNOT_DIR = f"{DATA_BASE_DIR}/bedlam_fixed_annots"
    BEDLAM_IMAGE_DIR = f"{DATA_BASE_DIR}/bedlam"
    EMDB_DIR = f"{DATA_BASE_DIR}/emdb"

    PARSED_DATA_DIR = f"{THREEPO_BASE_DIR}/data/parsed_data"

    # BODY MODELS
    SMPL_MODEL_DIR = f"{THREEPO_BASE_DIR}/model/smpl"
    SMPLX_MODEL_DIR = f"{THREEPO_BASE_DIR}/model/smplx"
    SMPLX2SMPL_PTH = f"{THREEPO_BASE_DIR}/model/smplx/smplx2smpl.pkl"
    SMPL2SPARSE_PTH = f"{THREEPO_BASE_DIR}/model/smpl/J_regressor_coco+feet.npy"

    # Auxiliary
    YOLOV8_CKPT_PTH = "/checkpoint/soyongshin/checkpoints/yolov8x.pt"