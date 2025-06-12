

def load_ATLAS(model_type="atlas", lod="lod3", device="cuda"):
    if model_type == "atlas":
        from llib.atlas.atlas import ATLAS
        from llib.atlas.module import BodyPoseGMMPrior
        num_shape_comps = 16
        num_scale_comps = 16
        num_hand_comps = 32 # PER HAND!
        num_expr_comps = 10
        gmm_comps = 32
        asset_dir = "/large_experiments/3po/model/atlas/"

    atlas = ATLAS(asset_dir, 
                  num_shape_comps, 
                  num_scale_comps, 
                  num_hand_comps, 
                  num_expr_comps, 
                  lod=lod, 
                  load_keypoint_mapping=lod=="lod3",
                  verbose=True
    ).to(device)
    pose_prior = BodyPoseGMMPrior(asset_dir, gmm_comps).to(device)

    return atlas, pose_prior