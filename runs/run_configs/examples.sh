# # Unconditional model
# ViT-Base
exp_name=regressor_xCond-ViT-Base-verts_306_2025-06-04-00-18-58

# ViT-Large
backbone=vitl exp_name=regressor_xCond-ViT-Large-verts_306_2025-06-04-00-19-00

# # Conditional model
# ViT-Base
model=encdec_Cond_base

# ViT-Large
backbone=vitl model=encdec_Cond_large exp_name=encdec_Cond-ViT-Large-verts_306_2025-06-04-03-26-16

# # Diffusion
# ViT-Base
--config-name "config_diffusion.yaml"

# ViT-Large
--config-name "config_diffusion.yaml" exp_name=diffusion-ViT-Large-verts_306_2025-06-04-21-34-45 backbone=vitl model.decoder.n_enc_layers=3 model.decoder.n_dec_layers=3 model.decoder.transformer_dim_feedforward=2048