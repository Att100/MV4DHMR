# !/bin/bash

# chi3d
python test.py --exp_name model_large_672_chi3d_sv \
    --dataset chi3d \
    --dataset_dir /ocean/projects/cis240055p/czheng3/chi3d \
    --pretrained_path models/multiHMR/multiHMR_672_L.pt \
    --smplx_dir models \
    --smplx2smpl_path models/smplx/smplx2smpl.pkl \
    --j_regressor_h36m_path models/J_regressor_h36m.npy \
    --image_size 672 \
    --backbone dinov2_vitl14 \
    --smplx_type smplx \
    --data_smplx_type smplx \
    --eval_mode finetune \
    --save_dir ./checkpoints/saved \
    --recenter \
    --device 0

# hi4d
# python test.py --exp_name model_large_672_hi4d_sv \
#     --dataset hi4d \
#     --dataset_dir /ocean/projects/cis240055p/czheng3/yijiehe/Multi-View-HMR/data/3DPW \
#     --pretrained_path models/multiHMR/multiHMR_672_L.pt \
#     --smplx_dir models \
#     --smplx2smpl_path models/smplx/smplx2smpl.pkl \
#     --j_regressor_h36m_path models/smplx/smplx2smpl.pkl \
#     --image_size 672 \
#     --backbone dinov2_vitl14 \
#     --smplx_type smpl \
#     --data_smplx_type smpl \
#     --eval_mode finetune \
#     --save_dir checkpoints/saved \
#     --device 0