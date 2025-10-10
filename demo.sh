# !/bin/bash

python demo.py --dataset chi3d \
    --dataset_dir D:/Workspace/datasets/chi3d \
    --pretrained_path ./pretrained/multiHMR_672_L.pt \
    --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt \
    --smplx_dir models \
    --smplx2smpl_path models/smplx/smplx2smpl.pkl \
    --j_regressor_h36m_path models/J_regressor_h36m.npy \
    --save_dir ./output/chi3d_s4_grab_07_full \
    --subject s4 \
    --sequence 'Grab 7' \
    --image_size 672 \
    --n_humans 2 \
    --backbone dinov2_vitl14 \
    --smplx_type smplx \
    --data_smplx_type smplx \
    --eval_mode finetune \
    --recenter \
    --smooth \
    --auto_interpolate \
    --vis \
    --vis_mode render \
    --vid \
    --save \
    --device 0

# python demo.py --dataset panoptic \
#     --dataset_dir D:\Workspace\datasets\panoptic-toolbox \
#     --pretrained_path ./models/multiHMR/multiHMR_896_L.pt \
#     --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt \
#     --sequence 160422_haggling1 \
#     --start_frame 3200 \
#     --end_frame 3500 \
#     --smplx_dir models \
#     --smplx2smpl_path models/smplx/smplx2smpl.pkl \
#     --j_regressor_h36m_path models/J_regressor_h36m.npy \
#     --image_size 896 \
#     --image_size_org 1920,1080 \
#     --n_humans 3 \
#     --backbone dinov2_vitl14 \
#     --smplx_type smplx \
#     --data_smplx_type smplx \
#     --eval_mode zeroshot \
#     --smooth \
#     --auto_interpolate \
#     --vis \
#     --vis_mode render \
#     --vid \
#     --save \
#     --device 0