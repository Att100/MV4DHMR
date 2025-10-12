@REM python demo.py --dataset chi3d ^
@REM     --dataset_dir D:/Workspace/datasets/chi3d ^
@REM     --pretrained_path ./pretrained/multiHMR_672_L.pt ^
@REM     --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt ^
@REM     --smplx_dir models ^
@REM     --smplx2smpl_path models/smplx/smplx2smpl.pkl ^
@REM     --j_regressor_h36m_path models/J_regressor_h36m.npy ^
@REM     --save_dir ./output/chi3d_s4_grab_07_full ^
@REM     --subject s04 ^
@REM     --sequence "Grab 7" ^
@REM     --image_size 672 ^
@REM     --n_humans 2 ^
@REM     --backbone dinov2_vitl14 ^
@REM     --smplx_type smplx ^
@REM     --data_smplx_type smplx ^
@REM     --eval_mode finetune ^
@REM     --recenter ^
@REM     --smooth ^
@REM     --auto_interpolate ^
@REM     --vis ^
@REM     --vis_mode render ^
@REM     --vid ^
@REM     --save ^
@REM     --device 0

python demo.py --dataset panoptic ^
    --dataset_dir D:\Workspace\datasets\panoptic-toolbox ^
    --pretrained_path ./models/multiHMR/multiHMR_896_L.pt ^
    --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt ^
    --sequence 160422_haggling1 ^
    --start_frame 3200 ^
    --end_frame 3500 ^
    --smplx_dir models ^
    --smplx2smpl_path models/smplx/smplx2smpl.pkl ^
    --j_regressor_h36m_path models/J_regressor_h36m.npy ^
    --save_dir ./output/panoptic_160422_haggling1_3200_3500 ^
    --image_size 896 ^
    --image_size_org 1920,1080 ^
    --n_humans 3 ^
    --backbone dinov2_vitl14 ^
    --smplx_type smplx ^
    --data_smplx_type smplx ^
    --eval_mode zeroshot ^
    --smooth ^
    --auto_interpolate ^
    --vis ^
    --vis_mode render ^
    --vid ^
    --save ^
    --device 0

@REM python demo.py --dataset shelf ^
@REM     --dataset_dir D:\\Workspace\\datasets\\Shelf ^
@REM     --pretrained_path ./models/multiHMR/multiHMR_896_L.pt ^
@REM     --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt ^
@REM     --start_frame 530 ^
@REM     --end_frame 870 ^
@REM     --smplx_dir models ^
@REM     --smplx2smpl_path models/smplx/smplx2smpl.pkl ^
@REM     --j_regressor_h36m_path models/J_regressor_h36m.npy ^
@REM     --save_dir ./output/shelf_530_870 ^
@REM     --image_size 896 ^
@REM     --image_size_org 1032,776 ^
@REM     --n_humans 4 ^
@REM     --backbone dinov2_vitl14 ^
@REM     --smplx_type smplx ^
@REM     --data_smplx_type smplx ^
@REM     --eval_mode zeroshot ^
@REM     --smooth ^
@REM     --auto_interpolate ^
@REM     --vis ^
@REM     --vis_mode scatter ^
@REM     --vid ^
@REM     --save ^
@REM     --device 0
