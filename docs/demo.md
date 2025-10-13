# Quick Start

## 1. Run Demo Sequence (Panoptic)

Activate virtual environment and:

- On Linux based OS

    ```
    # !! replace the path behind `--dataset_dir` with your own panoptic-toolbox path
    sh demo.sh
    ```

- On Windows OS

    ```
    # !! replace the path behind `--dataset_dir` with your own panoptic-toolbox path
    demo.cmd
    ```

After the whole process is finished, you can find the output:

- `Tracked SMPL/SMPL-X sequence`: <YOUR_OUTPUT_DIR>/smplx/sequence_smplx.pkl

    ```
    Exported format:
    dict(
        tracking={
            trk_id: [
                (frame_id, smplx_params),
                ...
            ],
            ...
        },
        inverse_index=[
            [
                (trk_id, index_in_frame_list_of_tracking),
                ...
            ],  # length = number of tracked humann in this frame
            ...
        ],  # length = n_frames
        smplx_type=smplx/smpl,
    )

    you can use the inverse_index to retrieve the smplx_params of each human in each frame.

    e.g. Loading tracking frame by frame:
    for frame_id in range(n_frames):  # each frame
        for trk_id, invi in inverse_index[frame_id]:  # each human in the frame (including trk_id)
            smplx_params = tracking[trk_id][invi][1]
    ```

- `Visualization`: <YOUR_OUTPUT_DIR>/visualization
- `Video`: <YOUR_OUTPUT_DIR>/output.mp4

## 2. Other Datasets

**CHI3D**

```
# !! replace the path behind `--dataset_dir` with your own CHI3D path
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
```

**Shelf**

```
# !! replace the path behind `--dataset_dir` with your own Shelf path
python demo.py --dataset shelf \
    --dataset_dir D:\\Workspace\\datasets\\Shelf \
    --pretrained_path ./models/multiHMR/multiHMR_896_L.pt \
    --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt \
    --start_frame 530 \
    --end_frame 870 \
    --smplx_dir models \
    --smplx2smpl_path models/smplx/smplx2smpl.pkl \
    --j_regressor_h36m_path models/J_regressor_h36m.npy \
    --save_dir ./output/shelf_530_870 \
    --image_size 896 \
    --image_size_org 1032,776 \
    --n_humans 4 \
    --backbone dinov2_vitl14 \
    --smplx_type smplx \
    --data_smplx_type smplx \
    --eval_mode zeroshot \
    --smooth \
    --auto_interpolate \
    --vis \
    --vis_mode scatter \
    --vid \
    --save \
    --device 0
```