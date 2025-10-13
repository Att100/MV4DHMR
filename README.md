# Training-free Multi-view 4D Human Motion Reconstruction Virtual Reality System

## 1. Quick Start (simplified version)

### 1.1 Environment

1. Install conda python virtual environment.
    (modified from https://github.com/naver/multi-hmr)

    ```
    conda env create -f conda.yaml
    conda activate multihmr
    ```

2. Install `ffmpeg` in the env created above.

    ```
    conda activate multihmr
    conda install conda-forge::ffmpeg
    ```

### 1.2 Our Ready to Run Codebase

- Please download our `ready-to-run` codebase (with all required model checkpoints and SMPL/SMPL-X support), from [Google Drive]() and extract the files.

- Under `sample_data`, we provide a short sequence from Panoptic dataset.

### 1.3 Run Demo Sequence (Panoptic)

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

### 1.4 Other Datasets

**CHI3D**

```
python demo.py --dataset chi3d \
    --dataset_dir ./sample_data/chi3d \
    --pretrained_path ./pretrained/multiHMR_672_L.pt \
    --checkpoint_path checkpoints/saved/model_large_672_chi3d_sv.pt \
    --smplx_dir models \
    --smplx2smpl_path models/smplx/smplx2smpl.pkl \
    --j_regressor_h36m_path models/J_regressor_h36m.npy \
    --save_dir ./output/chi3d_s4_grab_07_full \
    --subject s04 \
    --sequence "Grab 7" \
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



