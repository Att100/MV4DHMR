# Training-free Multi-view 4D Human Motion Reconstruction Virtual Reality System

## 1. Quick Start (simplified version)

### 1.1 Environment

1. Please follow [MultiHMR](https://github.com/naver/multi-hmr) to set up your python virtual environment.
2. Install `ffmpeg` in the env created above.

    ```
    conda activate multihmr
    conda install conda-forge::ffmpeg
    ```

### 1.2 Run Demo Sequence

Activate virtual environment and:

- On Linux based OS

    ```
    sh demo.sh
    ```

- On Windows OS

    ```
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


