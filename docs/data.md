# Data

## 1. CHI3D

- Download CHI3D training set from [CHI3D](https://ci3d.imar.ro/chi3d).

- Directory
    ```
    chi3d
    └─train
        ├─s02
        │  ├─camera_parameters
        │  │  ├─50591643
        │  │  ├─58860488
        │  │  ├─60457274
        │  │  └─65906101
        │  ├─gpp
        │  ├─joints3d_25
        │  ├─smplx
        │  └─videos
        │      ├─50591643
        │      ├─58860488
        │      ├─60457274
        │      └─65906101
        ...
    ```

- Extract frames
    - Create python file under `chi3d` with the following code and run, which while extract images adn save to `train/chi3d/s02/images`, ...

    ```
    import cv2
    import os
    from tqdm import tqdm


    def extract_frames(vid_path, frame_path):
        cap = cv2.VideoCapture(vid_path)
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(
                os.path.join(frame_path, f"{str(i).zfill(6)}.jpg"), frame)
            i += 1
            

    if __name__ == "__main__":
        root_path = "./"
        sub_set = 'train'
        sequences = ['s02', 's03', 's04']
        camera_ids = ['50591643', '58860488', '60457274', '65906101']
        
        for seq in sequences:
            for cam in camera_ids:
                seq_path = os.path.join(root_path, sub_set, seq, 'videos', str(cam))
                for vid_fname in tqdm(os.listdir(seq_path)):
                    seq_vid_frame_path = os.path.join(
                        root_path, sub_set, seq, 'images', str(cam), vid_fname.split(".")[0])
                    if not os.path.exists(seq_vid_frame_path):
                        os.makedirs(seq_vid_frame_path)
                    extract_frames(
                        os.path.join(seq_path, vid_fname),
                        seq_vid_frame_path)
                print(f"seq: {seq} - cam: {cam} finished !!")

    ```

## 2. Hi4D

## 3. Panoptic

- Download Panoptic toolbox

    ```
    git clone https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git
    cd panoptic-toolbox
    ```

- Modify code in `scripts/getData.sh`

    ```
    # Modify line (under `Download HD video` section):
    nodes=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

    # To
    nodes=(12 6 23 13 3)
    ```

- Download sequence

    ```
    ./scripts/getData.sh <SEQUENCE_NAME> 0 5
    ```

- Extact frames and data

    ```
    ./scripts/extractAll.sh <SEQUENCE_NAME>
    ```

## 4. Shelf

