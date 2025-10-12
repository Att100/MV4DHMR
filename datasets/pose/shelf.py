import numpy as np
import os
import json

CAM_IDS = [0, 1, 2, 3, 4]


class ShelfSequenceMetaLoader(object):
    def __init__(self, root_path):
        self.root_path = root_path
    
    def load(self):
        K, R, t = [], [], []
        for cid in CAM_IDS:
            with open(os.path.join("data/shelf/camera_parameters", f"fisheye_param_0{cid}.json"), "r") as f:
                calib = json.load(f)
                Kc = np.array(calib['intrinsic'], dtype=np.float32)
                Kc = Kc[:3,:3]
                Kc[-1,-1] = 1.0
                Rc = np.array(calib['extrinsic_r'], dtype=np.float32)
                tc = np.array(calib['extrinsic_t'], dtype=np.float32).reshape(1, 3)
            K.append(Kc[None])
            R.append(Rc[None])
            t.append(tc[None])
        
        K, R, t = [np.concatenate(v, axis=0) for v in [K, R, t]]
        
        fnames = [v.split(".")[0] for v in \
            sorted(os.listdir(os.path.join(self.root_path, "Camera0")))]
        
        images = []
        for fname in fnames:
            cur_images = []
            for cid in CAM_IDS:
                cur_images.append(
                    os.path.join(self.root_path, f"Camera{cid}", f"{fname}.png")
                )
            images.append(cur_images)
            
        return images, K, R, t
            
        