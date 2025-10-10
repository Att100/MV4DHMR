import numpy as np
import os
import json

CAM_IDS = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]

M = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0]])

class PanopticSequenceMetaLoader(object):
    def __init__(self, root_path):
        self.root_path = root_path
    
    def load(self, seq_name):
        with open(os.path.join(self.root_path, seq_name, f'calibration_{seq_name}.json')) as f:
            cams_calib = json.load(f)
        
        cams = dict()
        for cam in cams_calib["cameras"]:
            if (cam['panel'], cam['node']) in CAM_IDS:
                sel_cam = {}
                sel_cam['K'] = np.array(cam['K'])
                sel_cam['R'] = np.array(cam['R']).dot(M)
                sel_cam['t'] = np.array(cam['t']).reshape((1, 3)) / 100.0  # cm -> m
                key = f"{str(cam['panel']).zfill(2)}_{str(cam['node']).zfill(2)}"
                cams[key] = sel_cam
                
        K, R, t = [], [], []
        for panel, node in CAM_IDS:
            key = f"{str(panel).zfill(2)}_{str(node).zfill(2)}"
            K.append(cams[key]['K'][None])
            R.append(cams[key]['R'][None])
            t.append(cams[key]['t'][None])
        
        K, R, t = [np.concatenate(v, axis=0) for v in (K, R, t)]
        
        fnames = os.listdir(
            os.path.join(
                self.root_path, seq_name, 'hdImgs', 
                f"{str(CAM_IDS[0][0]).zfill(2)}_{str(CAM_IDS[0][1]).zfill(2)}"))
        fnames = sorted([v.split(".")[0].split("_")[-1] for v in fnames])
        
        images = []
        for fn in fnames:
            cur_images = []
            for panel, node in CAM_IDS:
                key = f"{str(panel).zfill(2)}_{str(node).zfill(2)}"
                cur_images.append(os.path.join(self.root_path, seq_name, 'hdImgs', key, f"{key}_{fn}.jpg"))
            images.append(cur_images)
        
        return images, K, R, t