import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps
import os
import json
import roma
from tqdm import tqdm
import pickle

from utils.image import normalize_rgb
from datasets.closeint.utils import get_ratio_padding, \
    adjust_K_by_ratio_padding
from utils.tensor_manip import recursive_apply


CAM_IDS = ['50591643', '58860488', '60457274', '65906101']
TRAIN_SUBJECTS = ['s02', 's03']
TEST_SUBJECTS = ['s04']
IMG_SIZE_ORG = (900, 900)
N_MAX_HUMANS = 2
INTERVAL_TRAINING = 3
INTERVAL_TEST = 3


def build_k(fx, fy, cx, cy):
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

def get_seq_names(root_path, subject):
    return sorted([v.split(".")[0] for v in os.listdir(os.path.join(root_path, 'train', subject, 'smplx'))])

def load_vid_meta(root_path, subject, seq_name, camera_ids=CAM_IDS, interval=1):
    with open(os.path.join(root_path, 'train', subject, 'smplx', f"{seq_name}.json")) as f:
        smplx_params = json.load(f)
    for key in smplx_params:
        smplx_params[key] = np.array(smplx_params[key])  # (n_humans, n_frames, ...)
    
    with open(os.path.join(root_path, 'train', subject, 'joints3d_25', f"{seq_name}.json")) as f:
        j3ds = json.load(f)
    j3ds = np.array(j3ds['joints3d_25'])
    seq_len = j3ds.shape[-3]
    
    K, R, t = [], [], []
    for cam_id in camera_ids:
        with open(os.path.join(root_path, 'train', subject, 'camera_parameters', cam_id, f"{seq_name}.json")) as f:
            cam_params = json.load(f)
        Kc = build_k(
            cam_params['intrinsics_wo_distortion']['f'][0],
            cam_params['intrinsics_wo_distortion']['f'][1],
            cam_params['intrinsics_wo_distortion']['c'][0],
            cam_params['intrinsics_wo_distortion']['c'][1])
        Rc = np.array(cam_params["extrinsics"]["R"])
        tc = -np.array(cam_params["extrinsics"]["T"]) @ Rc.T
        K.append(Kc)
        R.append(Rc)
        t.append(tc)
    K, R, t = [np.concatenate([vv[None] for vv in v], axis=0) for v in [K, R, t]]
    
    samples_meta = []
    for frame_id in tqdm(range(seq_len)):
        if (frame_id+1) % interval == 0:
            meta = dict(
                subject=subject,
                seq_name=seq_name, 
                frame_name=str(frame_id).zfill(6),
                images=[], 
                smplx_params=dict(), 
                cams=dict(K=K, R=R, t=t))
            for cam_id in camera_ids:
                meta["images"].append(os.path.join(
                    root_path, 'train', subject, 'images', cam_id, seq_name, str(frame_id).zfill(6)+".jpg"))
            for k in smplx_params.keys():
                v = smplx_params[k][:, frame_id, ...]
                if k not in ['expression', 'betas', 'transl']:
                    v = roma.rotmat_to_rotvec(
                        torch.from_numpy(v).float().view(-1, 3, 3)).view(v.shape[:-2]+(3,)).numpy()
                meta['smplx_params'][k] = v
                
            samples_meta.append(meta)
            
    return samples_meta

class CHI3D(Dataset):
    def __init__(self, root_path, cache_path, img_size, split='test', reload_metas=False):
        super().__init__()
        
        self.root_path = root_path
        self.cache_path = cache_path
        self.img_size = img_size
        self.split = split
        
        subjects = TEST_SUBJECTS if split == "test" else TRAIN_SUBJECTS
        interval = INTERVAL_TRAINING if split=='training' else INTERVAL_TEST
        
        self.metas_dir = os.path.join(self.cache_path, f"{split}.meta.pkl")
        
        if not os.path.exists(self.metas_dir) or reload_metas:
            self.metas = self.load_metas(root_path, subjects, CAM_IDS, interval=interval)
            with open(self.metas_dir, 'wb') as f:
                pickle.dump(self.metas, f)
        else:
            with open(self.metas_dir, 'rb') as f:
                self.metas = pickle.load(f)
    
    def load_metas(self, root_path, subjects, cam_ids, interval=1):
        metas = []
        for subject in subjects:
            for seq_name in get_seq_names(root_path, subject):
                print(f"Processing {seq_name}/{subject}")
                metas += load_vid_meta(root_path, subject, seq_name, cam_ids, interval)
        return metas
        
    def __getitem__(self, index):
        meta = self.metas[index]
        
        ratio, px, py = get_ratio_padding(IMG_SIZE_ORG, self.img_size)
        
        imgs, K, R, t = [], [], [], []
        for img_path in meta['images']:
            # Image
            img_pil = Image.open(img_path).convert('RGB')
            img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size)) # keep the same aspect ratio
            img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size)) # pad with zero on the smallest side
            resize_img = normalize_rgb(np.asarray(img_pil))
            imgs.append(torch.from_numpy(resize_img).unsqueeze(0))
        
        K, R, t = [torch.from_numpy(meta['cams'][k]) for k in ['K', 'R', 't']]
        K = adjust_K_by_ratio_padding(K, ratio, px, py)  # (n_views, 3, 3)
            
        imgs = torch.concat(imgs, dim=0)  # (n_views, 3, H, W)
        gt_masks = torch.ones((N_MAX_HUMANS,))
        smplx_params = recursive_apply(meta['smplx_params'], lambda v:torch.from_numpy(v).float())
        
        return imgs, K, R, t, smplx_params, gt_masks
        
    def __len__(self):
        return len(self.metas)


class CHI3D_SV(CHI3D):
    def __init__(self, root_path, cache_path, img_size, smpl_model, split='test', reload_metas=False):
        super().__init__(root_path, cache_path, img_size, split, reload_metas)
        
        self.metas_dir = os.path.join(self.cache_path, f"{split}.sv.meta.pkl")
        
        if not os.path.exists(self.metas_dir) or reload_metas:
            metas_new = []
            for meta in tqdm(self.metas):
                metas_new += self.meta_world2cam(smpl_model, meta)
            self.metas = metas_new
            with open(self.metas_dir, 'wb') as f:
                pickle.dump(self.metas, f)
        else:
            with open(self.metas_dir, 'rb') as f:
                self.metas = pickle.load(f)
                
    @torch.no_grad()
    def meta_world2cam(self, smpl_model, meta):
        smplx_params = {k:torch.from_numpy(v).float().cuda() for k, v in meta["smplx_params"].items()}
        
        K, R, t = [torch.from_numpy(meta['cams'][k]).float().cuda() for k in ['K', 'R', 't']]
        output = smpl_model(**{k:v for k,v in smplx_params.items() if k != 'transl'})
        mesh = output.vertices + smplx_params['transl'].unsqueeze(1)
        joints = output.joints + smplx_params['transl'].unsqueeze(1)

        metas = []
        for vid in range(K.shape[0]):
            # apply camera exrinsic (translation) - it will compenstate rotation (translation from origin to root joint was not canceled)
            root_cam = joints[:, 0:1, :]
            mesh_cam = mesh - root_cam + torch.matmul(
                R[vid:vid+1].repeat(2, 1, 1), 
                root_cam.transpose(-1, -2)).transpose(-1, -2) + t[vid:vid+1]  # camera-centered coordinate system

            # find real transl in camera coordinate system
            transl = (mesh_cam - output.vertices)[:, 0]
            global_orient = roma.rotmat_to_rotvec(torch.matmul(
                R[vid:vid+1].repeat(2, 1, 1), 
                roma.rotvec_to_rotmat(smplx_params['global_orient'][:, 0]))).unsqueeze(1)
            _smplx_params = {k:v.cpu().numpy() for k,v in smplx_params.items() if k not in ['transl', 'global_orient']}
            _smplx_params.update(dict(
                global_orient=global_orient.reshape(-1, 1, 3).cpu().numpy(),
                transl=transl.reshape(-1, 3).cpu().numpy()
            ))
            metas.append(dict(
                subject=meta['subject'],
                seq_name=meta['seq_name'],
                frame_name=meta['frame_name'],
                view_id=vid,
                image=meta['images'][vid],
                smplx_params=_smplx_params,
                cam=dict(K=K[vid].cpu().numpy(), R=R[vid].cpu().numpy(), t=t[vid].cpu().numpy())
            ))
            
        return metas
        
    def __getitem__(self, index):
        meta = self.metas[index]
        
        ratio, px, py = get_ratio_padding(IMG_SIZE_ORG, self.img_size)

        img_pil = Image.open(meta['image']).convert('RGB')
        img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size)) # keep the same aspect ratio
        img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size)) # pad with zero on the smallest side
        img = torch.from_numpy(normalize_rgb(np.asarray(img_pil)))
        
        K, R, t = [torch.from_numpy(meta['cam'][k]) for k in ['K', 'R', 't']]
        K = adjust_K_by_ratio_padding(K[None, :, :], ratio, px, py)[0]  # (3, 3)
        
        gt_masks = torch.ones((N_MAX_HUMANS,))
        smplx_params = {k:torch.from_numpy(v).float() for k, v in meta['smplx_params'].items()}
        
        return img, K, R, t, smplx_params, gt_masks


class CHI3DSequenceMetaLoader(object):
    def __init__(self, root_path):
        self.root_path = root_path
    
    def load(self, subject, seq_name):
        K, R, t = [], [], []
        for cam_id in CAM_IDS:
            with open(os.path.join(self.root_path, 'train', subject, 'camera_parameters', cam_id, f"{seq_name}.json")) as f:
                cam_params = json.load(f)
            Kc = build_k(
                cam_params['intrinsics_wo_distortion']['f'][0],
                cam_params['intrinsics_wo_distortion']['f'][1],
                cam_params['intrinsics_wo_distortion']['c'][0],
                cam_params['intrinsics_wo_distortion']['c'][1])
            Rc = np.array(cam_params["extrinsics"]["R"])
            tc = -np.array(cam_params["extrinsics"]["T"]) @ Rc.T
            K.append(Kc)
            R.append(Rc)
            t.append(tc)
            
        K, R, t = [np.concatenate([vv[None] for vv in v], axis=0) for v in [K, R, t]]
        
        seq_len = len(os.listdir(os.path.join(self.root_path, 'train', subject, 'images', CAM_IDS[0], seq_name)))
        
        images = []
        for frame_id in range(seq_len):
            cur_images = []
            for cam_id in CAM_IDS:
                cur_images.append(
                    os.path.join(
                        self.root_path, 'train', subject, 'images', cam_id, seq_name, str(frame_id).zfill(6)+".jpg"))
                
            images.append(cur_images)
            
        return images, K, R, t