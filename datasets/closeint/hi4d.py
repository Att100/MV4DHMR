import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageOps, ImageFile
import os
from tqdm import tqdm
import pickle
import roma

from utils.image import normalize_rgb, denormalize_rgb
from utils.camera import perspective_projection
from datasets.closeint.utils import get_ratio_padding, adjust_K_by_ratio_padding

ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid "OSError: image file is truncated"

CAM_IDS = [4, 16, 40, 64, 76]
IMG_SIZE_ORG = (940, 1280)
N_MAX_HUMANS = 2
INTERVAL_TRAINING = 1
INTERVAL_TEST = 1

class Hi4D(Dataset):
    """
    Hi4D multi-view image-based dataset implementation
    
    """
    def __init__(self, root_path, cache_path, img_size, split='test', reload_metas=False, return_key=False):
        super().__init__()
        
        self.root_path = root_path
        self.cache_path = cache_path
        self.img_size = img_size
        self.split = split
        self.interval = INTERVAL_TRAINING if split=='training' else INTERVAL_TEST
        self.return_key = return_key
        
        self.image_dir = os.path.join(self.root_path, 'imageFiles')
        self.metas_dir = os.path.join(self.cache_path, f"{split}.meta.pkl")
        
        if not os.path.exists(self.metas_dir) or reload_metas:
            self.metas = self.load_metas()
            with open(self.metas_dir, 'wb') as f:
                pickle.dump(self.metas, f)
        else:
            with open(self.metas_dir, 'rb') as f:
                self.metas = pickle.load(f)
        
    @torch.no_grad()
    def load_metas(self):
        fnames = os.listdir(os.path.join(self.root_path, 'sequenceFiles', self.split))
        fnames.sort()
        
        samples_meta = []
        for fname in fnames:
            with open(os.path.join(self.root_path, 'sequenceFiles', self.split, fname), 'rb') as f:
                metadata = pickle.load(f, encoding='latin1')
            
            seq_name = fname.replace('.pkl', '')
            seq_keys = sorted(list(metadata['smpl'].keys()))
            print(f"Processing: {seq_name}")
            
            K = metadata['intrinsics']
            R = metadata['extrinsics'][:,:3,:3]
            t = metadata['extrinsics'][:,:3,-1][:, None, :]
            
            seq_metas = []
            for i in tqdm(range(len(seq_keys))):
                if (i+1) % self.interval != 0: continue
                
                k = seq_keys[i]
                n_humans = metadata['smpl'][k]['transl'].shape[0]
                
                global_orient = metadata['smpl'][k]['global_orient']
                body_pose = metadata['smpl'][k]['body_pose']
                betas = metadata['smpl'][k]['betas'][:10]
                transl = metadata['smpl'][k]['transl']
                
                seq_metas.append(dict(
                    seq_name=seq_name,
                    frame_name=k,
                    images=[os.path.join(
                        self.image_dir, self.split, seq_name, str(camid), f"{k}.jpg") for camid in CAM_IDS],
                    smpl_params=dict(
                        global_orient=global_orient.reshape(n_humans, 1, 3),
                        body_pose=body_pose.reshape(n_humans, 23, 3),
                        betas=betas,
                        transl=transl.reshape(n_humans, 3)
                    ),
                    cams=dict(
                        K=K, R=R, t=t
                    )
                ))
            
            samples_meta += seq_metas

        return samples_meta
    
    def __getitem__(self, index):
        meta = self.metas[index]
        
        ratio, px, py = get_ratio_padding(IMG_SIZE_ORG, self.img_size)
        
        imgs = []
        for img_path in meta['images']:
            img_pil = Image.open(img_path).convert('RGB')
            img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size)) # keep the same aspect ratio
            img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size)) # pad with zero on the smallest side
            resize_img = normalize_rgb(np.asarray(img_pil))
            imgs.append(torch.from_numpy(resize_img).unsqueeze(0))
        
        K, R, t = [torch.from_numpy(meta['cams'][k]) for k in ['K', 'R', 't']]
        K = adjust_K_by_ratio_padding(K, ratio, px, py)  # (n_views, 3, 3)
            
        imgs = torch.concat(imgs, dim=0)  # (n_views, 3, H, W)
        gt_masks = torch.ones((N_MAX_HUMANS,))
        smplx_params = {k:torch.from_numpy(v).float() for k, v in meta['smpl_params'].items()}
        
        if self.return_key:
            return imgs, K, R, t, smplx_params, gt_masks, meta['seq_name'], meta['frame_name']
        
        return imgs, K, R, t, smplx_params, gt_masks
    
    def __len__(self):
        return len(self.metas)


class Hi4D_SV(Hi4D):
    """
    Hi4D single-view image-based dataset implementation
    
    """
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
        smpl_params = {k:torch.from_numpy(v).float().cuda() for k,v in meta["smpl_params"].items()}
        K, R, t = [torch.from_numpy(meta['cams'][k]).float().cuda() for k in ['K', 'R', 't']]
        output = smpl_model(
            global_orient=smpl_params['global_orient'],
            body_pose=smpl_params['body_pose'],
            betas=smpl_params['betas']
        )
        mesh = output.vertices + smpl_params['transl'].unsqueeze(1)
        joints = output.joints + smpl_params['transl'].unsqueeze(1)

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
                roma.rotvec_to_rotmat(smpl_params['global_orient'][:, 0]))).unsqueeze(1)
            metas.append(dict(
                seq_name=meta['seq_name'],
                frame_name=meta['frame_name'],
                view_id=vid,
                image=meta['images'][vid],
                smpl_params=dict(
                    global_orient=global_orient.reshape(smpl_params['global_orient'].shape).cpu().numpy(),
                    body_pose=smpl_params['body_pose'].cpu().numpy(),
                    betas=smpl_params['betas'].cpu().numpy(),
                    transl=transl.reshape(smpl_params['transl'].shape).cpu().numpy()
                ),
                cam=dict(K=K[vid].cpu().numpy(), R=R[vid].cpu().numpy(), t=t[vid].cpu().numpy())
            ))
            
        return metas
    
    def __getitem__(self, index):
        meta = self.metas[index]
        
        ratio, px, py = get_ratio_padding(IMG_SIZE_ORG, self.img_size)
        
        img_pil = Image.open(meta['image']).convert('RGB')
        img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size)) # keep the same aspect ratio
        img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size)) # pad with zero on the smallest side
        resize_img = normalize_rgb(np.asarray(img_pil))
        img = torch.from_numpy(resize_img)
        
        K, R, t = [torch.from_numpy(meta['cam'][k]) for k in ['K', 'R', 't']]
        K = adjust_K_by_ratio_padding(K[None, :, :], ratio, px, py)[0]  # (3, 3)
        
        gt_masks = torch.ones((N_MAX_HUMANS,))
        smplx_params = {k:torch.from_numpy(v).float() for k, v in meta['smpl_params'].items()}
        
        return img, K, R, t, smplx_params, gt_masks