import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import smplx
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import subprocess

from model_org import Model as ModelOrg
from model import Model
from multiview.refine import RefineWithMultiView
from multiview.tracking import Sort
from multiview.smooth import OneEuroFilter
from datasets.closeint.utils import get_ratio_padding, \
    adjust_K_by_ratio_padding
from utils.image import normalize_rgb, denormalize_rgb
from utils.tensor_manip import recursive_apply
from utils.render import render_meshes
from utils.camera import perspective_projection


CAM_VIEWS_NUM = {
    'chi3d': 4,
    'hi4d': 5,
    'panoptic': 5,
    'shelf': 5
}

colors_2humans = [(0.8, 0.144, 0.610), (0.042, 0.797, 0.906)]

class Colors:
    def __init__(self, seed=1):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def colors(self, n):
        return [tuple(self.rng.random(3)) for _ in range(n)]

def overlay_human_meshes(humans, K, faces, img, colors=None, alpha=0.8, seed_color=42):
    # Color of humans seen in the image.
    if colors is None:
        colors = Colors(seed_color).colors(len(humans))
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0, 0], K[1, 1]])
    princpt = np.asarray([K[0, -1], K[1, -1]])

    # Get the vertices produced by the model.
    verts_list = [humans[j] for j in range(len(humans))]
    faces_list = [faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.zeros_like(img), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=colors)
    
    non_human_mask = pred_rend_array <= 0
    img_masked = img * non_human_mask + img * (1-non_human_mask) * (1-alpha)
    img_overlay = img_masked + alpha * pred_rend_array
    
    return img_overlay.astype(np.uint8)

def postprocess_pred(pred, idx, bs, n_humans=2):
    sid, counts = torch.unique(idx[0], sorted=True, return_counts=True)
    n_max_human_det = n_humans if counts.max() < n_humans else counts.max()
    
    idx_human = torch.concat([torch.arange(i) for i in counts])
    idx_flatten = torch.arange(0, pred['v3d'].shape[0], device=idx[0].device, dtype=torch.long)
        
    pred_out = dict()
    for k in ['v3d', 'j3d', 'v2d', 'j2d', 'rotvec', 'shape', 'expression']:
        pred_out[k] = torch.zeros([bs, n_max_human_det]+list(pred[k].shape)[1:], device=idx[0].device)
        pred_out[k][idx[0], idx_human] = pred[k][idx_flatten].float()
        pred_out[k] = pred_out[k][:, :n_humans]
    
    pred_mask = torch.zeros(bs, n_max_human_det, device=idx[0].device)
    pred_mask[idx[0], idx_human] = 1
    pred_out['mask'] = pred_mask[:, :n_humans]
    return pred_out

def interpolate_missing_frames(latest_payload, cur_payload):
    # simple linear interpolation
    frame_id_latest, payload_latest = latest_payload
    frame_id_cur, payload_cur = cur_payload
    
    frames_to_interp = frame_id_cur - frame_id_latest - 1
    payload_interpoated = []
    
    for i in range(frames_to_interp):
        alpha = (i + 1) / (frames_to_interp + 1)
        payload_interp = dict(
            pred_v3d=(1 - alpha) * payload_latest['pred_v3d'] + alpha * payload_cur['pred_v3d'],
            pred_j3d=(1 - alpha) * payload_latest['pred_j3d'] + alpha * payload_cur['pred_j3d'],
            smplx_params={
                k:(1 - alpha) * v + alpha * payload_cur['smplx_params'][k] \
                    for k, v in payload_latest['smplx_params'].items()}
        )
        payload_interpoated.append((frame_id_latest + i + 1, payload_interp))
        
    return payload_interpoated

def images_to_video(
        image_dir,
        output_path,
        fps=30,
        image_format="png",
        pattern_type="sequence"
    ):

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} not found")

    if pattern_type == "sequence":
        input_pattern = os.path.join(image_dir, f"%08d.{image_format}")
    else:
        input_pattern = os.path.join(image_dir, f"*.{image_format}")

    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-pattern_type", pattern_type,
        "-i", input_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-y",
        output_path
    ]

    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


class Pipeline(object):
    def __init__(self, args, device=torch.device('cuda:0')):
        super().__init__()
        
        self.args = args
        self.device = device
        self.n_views = args.n_views if args.n_views > 0 else CAM_VIEWS_NUM[args.dataset]
        self.in_img_size = [int(v) for v in args.image_size_org.split(",")]
        self.img_transform = get_ratio_padding(self.in_img_size, self.args.image_size)
        
        model_class = Model if args.eval_mode=='finetune' else ModelOrg
        self.hmr_model = model_class(
            backbone=args.backbone, 
            img_size=args.image_size,
            smplx_type=args.smplx_type).to(device)
        if args.eval_mode == 'zeroshot':
            ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=False)['model_state_dict']
        else:
            ckpt = torch.load(args.checkpoint_path, map_location=device)
        self.hmr_model.load_state_dict(ckpt, strict=False)
        self.hmr_model.eval()
        
        self.smpl_model = smplx.create(
            args.smplx_dir, "smpl", use_pca=False, flat_hand_mean=True, gender='neutral', num_betas=10).to(device)
        self.smplx_model = smplx.create(
            args.smplx_dir, "smplx", use_pca=False, flat_hand_mean=True, gender='neutral', num_betas=10).to(device)
        
        self.refine_model = RefineWithMultiView(self.n_views, args.n_humans, args.recenter, smplx_type=args.smplx_type)
        self.tracker = Sort(max_age=5)
        self.smoothers = dict()
        
        self.counter = -1
        self.tracking = defaultdict(list)
        self.tracking_inv_index = []
        self.cache = []
        
        self.colors = Colors().colors(max(args.n_humans, 50))

    @torch.no_grad()
    def forward_hmr(self, imgs, K):
        """
        imgs: (n_views, 3, H, W)
        K: (n_views, 3, 3)
        """

        with torch.amp.autocast('cuda', enabled=True):
            output = self.hmr_model(imgs, K=K, is_training=False, return_list=False)
            
        if output == []:
            print("No Human Detected in all views")
            return None
        else:
            pred, idx = output
        
        pred = postprocess_pred(pred, idx, bs=self.n_views, n_humans=self.args.n_humans)
        
        return pred

    @torch.no_grad()
    def forward_refine(self, pred, K, R, t):
        try:
            humans_refined, labels_matched = self.refine_model(
                pred['j3d'], pred['j2d'], pred['mask'], pred, K, R, t)
            smplx_params, pred_v3d, pred_j3d = self.refine_model.postprocess_refined(
                self.smplx_model if self.args.smplx_type=="smplx" else self.smpl_model, 
                humans_refined)
            return smplx_params, pred_v3d, pred_j3d, labels_matched
        except Exception as e:
            print(e)
            print("Possible cause: only one human detected in all views")
            
            return None, None, None, None
    
    def forward_tracking(self, smplx_params, pred_v3d, pred_j3d):
        assert self.counter >= 0
        
        payload = [
            recursive_apply(dict(
                smplx_params=smplx_params, 
                pred_v3d=pred_v3d,
                pred_j3d=pred_j3d), lambda x:x[i].cpu().numpy()) for i in range(len(pred_j3d))]
        tracks = self.tracker.update(pred_j3d.cpu().numpy(), payload)
        
        inv_index = []
        for trk in tracks:
            trk_id = int(trk[0][0, -1])
            payload = trk[1]
            
            if self.args.auto_interpolate:
                if trk_id in self.tracking.keys():
                    if 0 < self.counter - self.tracking[trk_id][-1][0] - 1 <= 5:
                        payloads_interp = interpolate_missing_frames(
                            self.tracking[trk_id][-1], (self.counter, payload))
                        cur_frames = len(self.tracking[trk_id])
                        for invi_offset, (fid_interp, payload_interp) in enumerate(payloads_interp):
                            self.tracking_inv_index[fid_interp].append((trk_id, cur_frames + invi_offset))
                        self.tracking[trk_id].extend(payloads_interp)
                            
            self.tracking[trk_id].append((self.counter, payload))
            inv_index.append((trk_id, len(self.tracking[trk_id])-1))
        
        self.tracking_inv_index.append(inv_index)
        
        return tracks
            
    def forward_smoother(self):
        smoother = dict()
        smoothed_tracking = defaultdict(list)
        
        for trk_id, frames in self.tracking.items():
            if trk_id not in smoother.keys():
                smoother[trk_id] = [
                    OneEuroFilter(freq=30, min_cutoff=0.001, beta=0.9),
                    OneEuroFilter(freq=30, min_cutoff=0.001, beta=0.9)
                ]
            
            cur_frame_id = -1
            for frame_id, payload in frames:
                if cur_frame_id >=0 and frame_id - cur_frame_id > 1:
                    # reset smoother
                    smoother[trk_id] = [
                        OneEuroFilter(freq=30, min_cutoff=0.001, beta=0.9),
                        OneEuroFilter(freq=30, min_cutoff=0.001, beta=0.9)
                    ]
                
                # smooth v3d and j3d only
                payload['pred_v3d'] = smoother[trk_id][0](payload['pred_v3d'])
                payload['pred_j3d'] = smoother[trk_id][1](payload['pred_j3d'])
                smoothed_tracking[trk_id].append((frame_id, payload))
                
                cur_frame_id = frame_id
                
        self.tracking = smoothed_tracking
        
    def preprocess_input(self, imgs_path, K, R, t, cache=True):
        if cache: self.cache.append((imgs_path, K, R, t))
        
        K, R, t = [
            torch.from_numpy(v).float().to(self.device) \
                for v in [K, R, t]]
        
        ratio, px, py = self.img_transform
        image_size = self.args.image_size
        imgs_input = []
        
        for img_path in imgs_path:
            img_pil = Image.open(img_path).convert('RGB')
            img_pil = ImageOps.contain(img_pil, (image_size, image_size)) # keep the same aspect ratio
            img_pil = ImageOps.pad(img_pil, size=(image_size, image_size)) # pad with zero on the smallest side
            img = torch.from_numpy(normalize_rgb(np.asarray(img_pil))).float()
            imgs_input.append(img.unsqueeze(0))
            
        imgs_input = torch.concat(imgs_input, dim=0).to(self.device)
        K = adjust_K_by_ratio_padding(K, ratio, px, py)
        
        return imgs_input, K, R, t
    
    def visuaize_frame(self, frame_id, imgs, tracks, K, R, t, color_mapping=None, view_id=0):
        assert self.args.vis_mode in ['render', 'scatter']
        
        img = denormalize_rgb(imgs[view_id].cpu().numpy())
        h, w = img.shape[:2]

        Kc, Rc, tc = [v[view_id].cpu().numpy() for v in [K, R, t]]
        
        if self.args.vis_mode == 'scatter':
            plt.imshow(img)
            plt.axis('off')
        
            for trk_id, payload in tracks:            
                v3d = payload['pred_v3d']
                v3d_cam = (Rc @ v3d.T).T + tc
                v2d = perspective_projection(
                    torch.from_numpy(v3d_cam)[None],
                    torch.from_numpy(Kc)[None]
                )[0]
                
                # filter v2d
                mask = (v2d[:, 0] >= 0) & (v2d[:, 0] < w) & \
                    (v2d[:, 1] >= 0) & (v2d[:, 1] < h)
                
                if color_mapping is not None:
                    plt.scatter(v2d[mask, 0], v2d[mask, 1], s=0.1, color=color_mapping[trk_id])
                else:
                    plt.scatter(v2d[mask, 0], v2d[mask, 1], s=0.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.args.save_dir, 
                'visualization', 
                f"{frame_id:08d}.png"),
                bbox_inches='tight')
            plt.close()
            
        else: # render
            humans = []
            colors = []
            for trk_id, payload in tracks:            
                v3d = payload['pred_v3d']
                v3d_cam = (Rc @ v3d.T).T + tc
                humans.append(v3d_cam)
                colors.append(color_mapping[trk_id])
            
            fig = overlay_human_meshes(
                humans, Kc, 
                (self.smplx_model if self.args.smplx_type=="smplx" else self.smpl_model).faces,
                img, colors)
            
            Image.fromarray(fig).save(
                os.path.join(
                    self.args.save_dir, 
                    'visualization', 
                    f"{frame_id:08d}.png"))
        
    def step(self, imgs, K, R, t):
        """

        Args:
            imgs (np.ndarray): input images, [n_views, H, W, 3]
            K (np.ndarray): [n_views, 3, 3]
            R (np.ndarray): [n_views, 3, 3]
            t (np.ndarray): [n_views, 3, 1, 3]
        """
        
        self.counter += 1
        imgs, K, R, t = self.preprocess_input(imgs, K, R, t)
        
        output_hmr = self.forward_hmr(imgs, K)
        if output_hmr is None:
            return None

        smplx_params, pred_v3d, pred_j3d, labels_matched = self.forward_refine(
            output_hmr, K, R, t)
        if smplx_params is None:
            return None
        
        tracks = self.forward_tracking(smplx_params, pred_v3d, pred_j3d)
        
        self.visuaize_frame(self.counter, imgs, tracks, K, R, t)
        
        return tracks
        
    def finalize(self):
        if self.args.smooth:
            self.forward_smoother()
            
        if self.args.vis:
            print(f"Processing visualization ...")
            trk_ids = sorted(list(self.tracking.keys()))
            color_mapping = {k:v for k, v in zip(trk_ids, self.colors[:len(trk_ids)])}
            
            for frame_id in tqdm(range(self.counter+1)):
                imgs_path, K, R, t = self.cache[frame_id]
                imgs_input, K, R, t = self.preprocess_input(imgs_path, K, R, t, cache=False)
                
                tracks = []
                for trk_id, invi in self.tracking_inv_index[frame_id]:
                    payload = self.tracking[trk_id][invi][1]
                    tracks.append((trk_id, payload))
                    
                self.visuaize_frame(frame_id, imgs_input, tracks, K, R, t, color_mapping)
            
            vis_path = os.path.join(self.args.save_dir,"visualization")
            print(f"Visualization saved to '{vis_path}'")
                
        if self.args.vid:
            images_to_video(
                os.path.join(self.args.save_dir, "visualization"),
                os.path.join(self.args.save_dir, "output.mp4"),
                fps=30,
                image_format="png",
                pattern_type="sequence"
            )
            
        if self.args.save:
            self.export()
        
        return self.tracking
        
    def export(self):
        """Export the tracked SMPL/SMPL-X parameters to a pickle file.
        
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
        for frame_id in range(n_frames):
            for trk_id, invi in inverse_index[frame_id]:
                smplx_params = tracking[trk_id][invi][1]
                    
        """
        smplx_to_export = dict(
            tracking=defaultdict(list),
            inverse_index=self.tracking_inv_index,
            smplx_type=self.args.smplx_type,
        )
        
        for trk_id, frames in self.tracking.items():
            for frame_id, payload in frames:
                smplx_to_export['tracking'][trk_id].append((
                    frame_id, payload['smplx_params']
                ))
        
        export_path = os.path.join(self.args.save_dir, "smplx", "sequence_smplx.pkl")
        with open(export_path, "wb") as f:
            pickle.dump(smplx_to_export, f)
        
        print(f"Tracked SMPL/SMPL-X sequence saved to '{export_path}'")
            
            
def postprocess_exported(smplx_exported, smplx_model, n_humans):
    """
    Tracking results is not always accurate. In some cases, there are 
    N human in scene, but more than N humans are tracked. This function 
    will merge tracking that link to the same human
    """
    
    tracking = smplx_exported['tracking']
    inverse_index = smplx_exported['inverse_index']
    