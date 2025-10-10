import torch
import torch.nn as nn
import numpy as np
import roma
import random
from sklearn.cluster import KMeans

from utils import perspective_projection


def world2cam(pts, R, t):
    return torch.matmul(R, pts.transpose(-1, -2)).transpose(-1, -2) + t

def cam2world(pts, R, t):
    return torch.matmul(R.transpose(-1, -2), (pts - t).transpose(-1, -2)).transpose(-1, -2)

class RefineWithMultiView(nn.Module):
    def __init__(self, 
                 n_views=5, 
                 n_max_humans=2, 
                 recenter=False, 
                 skip_single_matched=False, 
                 pelvis_idx=0, 
                 smplx_type="smplx",
                 select_best_view=True):
        super().__init__()
        
        self.n_views = n_views
        self.n_max_humans = n_max_humans
        self.recenter = recenter  # recenter the joints on pelvis before K-means matching
        self.pelvis = pelvis_idx
        self.skip_single_matched = skip_single_matched  # will ignore humans with only one matched view
        self.smplx_type = smplx_type
        self.select_best_view = select_best_view
    
    @torch.no_grad()
    def forward(self, j3ds, j2ds, mask, smpl_params, K, R, t):
        """
        j3ds: (n_views, n_max_humans, N, 3)
        j2ds: (n_views, n_max_humans, N, 2)
        mask: (n_views, n_max_humans)
        K: (n_views, 3, 3)
        R: (n_views, 3, 3)
        t: (n_views, 1, 3)
        """
        
        N = j3ds.shape[2]
        
        Ks, Rs, ts = [v.unsqueeze(1).repeat(1, self.n_max_humans, 1, 1) for v in [K, R, t]]
        mask = mask.reshape(-1).bool()
        
        j3ds_w = cam2world(j3ds, Rs, ts).reshape(-1, N, 3)
        
        # clustering (simple cross-view matching)
        if self.recenter:
            j3ds_wrc = j3ds_w - j3ds_w[:, self.pelvis:self.pelvis+1, :]
        else: j3ds_wrc = j3ds_w
        j3ds_wrc = j3ds_wrc.reshape(self.n_views * self.n_max_humans, -1)[mask].cpu().numpy()
        labels_matched = torch.from_numpy(self.cluster(j3ds_wrc)).to(j3ds.device)
        
        # get human with only one view observed (skip transl/rot calibration or just ignore this human)
        vals, counts = torch.unique(labels_matched, return_counts=True, sorted=True)
        single_observed = [skip_id.item() for skip_id, count in zip(vals, counts) if count == 1]
        
        # triangulation
        j2ds = [j2ds.reshape(-1, N, 2)[mask][labels_matched==i] for i in range(self.n_max_humans)]
        Ks = [Ks.reshape(-1, 3, 3)[mask][labels_matched==i] for i in range(self.n_max_humans)]
        Rs = [Rs.reshape(-1, 3, 3)[mask][labels_matched==i] for i in range(self.n_max_humans)]
        ts = [ts.reshape(-1, 1, 3)[mask][labels_matched==i] for i in range(self.n_max_humans)]
            
        pelvis_triangulated = torch.stack([self.triangulate(
            j2ds[i][:, self.pelvis, :], Ks[i], Rs[i], ts[i]
        ) for i in range(self.n_max_humans)])  # (n_max_humans, 3)
        
        # calibration
        humans_calibrated = []
        for i in range(self.n_max_humans):
            if i not in single_observed:  # have more than one observations
                if self.select_best_view:
                    # best view selection:  
                    #   -> argmin(mpjpe(Proj(j3d, K), j2d))
                    _j3ds_w = j3ds_w[mask][labels_matched==i]
                    _j3ds_w = _j3ds_w - _j3ds_w[:, self.pelvis:self.pelvis+1, :] + \
                        pelvis_triangulated[i][None, None, :]  # translation calibrated
                    _j3ds_c = world2cam(
                        _j3ds_w.unsqueeze(1).repeat(1, _j3ds_w.shape[0], 1, 1),
                        Rs[i].unsqueeze(0).repeat(_j3ds_w.shape[0], 1, 1, 1),
                        ts[i].unsqueeze(0).repeat(_j3ds_w.shape[0], 1, 1, 1)
                    )  # (n_observations, n_observations, N, 3)
                    _j3ds_proj = perspective_projection(
                        _j3ds_c.reshape(-1, N, 3),
                        Ks[i].unsqueeze(0).repeat(_j3ds_w.shape[0], 1, 1, 1).reshape(-1, 3, 3),
                    ).reshape(list(_j3ds_c.shape[:3])+[2])
                    mpjpe = (torch.sqrt(((_j3ds_proj - j2ds[i].unsqueeze(0)) ** 2).sum(-1))).mean(-1)  # (n_observations, n_observations)
                    best_view = torch.argmin(mpjpe.mean(-1))
                else:
                    best_view = random.randint(0, j2ds[i].shape[0]-1)
                
                rotvec = smpl_params['rotvec'].reshape(self.n_views * self.n_max_humans, -1, 3)
                rotvec = rotvec[mask][labels_matched==i][best_view]
                betas = smpl_params['shape'].reshape(-1, 10)[mask][labels_matched==i][best_view]
                global_orient = Rs[i][best_view].T @ roma.rotvec_to_rotmat(rotvec[0])  # cam2world
                global_orient_rotvec = roma.rotmat_to_rotvec(global_orient)
                
                # smpl parameters calibration (final)
                if self.smplx_type == "smpl":
                    humans_calibrated.append(dict(
                        global_orient=global_orient_rotvec[None, :],
                        body_pose=rotvec[1:],
                        betas=betas,
                        transl=pelvis_triangulated[i]  # real pelvis transl, not smpl transl
                    ))
                elif self.smplx_type == "smplx":
                    humans_calibrated.append(dict(
                        global_orient=global_orient_rotvec[None, :],
                        body_pose=rotvec[1:22],
                        left_hand_pose=rotvec[22:37],
                        right_hand_pose=rotvec[37:52],
                        jaw_pose=rotvec[52:53],
                        betas=betas,
                        transl=pelvis_triangulated[i]  # real pelvis transl, not smpl transl
                    ))
            else:
                if self.skip_single_matched: continue
                else:
                    # use original prediction
                    rotvec = smpl_params['rotvec'].reshape(self.n_views * self.n_max_humans, -1, 3)
                    rotvec = rotvec[mask][labels_matched==i][0]
                    betas = smpl_params['shape'].reshape(-1, 10)[mask][labels_matched==i][0]
                    global_orient = Rs[i][0].T @ roma.rotvec_to_rotmat(rotvec[0])  # cam2world
                    global_orient_rotvec = roma.rotmat_to_rotvec(global_orient)
                    
                    if self.smplx_type == "smpl":
                        humans_calibrated.append(dict(
                            global_orient=global_orient_rotvec[None, :],
                            body_pose=rotvec[1:],
                            betas=betas,
                            transl=j3ds_w[mask][labels_matched==i][0, self.pelvis, :]
                        ))
                    elif self.smplx_type == "smplx":
                        humans_calibrated.append(dict(
                            global_orient=global_orient_rotvec[None, :],
                            body_pose=rotvec[1:22],
                            left_hand_pose=rotvec[22:37],
                            right_hand_pose=rotvec[37:52],
                            jaw_pose=rotvec[52:53],
                            betas=betas,
                            transl=j3ds_w[mask][labels_matched==i][0, self.pelvis, :]
                        ))
            
        return humans_calibrated, labels_matched
            
    def cluster(self, j3ds_flatten):        
        kmeans = KMeans(n_clusters=self.n_max_humans, random_state=42)
        labels = kmeans.fit_predict(j3ds_flatten)
        return labels
    
    def triangulate(self, points_2d, K, R, t):
        """
        points_2d: (m, 2)
        K: (m, 3, 3)
        R: (m, 3, 3)
        t: (m, 1, 3)
        return: point_3d (3,)
        """
        m = points_2d.shape[0]

        A = []

        for i in range(m):
            Rt = torch.cat([R[i], t[i][0].unsqueeze(1)], dim=1)  # (3, 4)
            P = K[i] @ Rt  # (3, 4)

            x, y = points_2d[i]
            A.append(x * P[2] - P[0])  # x * P3 - P1
            A.append(y * P[2] - P[1])  # y * P3 - P2

        A = torch.stack(A)  # (2m, 4)

        _, _, V = torch.svd(A)
        X_h = V[:, -1]  
        X = X_h[:3] / X_h[3] 

        return X  # shape: (3,)
    
    def postprocess_refined(self, smpl_model, humans_initialized):
        smpl_params = {k:[] for k in humans_initialized[0].keys() if k != 'transl'}
        for human in humans_initialized:
            for k in smpl_params.keys():
                smpl_params[k].append(human[k].unsqueeze(0))
        for k, v in smpl_params.items():
            smpl_params[k] = torch.concat(v, dim=0)
        if self.smplx_type == "smplx":
            smpl_params['leye_pose'] = smpl_model.leye_pose.repeat(self.n_max_humans, 1)
            smpl_params['reye_pose'] = smpl_model.reye_pose.repeat(self.n_max_humans, 1)  
            smpl_params['expression'] = smpl_model.expression.repeat(self.n_max_humans, 1)
        transl = torch.concat([h['transl'][None, None, :] for h in humans_initialized], dim=0)
        humans = smpl_model(**smpl_params)
        
        smpl_params['transl'] = transl - humans.joints[:, 0:1, :]
        
        vertices = humans.vertices - humans.joints[:, 0:1, :] + transl
        joints = humans.joints - humans.joints[:, 0:1, :] + transl
        
        return smpl_params, vertices, joints
        
        