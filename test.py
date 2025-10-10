import torch
import smplx
import os
import roma
from tqdm import tqdm
import numpy as np
import pickle
import argparse
from torch.utils.data import DataLoader

from datasets.closeint.hi4d import Hi4D
from datasets.closeint.chi3d import CHI3D
from utils.humans import get_smplx_joint_names
from utils.training import AverageMeter, match_greedy, compute_prf1
from model_org import Model as ModelOrg
from model import Model
from multiview.refine import RefineWithMultiView

JOINT_INDEX_HEAD = get_smplx_joint_names().index("head")
JOINT_INDEX_PELVIS = get_smplx_joint_names().index("pelvis")
KEYS_FOR_TRAIN = [
    'v3d', 'j3d', 'v2d', 'j2d', 'transl', 'transl_pelvis', 
    'loc', 'rotmat', 'shape']
IMG_SIZE = 672
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='model_small_672_chi3d_sv')
    
    # Path / Directory args
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/pretrained/multiHMR_672_S.pt')
    parser.add_argument('--smplx_dir', type=str, default='./checkpoints')
    parser.add_argument('--smplx2smpl_path', type=str, default='./checkpoints/smplx/smplx2smpl.pkl')
    parser.add_argument('--j_regressor_h36m_path', type=str, default='./checkpoints/J_regressor_h36m.npy')
    parser.add_argument('--dataset_dir', type=str, default='/ocean/projects/cis240055p/czheng3/chi3d')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/saved')
    
    # Model args
    parser.add_argument('--backbone', type=str, default='dinov2_vits14')
    parser.add_argument("--smplx_type", type=str, default='smplx', choices=['smpl', 'smplx'])
    parser.add_argument('--image_size', type=int, default=672, choices=[672, 896])
    
    # Dataset args
    parser.add_argument('--dataset', type=str, default='chi3d', choices=['chi3d', 'hi4d'])
    
    # Test args
    parser.add_argument('--eval_mode', type=str, default='zeroshot', choices=['zeroshot', 'finetune'])
    parser.add_argument('--recenter', action='store_true', help='recenter the pelvis of each human to origin')
    parser.add_argument('--n_humans', type=int, default=2)
    parser.add_argument('--n_views', type=int, default=-1)  # -1 for all views
    parser.add_argument('--data_smplx_type', type=str, default='smplx')
    parser.add_argument('--sample_interval', type=int, default=25)
    parser.add_argument('--device', type=str, default='0')
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(0)
    exp_name = args.exp_name
    n_views = args.n_views if args.n_views > 0 else (4 if args.dataset=='chi3d' else 5)
    dataset_class = CHI3D if args.dataset == 'chi3d' else Hi4D
    model_class = ModelOrg if args.eval_mode == 'zeroshot' else Model
    
    with open(args.smplx2smpl_path, 'rb') as f:
        smplx2smpl_regressor = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(device)
    j_regressor_h36m = torch.Tensor(np.load(args.j_regressor_h36m_path)).to(device)
    
    smpl_model = smplx.create(
        args.smplx_dir, "smpl", use_pca=False, flat_hand_mean=True, gender='neutral', num_betas=10).to(device)
    smplx_model = smplx.create(
        args.smplx_dir, "smplx", use_pca=False, flat_hand_mean=True, gender='neutral', num_betas=10).to(device)
    
    test_dataset = dataset_class(
        root_path=args.dataset_dir,
        cache_path=f"./data/{args.dataset}",
        img_size=args.image_size,
        split='test')
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    
    model = model_class(
                backbone=args.backbone, 
                img_size=args.image_size,
                smplx_type=args.smplx_type).to(device)
    if args.eval_mode == 'zeroshot':
        ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=False)['model_state_dict']
    else:
        ckpt = torch.load(os.path.join(args.save_dir, f"{exp_name}.pt"), map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    # Our Multi-view approach
    model_refine = RefineWithMultiView(
        n_views, 
        args.n_humans,
        smplx_type=args.smplx_type, 
        recenter=args.recenter).to(device)
    
    # Evaluation
    count, miss, fp = 0, 0, 0
    meters = {k: AverageMeter(k) for k in ['pve', 'pa_pve', 'mpjpe', 'pa_mpjpe', 'transl_pelvis']}
    with tqdm(total=len(test_loader), desc=f'[test]', unit='batch') as pbar:
        with torch.no_grad():
            for i, sample in enumerate(test_loader):                
                imgs, K, R, t, smplx_params, gt_masks = sample
                
                # input
                imgs = imgs[0, :n_views].to(device)
                K, R, t = [v.to(device).float()[0, :n_views] for v in [K, R, t]]
                
                # pre-process gt
                gt_masks = gt_masks.to(device)
                gt_smplx_params = {k:v.to(device).float()[0] for k, v in smplx_params.items()}
                gt_humans = (smpl_model if args.data_smplx_type=="smpl" else smplx_model)(**gt_smplx_params)
                gt_v3d, gt_j3d = gt_humans.vertices, gt_humans.joints
                
                # multiview
                with torch.amp.autocast('cuda', enabled=True):
                    output = model(imgs, K=K, is_training=False, return_list=False)
                    
                if output == []:
                    print("No Human Detected")
                    count += float(gt_masks.sum())
                    miss += float(gt_masks.sum())
                    pbar.update(1)
                    continue
                else:
                    pred, idx = output
                
                try:
                    pred = postprocess_pred(pred, idx, bs=n_views, n_humans=2)
                    humans_refined, labels_matched = model_refine(
                        pred['j3d'], pred['j2d'], pred['mask'], pred, K, R, t)
                    smpl_params, pred_v3d, pred_j3d = model_refine.postprocess_refined(
                        smplx_model if args.smplx_type=="smplx" else smpl_model, 
                        humans_refined)
                except Exception as e:
                    print("Only one Human Detected")
                    count += float(gt_masks.sum())
                    miss += float(gt_masks.sum())
                    pbar.update(1)
                    continue
                
                if args.smplx_type == "smplx":
                    pred_v3d = smplx2smpl_regressor @ pred_v3d
                if args.data_smplx_type == "smplx":
                    gt_v3d = smplx2smpl_regressor @ gt_v3d
                
                # evaluation
                bestMatch, falsePositives, misses = match_greedy(
                    pred_v3d.cpu().numpy(), gt_v3d.cpu().numpy(), 
                    np.ones(gt_v3d.shape[:2]).astype(np.bool_),
                    500,
                    metric='pve')
                
                count += float(gt_masks.sum())
                miss += len(misses)
                fp += len(falsePositives)
                
                if len(bestMatch) > 0:
                    for (pid, gid) in bestMatch:
                        # gt/pred vertices centerex around pelvis
                        v3d_gt_ctx = gt_v3d[gid] - gt_j3d[gid, 0:1]
                        v3d_pred_ctx = pred_v3d[pid] - pred_j3d[pid, 0:1]
                        
                        j3d_gt_ctx = (j_regressor_h36m @ v3d_gt_ctx)[H36M_TO_J14, :]
                        j3d_pred_ctx = (j_regressor_h36m @ v3d_pred_ctx)[H36M_TO_J14, :]
                            
                        # pelvis (root) translation
                        transl_err = torch.sqrt((((gt_j3d[gid, 0:1] - pred_j3d[pid, 0:1]) * 1000)**2).sum())
                        meters['transl_pelvis'].update(transl_err.item())  # mm
                        
                        # MPJPE
                        mpjpe = ((torch.sqrt(((j3d_gt_ctx - j3d_pred_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['mpjpe'].update(mpjpe.item())

                        # PA-MPJPE
                        (R2,t2,s2) = roma.rigid_points_registration(j3d_pred_ctx, j3d_gt_ctx , compute_scaling=True)
                        pa_j3d_pred_ctx = s2 * (R2.reshape(1,3,3) @ j3d_pred_ctx.reshape(-1,3,1)).reshape(-1,3) + t2
                        pa_mpjpe = ((torch.sqrt(((j3d_gt_ctx - pa_j3d_pred_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['pa_mpjpe'].update(pa_mpjpe.item())
                        
                        # PVE
                        pve = ((torch.sqrt(((v3d_gt_ctx - v3d_pred_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['pve'].update(pve.item())
                        
                        # PA-PVE
                        (R2, t2, s) = roma.rigid_points_registration(v3d_pred_ctx, v3d_gt_ctx, compute_scaling=True)
                        pa_v3d_pred_ctx = s * (R2.reshape(1, 3, 3) @ v3d_pred_ctx.reshape(-1, 3, 1)).reshape(-1, 3) + t2
                        pa_pve = ((torch.sqrt(((v3d_gt_ctx - pa_v3d_pred_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['pa_pve'].update(pa_pve.item())
                        
                pbar.set_postfix({
                    'mpjpe': meters['mpjpe'].avg,
                    'pa-mpjpe': meters['pa_mpjpe'].avg,
                    'pve': meters['pve'].avg,
                    'pa-pve': meters['pa_pve'].avg,
                    'transl-pelvis': meters['transl_pelvis'].avg})
                
                pbar.update(1)
                
    precision, recall, f1_score = compute_prf1(count, miss, fp)

    results = dict(
        mpjpe=meters['mpjpe'].avg, pa_mpjpe=meters['pa_mpjpe'].avg,
        pve=meters['pve'].avg, pa_pve=meters['pa_pve'].avg,
        transl_pelvis=meters['transl_pelvis'].avg,
        precision=precision, recall=recall, f1_score=f1_score)

    for k, v in results.items():
        print(f"{k}: {v}")

                
