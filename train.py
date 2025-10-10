import torch
import smplx
import os
import random
import roma
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse

from datasets.closeint.hi4d import Hi4D_SV
from datasets.closeint.chi3d import CHI3D_SV

from utils.camera import perspective_projection, focal_length_normalization, log_depth
from utils.humans import get_smplx_joint_names
from utils.image import denormalize_rgb
from utils.training import AverageMeter
from model import Model
from loss import Loss

JOINT_INDEX_HEAD = get_smplx_joint_names().index("head")
JOINT_INDEX_PELVIS = get_smplx_joint_names().index("pelvis")
KEYS_FOR_TRAIN = [
    'v3d', 'j3d', 'v2d', 'j2d', 'transl', 'transl_pelvis', 
    'loc', 'rotmat', 'shape']
IMG_SIZE = 672
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_det_gt(j2d, mask, image_size, patch_size, device=torch.device("cuda:0")):
    # j3d: (bs, n_max_humans, N, 3) [camera coords]
    # mask: (bs, n_max_humans)
    # K: (bs, 3, 3)
    # R: (bs, 3, 3)
    # t: (bs, 1, 3)
    
    bs, n_max_humans = mask.shape
    n_patch = image_size // patch_size
    
    mask = mask.view(-1, n_max_humans).clone()
    
    pk_loc = j2d[:, :, JOINT_INDEX_HEAD, :].view(-1, 2)
    pk_mask = (pk_loc[:, 0] < image_size) * (pk_loc[:, 1] < image_size) * \
        (pk_loc[:, 0] >= 0) * (pk_loc[:, 1] >= 0)  # ignore projection outside image
    mask = (mask.view(-1) * pk_mask).view(-1, n_max_humans).bool()

    idx_h = torch.where(mask) # tuple of lenght=2
    nhv = int(mask.sum())

    pk_loc = pk_loc[mask.flatten(), :]
    pk_coarse_loc = (pk_loc // patch_size).int() # (nhv,2)
    pk_idx = torch.clamp(pk_coarse_loc, 0, n_patch - 1) # (nhv,2)
    pk_offset = (pk_loc - (pk_idx + 0.5) * patch_size) / patch_size # normalize from -0.5 to 0.5 from the center of the patch
        
    # Scores & updating valid_humans according to occlusion - wap X and Y for scores only
    gt_det = dict()
    scores = torch.zeros((bs, n_patch, n_patch)).to(device)
    visible_humans = torch.ones(nhv).to(device) # by default no occlusion so all visible

    for k in range(nhv):
        i = int(idx_h[0][k]) # index of the batch size
        j = int(idx_h[1][k]) # index of the human in this image
        _x = pk_idx[k, 1]
        _y = pk_idx[k, 0]
        if scores[i, _x, _y] == 1:
            mask[i, j] = 0
            visible_humans[k] = 0
        else:
            scores[i, _x, _y] = 1
    gt_det['loc'] = pk_loc
    gt_det['offset'] = pk_offset
   
    # Update with visibility indice
    idx_vis = torch.where(visible_humans)[0]

    gt_det['idx'] = tuple([
        idx_h[0].to(device)[idx_vis],
        pk_idx[:,1].to(device)[idx_vis], 
        pk_idx[:,0].to(device)[idx_vis],
        torch.zeros_like(idx_h[0].to(device)[idx_vis]) # to match the size of the forward model
        ])
    gt_det['scores'] = scores # [bs, patch_size, patch_size]
    gt_det['offset'] = gt_det['offset'][idx_vis].clone().detach()
    gt_det['idx_vis'] = idx_vis
    
    return gt_det, mask

def preprocess_gt(
        smplx_params,
        gt_masks,
        K, R, t,
        smplx_model, 
        image_size, 
        patch_size, 
        device=torch.device("cuda:0"),
        apply_gt_masks=True, 
        smplx_type="smplx"):
    
    bs, n_humans = gt_masks.shape

    humans = smplx_model(**{k:v.view((-1,)+v.shape[2:]) for k, v in smplx_params.items()})
    
    # 1. smpl/smplx parameters; joints 2D/3D; vertices 2D/3D
    gt = dict(shape=smplx_params["betas"])
    if smplx_type == "smplx":
        gt['rotvec']=torch.concat([
            smplx_params["global_orient"],
            smplx_params["body_pose"],
            smplx_params["left_hand_pose"],
            smplx_params["right_hand_pose"],
            smplx_params["jaw_pose"]], dim=2)
        gt['expression']=smplx_params['expression']
    else:
        gt['rotvec']=torch.concat([
            smplx_params["global_orient"],
            smplx_params["body_pose"]], dim=2)
    gt['rotmat'] = roma.rotvec_to_rotmat(gt['rotvec'])
    gt['v3d'] = humans.vertices.view(bs, n_humans, -1, 3)
    gt['j3d'] = humans.joints.view(bs, n_humans, -1, 3)
    gt['v2d'] = perspective_projection(
            humans.vertices, K.unsqueeze(1).repeat(1, n_humans, 1, 1).view(-1, 3, 3)
        ).reshape(bs, n_humans, -1, 2)
    gt['j2d'] = perspective_projection(
            humans.joints, K.unsqueeze(1).repeat(1, n_humans, 1, 1).view(-1, 3, 3)
        ).reshape(bs, n_humans, -1, 2)
    gt['transl'] = gt['j3d'][:, :, JOINT_INDEX_HEAD, :]
    gt['transl_pelvis'] = gt['j3d'][:, :, JOINT_INDEX_PELVIS, :]
    
    # 2. detection gt
    gt_det, gt_masks = make_det_gt(gt['j2d'], gt_masks, image_size, patch_size, device)
    gt.update(gt_det)
    
    # 3. validation mask
    gt["mask"] = gt_masks
    
    # 4. distance gt
    gt['dist'] = gt['j3d'][:, :, 0, -1] # [bnhv]
    # We may predict dist in log space, or normalized values.
    non_euclidean_dist = log_depth(gt['dist'])
    # Normalise by focal
    focal = K[:, 0, 0][:, None].repeat(1, gt['dist'].shape[1])  # only focal of x
    non_euclidean_dist = focal_length_normalization(non_euclidean_dist, focal, fovn=60, img_size=image_size)
    gt['dist_postprocessed'] = non_euclidean_dist
    
    if apply_gt_masks:
        for k in [
                'shape', 'rotvec', 'rotmat', 'v3d', 'j3d', 'v2d', 'j2d', 
                'transl', 'transl_pelvis', 'mask', 'dist', 'dist_postprocessed']:
            gt[k] = gt[k].reshape((-1,)+gt[k].shape[2:])[gt['idx_vis']]
    
    return gt

def postprocess_pred(pred, idx, bs):
    sid, counts = torch.unique(idx[0], sorted=True, return_counts=True)
    n_max_human_det = counts.max()
    
    idx_human = torch.concat([torch.arange(i) for i in counts])
    idx_flatten = torch.arange(0, pred['v3d'].shape[0], device=idx[0].device, dtype=torch.long)
        
    pred_out = dict()
    for k in ['v3d', 'j3d', 'v2d', 'transl_pelvis']:
        pred_out[k] = torch.zeros([bs, n_max_human_det]+list(pred[k].shape)[1:], device=idx[0].device)
        pred_out[k][idx[0], idx_human] = pred[k][idx_flatten].float()
    
    pred_mask = torch.zeros(bs, n_max_human_det, device=idx[0].device)
    pred_mask[idx[0], idx_human] = 1
    pred_out['mask'] = pred_mask
    
    return pred_out

def simple_matching_pred_label(pred_v3d, gt_v3d, pred_mask, gt_mask):
    # eval only
    # pred_v3d: (bs, n_det_humans, n_verts, 3)
    # gt_v3d: (bs, n_max_humans, n_verts, 3)
    # pred_mask: (bs, n_det_humans)
    # gt_mask: (bs, n_max_humans)
    
    with torch.no_grad():
        # (bs, max_humans, 15, 2)
        pred_v3d_expanded = pred_v3d.unsqueeze(2)  # (bs, n_det_humans, 1, n_verts, 3)
        gt_v3d_expanded = gt_v3d.unsqueeze(1)  # (bs, 1, n_max_humans, n_verts, 3)
        mask_expanded = gt_mask.unsqueeze(1) * pred_mask.unsqueeze(2) # (bs, n_det_humans, n_max_humans)
        mask_expanded = mask_expanded.bool()
        
        # batch pve
        pve = (torch.sqrt(((pred_v3d_expanded - gt_v3d_expanded) ** 2).sum(-1))).mean(-1)
        # (bs, n_det_humans, n_max_humans)
        
        # apply gt mask
        pve_masked = pve.masked_fill(~mask_expanded, float('inf')) 
        
        # match a gt for each pred
        min_distances, min_indices = torch.min(pve_masked, dim=-1)
    return min_indices  # (bs, n_det_humans)

def gather_gt(gt, matched_index):
    # eval only
    new_gt = dict()
    dims0 = list(matched_index.shape)
    for k, v in gt.items():
        if k in ['v3d', 'j3d', 'transl_pelvis', 'mask']:
            new_gt[k] = torch.gather(
                v, 1, 
                matched_index.view(
                    dims0+([1]*(len(v.shape)-2) if len(v.shape) > 2 else [])).expand(dims0+list(v.shape)[2:]))
            new_gt[k] = new_gt[k].view([-1]+list(new_gt[k].shape)[2:])
    return new_gt

def batch_metrics(pred, label, smplx2smpl_regressor, h36m_regressor, smplx_type="smplx"):    
    if len(pred['v3d'].shape) == 3:
        n, m, _ = pred['v3d'].shape
    else:
        bs, nh, m, _ = pred['v3d'].shape
        n = bs*nh
        pred['v3d'] = pred['v3d'].view(n, -1, 3)
        pred['j3d'] = pred['j3d'].view(n, -1, 3)
        if "mask" in pred.keys():
            pred['mask'] = pred['mask'].view(-1)
        
    # mask
    mask = label['mask'].bool()
    if "mask" in pred.keys():
        mask = mask & pred['mask'].bool()
    
    # gt/pred transl
    transl = label['j3d'][:, 0, :][mask]
    transl_hat = pred['j3d'][:, 0, :][mask]
    
    # gt/pred vertices centerex around pelvis
    v3d_ctx = (label['v3d'] - label['j3d'][:, 0:1, :]).view(n, m, 3)[mask]
    v3d_hat_ctx = (pred['v3d'] - pred['j3d'][:, 0:1, :]).view(n, m, 3)[mask]
    
    # gt/pred joints centerex around pelvis (from joints regressor)
    if smplx_type == "smplx":
        j3d_ctx = (h36m_regressor @ (smplx2smpl_regressor @ v3d_ctx))[:, H36M_TO_J14, :]
        j3d_hat_ctx = (h36m_regressor @ (smplx2smpl_regressor @ v3d_hat_ctx))[:, H36M_TO_J14, :]
    else:
        j3d_ctx = (h36m_regressor @ v3d_ctx)[:, H36M_TO_J14, :]
        j3d_hat_ctx = (h36m_regressor @ v3d_hat_ctx)[:, H36M_TO_J14, :]
    
    pve = ((torch.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean()
    mpjpe = ((torch.sqrt(((j3d_ctx - j3d_hat_ctx) ** 2).sum(-1))) * 1000).mean()
    transl = ((torch.sqrt(((transl - transl_hat) ** 2).sum(-1))) * 1000).mean()
        
    return pve, mpjpe, transl

def train_one_epoch(
        exp_name, 
        epoch, 
        model, 
        smplx_model, 
        objective_func, 
        optimizer,
        train_loader, 
        test_loader, 
        writer, 
        smplx2smpl_regressor, 
        h36m_regrerssor,
        image_size, 
        epochs=20, 
        sample_interval=1, 
        device=torch.device("cuda:0"),
        smplx_type="smplx"):
    
    meters = dict(
        pve=AverageMeter("pve"), pve_test=AverageMeter("pve-test"), 
        mpjpe=AverageMeter("mpjpe"), mpjpe_test=AverageMeter("mpjpe-test"), 
        transl_pelvis=AverageMeter("transl-pelvis"), 
        transl_pelvis_test=AverageMeter("transl-pelvis_test"), )
    
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs} [train]', unit='batch') as pbar:
        for i, sample in enumerate(train_loader):
            imgs, K, R, t, smplx_params, gt_masks = sample
            
            # input
            imgs = imgs.to(device)
            K, R, t = [v.to(device).float() for v in [K, R, t]]
            
            # pre-process gt
            gt_masks = gt_masks.to(device)
            smplx_params = {k:v.to(device).float() for k, v in smplx_params.items()}
            gt = preprocess_gt(
                smplx_params, 
                gt_masks, K, R, t, 
                smplx_model, 
                image_size, 
                model.patch_size, 
                device,
                smplx_type=smplx_type)

            # forward
            with torch.amp.autocast('cuda', enabled=True):
                pred, idx = model(imgs, idx=gt['idx'], K=K, is_training=True)
                
                # loss func
                loss, loss_dict = objective_func(pred, gt, epoch, img_size=image_size)
        
            # backward update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            pve, mpjpe, transl_err = batch_metrics(
                pred, 
                gt, 
                smplx2smpl_regressor, 
                h36m_regrerssor,
                smplx_type)
            
            meters['pve'].update(pve.item())
            meters['mpjpe'].update(mpjpe.item())
            meters['transl_pelvis'].update(transl_err.item())
            pbar.set_postfix({
                'loss': loss.item(),
                'pve': meters['pve'].avg,
                'mpjpe': meters['mpjpe'].avg,
                'transl-pelvis': meters['transl_pelvis'].avg})
            pbar.update(1)
            
            for k_loss, v_loss in loss_dict.items():
                writer.add_scalar(f"train_{k_loss}_loss", v_loss.item(), epoch*len(train_loader) + i)
            
            writer.add_scalar("train_pve", pve.item(), epoch*len(train_loader) + i)
            writer.add_scalar("train_mpjpe", mpjpe.item(), epoch*len(train_loader) + i)
            writer.add_scalar("train_transl_pelvis", transl_err.item(), epoch*len(train_loader) + i)
            
    model.eval()
    with tqdm(total=len(test_loader), desc=f'Epoch {epoch+1}/{epochs} [test]', unit='batch') as pbar:
        with torch.no_grad():
            for i, sample in enumerate(test_loader):
                imgs, K, R, t, smplx_params, gt_masks = sample
                
                # input
                imgs = imgs.to(device)
                K, R, t = [v.to(device).float() for v in [K, R, t]]
                
                # pre-process gt
                gt_masks = gt_masks.to(device)
                smplx_params = {k:v.to(device).float() for k, v in smplx_params.items()}
                gt = preprocess_gt(
                    smplx_params, 
                    gt_masks, 
                    K, R, t, 
                    smplx_model, 
                    image_size, 
                    model.patch_size, 
                    device,
                    apply_gt_masks=False,
                    smplx_type=smplx_type)
                
                # forward
                with torch.amp.autocast('cuda', enabled=True):
                    pred, idx = model(imgs, K=K, is_training=False, return_list=False)
                        
                # post-process pred
                pred = postprocess_pred(pred, idx, imgs.shape[0])
                
                # metrics
                match_index = simple_matching_pred_label(pred['v3d'], gt['v3d'], pred['mask'], gt['mask'])
                gt_matched = gather_gt(gt, match_index)
                
                pve, mpjpe, transl_err = batch_metrics(
                    pred, 
                    gt_matched, 
                    smplx2smpl_regressor, 
                    h36m_regrerssor,
                    smplx_type)
                
                if (i+1) % sample_interval == 0:
                    img = denormalize_rgb(imgs[0].cpu().numpy())
                    verts = pred['v2d'][0].cpu().numpy()
                    
                    fig = plt.figure()
                    plt.imshow(img)
                    for hi in range(verts.shape[0]):
                        plt.scatter(verts[hi, :, 0], verts[hi, :, 1], s=0.1)
                    plt.savefig(f"./logs_vis/{exp_name}/vis_{i}.jpg")
            
                meters['pve_test'].update(pve.item())
                meters['mpjpe_test'].update(mpjpe.item())
                meters['transl_pelvis_test'].update(transl_err.item())
                pbar.set_postfix({
                    'pve': meters['pve_test'].avg,
                    'mpjpe': meters['mpjpe_test'].avg,
                    'transl-pelvis': meters['transl_pelvis_test'].avg})
                pbar.update(1)
                
        writer.add_scalar("test_pve", meters['pve_test'].avg, epoch+1)
        writer.add_scalar("test_mpjpe", meters['mpjpe_test'].avg, epoch+1)
        writer.add_scalar("test_transl_pelvis", meters['transl_pelvis_test'].avg, epoch+1)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='model_small_672_chi3d_sv')
    
    # Path/Directory args
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
    
    # Training args
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batchsize', type=int, default=8)
    parser.add_argument('--eval_batchsize', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sample_interval', type=int, default=25)
    parser.add_argument('--device', type=str, default='0')
    
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(f"cuda:{args.device}")
    exp_name = args.exp_name
    dataset_class = CHI3D_SV if args.dataset == 'chi3d' else Hi4D_SV
    
    with open(args.smplx2smpl_path, 'rb') as f:
        smplx2smpl_regressor = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(device)
    j_regressor_h36m = torch.Tensor(np.load(args.j_regressor_h36m_path)).to(device)
    
    if not os.path.exists(os.path.join("./logs", exp_name)):
        os.makedirs(os.path.join("./logs", exp_name))
    if not os.path.exists(os.path.join("./logs_vis", exp_name)):
        os.makedirs(os.path.join("./logs_vis", exp_name))
    
    smpl_model = smplx.create(
        args.smplx_dir, 
        args.smplx_type, 
        use_pca=False, 
        flat_hand_mean=True, 
        gender='neutral', num_betas=10).to(device)
    
    train_dataset = dataset_class(
        root_path=args.dataset_dir,
        cache_path=f"./data/{args.dataset}",
        img_size=args.image_size,
        smpl_model=smpl_model,
        split='training'
    )
    train_loader = DataLoader(
        train_dataset, 
        args.train_batchsize, 
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers)
    
    test_dataset = dataset_class(
        root_path=args.dataset_dir,
        cache_path=f"./data/{args.dataset}",
        img_size=args.image_size,
        smpl_model=smpl_model,
        split='test'
    )
    test_loader = DataLoader(
        test_dataset, 
        args.eval_batchsize, 
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers)
    
    model = Model(
                backbone=args.backbone, 
                img_size=args.image_size,
                smplx_type=args.smplx_type).to(device)
    ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=False)['model_state_dict']
    model.load_state_dict(ckpt, strict=False)
    
    objective_func = Loss().to(device)
    
    optimizer = Adam(model.parameters(), lr=5e-6)
    
    writer = SummaryWriter(f"logs/{exp_name}")
    
    for epoch in range(args.epochs):
        train_one_epoch(
            exp_name, 
            epoch, 
            model, 
            smpl_model, 
            objective_func, 
            optimizer, 
            train_loader, 
            test_loader, 
            writer, 
            smplx2smpl_regressor, 
            j_regressor_h36m,
            args.image_size, 
            args.epochs, 
            sample_interval=args.sample_interval, 
            device=device,
            smplx_type=args.smplx_type)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{exp_name}.pt"))
    
    