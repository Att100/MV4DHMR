import torch
import smplx
import os
import roma
from tqdm import tqdm
import numpy as np
import pickle
from PIL import Image
import argparse
from tqdm import tqdm

from multiview.refine import world2cam
from pipeline import Pipeline
from datasets.closeint.chi3d import CHI3DSequenceMetaLoader
from utils.camera import perspective_projection
from datasets import sequence_loaders


def main(args):
    device = torch.device(f'cuda:{args.device}')
    
    pipeline = Pipeline(args, device)
    loader = sequence_loaders[args.dataset](args.dataset_dir)
    
    if args.dataset == 'chi3d':
        images, K, R, t = loader.load(args.subject, args.sequence)
    elif args.dataset == 'panoptic':
        images, K, R, t = loader.load(args.sequence)
    elif args.dataset == 'shelf':
        images, K, R, t = loader.load()
    else:
        raise NotImplementedError()
    
    start_frame = args.start_frame
    end_frame = len(images) if args.end_frame == -1 else args.end_frame
    for frame_id in tqdm(range(start_frame, end_frame, args.sample_interval)):
        tracks = pipeline.step(images[frame_id], K, R, t)
            
    tracking = pipeline.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Path / Directory args
    parser.add_argument('--pretrained_path', type=str, default='./checkpoints/pretrained/multiHMR_672_S.pt')
    parser.add_argument('--smplx_dir', type=str, default='./checkpoints')
    parser.add_argument('--smplx2smpl_path', type=str, default='./checkpoints/smplx/smplx2smpl.pkl')
    parser.add_argument('--j_regressor_h36m_path', type=str, default='./checkpoints/J_regressor_h36m.npy')
    parser.add_argument('--dataset_dir', type=str, default='/ocean/projects/cis240055p/czheng3/chi3d')
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/saved/")
    
    # Model args
    parser.add_argument('--backbone', type=str, default='dinov2_vits14')
    parser.add_argument("--smplx_type", type=str, default='smplx', choices=['smpl', 'smplx'])
    parser.add_argument('--image_size', type=int, default=672, choices=[672, 896])
    parser.add_argument('--image_size_org', type=str, default='900,900')
    
    # Dataset args
    parser.add_argument('--dataset', type=str, default='chi3d', choices=['chi3d', 'hi4d', 'panoptic', 'shelf'])
    
    # Demo args
    parser.add_argument('--eval_mode', type=str, default='zeroshot', choices=['zeroshot', 'finetune'])
    parser.add_argument('--recenter', action='store_true', help='recenter the pelvis of each human to origin')
    parser.add_argument('--n_humans', type=int, default=2)
    parser.add_argument('--n_views', type=int, default=-1)  # -1 for all views
    parser.add_argument('--data_smplx_type', type=str, default='smplx')
    parser.add_argument('--device', type=str, default='0')
    
    # Pipeline args
    parser.add_argument('--smooth', action='store_true', help='apply 1euro filter to smooth the predictions')
    parser.add_argument('--auto_interpolate', action='store_true', help='automatically interpolate missing frames during tracking')
    parser.add_argument('--vis', action='store_true', help="save visualization")
    parser.add_argument('--vis_mode', type=str, default='scatter', choices=['scatter', 'render'])
    parser.add_argument('--vid', action='store_true', help='generate video by cat all visualization images')
    parser.add_argument('--save', action='store_true', help='save tracked SMPL/SMPL-X parameters')
    
    # Sequence args
    parser.add_argument('--subject', type=str, default="s04", choices=['s02', 's03', 's04'])  # for chi3d only
    parser.add_argument('--sequence', type=str, default="Grab 7")
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=-1)  # -1 for the last frame
    parser.add_argument('--sample_interval', type=int, default=1)
    
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.save_dir, "visualization")):
        os.makedirs(os.path.join(args.save_dir, "visualization"))
        
    if not os.path.exists(os.path.join(args.save_dir, "smplx")):
        os.makedirs(os.path.join(args.save_dir, "smplx"))
        
    main(args)
    