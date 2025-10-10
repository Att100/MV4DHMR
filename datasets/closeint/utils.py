import torch
import roma

def get_ratio_padding(size_org, size_out):
    # size_org: (W, H)
    # size_out: int, assume size_out is square
    if size_org[0] > size_org[1]:
        ratio = size_out / size_org[0]
    else:
        ratio = size_out / size_org[1]
    ws, hs = size_org[0] * ratio, size_org[1] * ratio
    px, py = (size_out-ws)/2, (size_out-hs)/2
    return ratio, px, py

def adjust_K_by_ratio_padding(K, ratio, px, py):
    K = K.clone()
    K[:,0,0], K[:,1,1] = K[:,0,0]*ratio, K[:,1,1]*ratio
    K[:,0,-1], K[:,1,-1] = K[:,0,-1]*ratio+px, K[:,1,-1]*ratio+py
    return K

# def smpl_world2cam(smpl_model, smpl_params, K, R, t):
#     # smpl_params: {k:v with shape (bs, n_humans, ...)}
#     # K/R: (bs, 3, 3); t: (bs, 1, 3)
    
#     bs, n_humans = smpl_params['transl'].shape[:2]
    
#     K, R, t = [v.repeat(1, 2, 1, 1).view(bs*n_humans, -1, 3) for v in [K, R, t]]
    
#     output = smpl_model(**{k:v.view((-1,)+v.shape[2:]) for k,v in smpl_params.items() if k != "transl"})
#     joints = output.joints + smpl_params['transl'].view((-1, 1, 3))

#     joints_cam = joints - joints[:, 0:1, :] + torch.matmul(
#         R, joints[:, 0:1, :].transpose(-1, -2)).transpose(-1, -2) + t

#     transl = (joints_cam - output.joints)[:, 0]
#     global_orient = roma.rotmat_to_rotvec(torch.matmul(
#         R, roma.rotvec_to_rotmat(smpl_params['global_orient'].view(-1, 3))))

#     smpl_params_cam = {k:v for k,v in smpl_params.items() if k not in ['transl', 'global_orient']}
#     smpl_params_cam.update(dict(
#         global_orient=global_orient.reshape(smpl_params['global_orient'].shape),
#         transl=transl.reshape(smpl_params['transl'].shape)))
        
#     return smpl_params_cam