# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np
from itertools import product

def compute_prf1(count, miss, fp):
    """
    Code modified from https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/simple_romp/evaluation/RH_evaluation/evaluation.py#L90
    """
    if count == 0:
        return 0, 0, 0
    all_tp = count - miss
    all_fp = fp
    all_fn = miss
    if all_tp == 0:
        return 0., 0., 0.
    all_f1_score = all_tp / (all_tp + 0.5 * (all_fp + all_fn))
    all_recall = all_tp / (all_tp + all_fn)
    all_precision = all_tp / (all_tp + all_fp)
    return 100. * all_precision, 100.* all_recall, 100. * all_f1_score

def match_greedy(
        pred_kps,
        gtkp,
        valid_mask,
        thresh=0.05,
        valid=None,
        metric='iou2d'):
    '''
    Code modified from: https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/simple_romp/trace2/evaluation/eval_3DPW.py#L232
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
            
    metric -> ['iou2d', 'iou3d', 'pve']
    '''
    
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_mask[comb[1]]
        assert vmask.sum()>0, print('no valid points')
        errors_per_pair[str(comb)] = np.linalg.norm(pred_kps[comb[0]][vmask] - gtkp[comb[1]][vmask], 2)
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            if sum(np.inf == errors_per_pair_list) == len(
                    errors_per_pair_list):
                print('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            
            # compute IOU/PVE
            if metric == 'iou2d':
                distance = get_bbx_overlap(
                    pred_kps[minComb[0]], gtkp[minComb[1]])
            elif metric == 'iou3d':
                distance = get_bbx_overlap_3d(
                    pred_kps[minComb[0]], gtkp[minComb[1]])
            elif metric == 'pve':  # mm
                distance = np.mean((np.sqrt(np.sum((gtkp[minComb[1]] - pred_kps[minComb[0]]) ** 2, axis=-1))) * 1000)
            else:
                raise Exception(f"Metric '{metric}' Not implemented, use ['iou2d', 'iou3d', 'pve']")
            
            is_iou = metric in ['iou2d', 'iou3d']
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and distance >= thresh if is_iou else distance <= thresh:
                #print(imgPath + ': found matching')
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if distance < thresh if is_iou else distance > thresh:
                    #print(
                    #   imgPath + ': false positive detected using threshold')
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    bestMatch = np.array(bestMatch)
    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    falsePositives = []
    misses = []

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            #print(imgPath + ': miss')
            misses.append(notAssignedIdGt)

    return bestMatch, falsePositives, misses  # tuples are (idx_pred_kps, idx_gt_kps)

def get_bbx_overlap(p1, p2):
    """
    Code modifed from https://github.com/Arthur151/ROMP/blob/4eebd3647f57d291d26423e51f0d514ff7197cb3/simple_romp/trace2/evaluation/eval_3DPW.py#L185
    """
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def get_bbx_overlap_3d(p1, p2):
    """
    Computes the 3D Intersection over Union (IoU) between two sets of 3D points.

    Parameters:
    p1, p2: np.ndarray of shape (N, 3) and (M, 3)
        - N and M are the number of points in each set
        - Each row represents a (x, y, z) coordinate

    Returns:
    iou: float
        - The 3D IoU score between the two bounding boxes
    """
    # Compute the min/max along each axis to get the bounding boxes
    min_p1, max_p1 = np.min(p1, axis=0), np.max(p1, axis=0)
    min_p2, max_p2 = np.min(p2, axis=0), np.max(p2, axis=0)

    # Define bounding boxes
    bb1 = {'x1': min_p1[0], 'x2': max_p1[0],
           'y1': min_p1[1], 'y2': max_p1[1],
           'z1': min_p1[2], 'z2': max_p1[2]}
    bb2 = {'x1': min_p2[0], 'x2': max_p2[0],
           'y1': min_p2[1], 'y2': max_p2[1],
           'z1': min_p2[2], 'z2': max_p2[2]}

    # Ensure valid bounding boxes
    assert bb1['x1'] < bb1['x2'] and bb1['y1'] < bb1['y2'] and bb1['z1'] < bb1['z2']
    assert bb2['x1'] < bb2['x2'] and bb2['y1'] < bb2['y2'] and bb2['z1'] < bb2['z2']

    # Compute the intersection box
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    z_front = max(bb1['z1'], bb2['z1'])

    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    z_back = min(bb1['z2'], bb2['z2'])

    # Compute intersection volume (only if there is overlap)
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_depth = max(0, z_back - z_front)

    intersection_volume = inter_width * inter_height * inter_depth

    # Compute volumes of each bounding box
    volume_bb1 = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']) * (bb1['z2'] - bb1['z1'])
    volume_bb2 = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']) * (bb2['z2'] - bb2['z1'])

    # Compute IoU
    iou = intersection_volume / (volume_bb1 + volume_bb2 - intersection_volume) if intersection_volume > 0 else 0.0

    return iou


class AverageMeter(object):
    """
    Code mofied from https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    Computes and stores the average and current value
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.detach()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
