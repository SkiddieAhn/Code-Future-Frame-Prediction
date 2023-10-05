import os
import torch 
from sklearn import metrics

from dataset import Label_loader
import dataset
from utils import *
import numpy as np
from evaluation.save_func import * 


def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero

    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def val_train_eval(cfg, train_scores, generator, iter):
    dataset_name = cfg.dataset
    video_folders = os.listdir(cfg.test_data)
    video_folders.sort()
    video_folders = [os.path.join(cfg.test_data, aa) for aa in video_folders]

    # psnr & auc 
    psnr_group = []
    auc = 0

    '''
    ===========================
    1. get PSNR Error 
    ===========================
    '''
    with torch.no_grad():
        for _, folder in enumerate(video_folders):
            one_video = dataset.test_dataset(cfg, folder)
            psnrs = []

            for _, clip in enumerate(one_video):
                frame_1 = clip[0:3, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_2 = clip[3:6, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_3 = clip[6:9, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_4 = clip[9:12, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 

                input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1).cuda() # (1, 12, 256, 256) 
                target_frame = clip[12:15, :, :].unsqueeze(0).cuda() # (1, 3, 256, 256)

                F_frame = generator(input)
                test_psnr = psnr_error(F_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))

            # record psnrs per video 
            psnr_group.append(np.array(psnrs))

    '''
    ===========================
    2. get AUC score
    ===========================
    '''
    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()
    assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)

    for i in range(len(psnr_group)):
        psnrs = psnr_group[i]
        distance = min_max_normalize(psnrs)

        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, gt[i][4:]), axis=0)  # Exclude the first 4 unpredictable frames in gt.

    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)

    '''
    ===========================
    3. record AUC Score 
    ===========================
    '''
    train_scores['iter_list'].append(iter)
    train_scores['auc_list'].append(auc)

    save_text(f"[{dataset_name}][{iter}] AUC: {auc}", f'results/auc_{dataset_name}.txt')
    save_auc_graph_train(iters=train_scores['iter_list'], scores=train_scores['auc_list'], file_path=f'results/auc_{dataset_name}.jpg')

    return auc, train_scores
