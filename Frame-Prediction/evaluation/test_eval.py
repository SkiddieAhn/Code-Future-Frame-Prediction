import os
import time
import torch 
from sklearn import metrics

from dataset import Label_loader
import dataset
from utils import *
import numpy as np
from torchvision.utils import save_image
from evaluation.save_func import * 


def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero

    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def val_test_eval(cfg, generator, iter):
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
        for i, folder in enumerate(video_folders):
            # Testing Log
            if not os.path.exists(f"results/{dataset_name}/{iter}/f{i+1}"):
                os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}")
                os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}/real")
                os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}/fake")
                os.makedirs(f"results/{dataset_name}/{iter}/f{i+1}/diff")

            one_video = dataset.test_dataset(cfg, folder)
            psnrs = []

            for j, clip in enumerate(one_video):
                frame_1 = clip[0:3, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_2 = clip[3:6, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_3 = clip[6:9, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 
                frame_4 = clip[9:12, :, :].unsqueeze(0).cuda()  # (1, 3, 256, 256) 

                input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1).cuda() # (1, 12, 256, 256) 
                target_frame = clip[12:15, :, :].unsqueeze(0).cuda() # (1, 3, 256, 256)

                F_frame = generator(input)
                test_psnr = psnr_error(F_frame, target_frame).cpu().detach().numpy()
                psnrs.append(float(test_psnr))

                '''
                =============================
                2. save Image, PSNR, Heatmap
                =============================
                '''
                # save image
                real_image = ((target_frame[0] + 1 ) / 2)[(2,1,0),...]
                pred_image = ((F_frame[0] + 1 ) / 2)[(2,1,0),...]
                save_image(real_image, f'results/{dataset_name}/{iter}/f{i+1}/real/{j}.jpg')
                save_image(pred_image, f'results/{dataset_name}/{iter}/f{i+1}/fake/{j}.jpg')

                # save PSNR
                save_text(f"[{i+1} video, {j} frame]: {test_psnr} psnr", f'results/{dataset_name}/{iter}/f{i+1}/psnrs.txt')

                # save Heatmap
                if cfg.show_heatmap:
                    save_heatmap(F_frame, target_frame, f'results/{dataset_name}/{iter}/f{i+1}/diff/{j}.jpg')

                torch.cuda.synchronize()
                
                # print FPS
                end = time.time()
                if j > 1:  # Compute fps by calculating the time used in one completed iteration, this is more accurate.
                    fps = 1 / (end - temp)
                    print(f'\rDetecting: [{i + 1:02d}] {j + 1}/{len(one_video)}, {fps:.2f} fps.', end='')
                temp = end

            # record psnrs per video 
            psnr_group.append(np.array(psnrs))

    '''
    ============================
    3. calc Anomaly & AUC Score
    ============================
    '''
    gt_loader = Label_loader(cfg, video_folders)  # Get gt labels.
    gt = gt_loader()
    assert len(psnr_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(psnr_group)} detected videos.'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)

    for i in range(len(psnr_group)):
        psnrs = psnr_group[i]
        distance = min_max_normalize(psnrs)

        label = gt[i][4:]
        scores = np.concatenate((scores, distance), axis=0)
        labels = np.concatenate((labels, label), axis=0) 

        '''
        ===========================
        4. Analyze per Video 
        ===========================
        '''
        # save anomaly_score
        for k in range(len(distance)):
            save_text(f"[{k+1}] Anomaly score: {distance[k]} | label: {label[k]}", f'results/{dataset_name}/{iter}/f{i+1}/score_label.txt')

        # save anomaly_score by graph 
        anomalies_idx = [i for i,l in enumerate(label) if l==1] 
        save_score_graph(y='Anomaly Score', answers_idx=anomalies_idx, scores=distance, file_path=f'results/{dataset_name}/{iter}/f{i+1}/anomaly_score.jpg')

        # save auc score 
        fpr, tpr, _ = metrics.roc_curve(label, distance, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        save_text(f"[{i+1} video] auc: {auc} auc\n", f'results/{dataset_name}/{iter}/auc.txt')

    # calc total auc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    print('auc:', auc)

    # save roc_curve with auc 
    if cfg.show_curve:
        save_roc_curve(fpr, tpr, auc, file_path=f'results/{dataset_name}/{iter}/total_auc_curve.jpg')
