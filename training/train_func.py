import torch 
import random
from utils import psnr_error
from eval import val
from training.train_ing_func import *
import time
import datetime


def training(cfg, dataset, dataloader, models, losses, opts, scores):
    # define start_iter
    start_iter = scores['step'] if scores['step'] > 0 else 0

    # find epoch 
    epoch = int(scores['step']/len(dataloader)) # [epoch: current step / (data size / batch size)] ex) 8/(16/4)

    # training start!
    training = True  
    torch.autograd.set_detect_anomaly(True)

    print('\n===========================================================')
    print('Training Start!')
    print('===========================================================')

    while training:
        '''
        ------------------------------------
        Training (1 epoch)
        ------------------------------------
        '''
        for indice, clips in dataloader:
            # define frame 1 to 4 
            frame_1 = clips[:, 0:3, :, :].cuda()  # (n, 3, 256, 256) 
            frame_2 = clips[:, 3:6, :, :].cuda()  # (n, 3, 256, 256) 
            frame_3 = clips[:, 6:9, :, :].cuda()  # (n, 3, 256, 256) 
            frame_4 = clips[:, 9:12, :, :].cuda()  # (n, 3, 256, 256) 

            # pop() the used video index
            for index in indice:
                dataset.all_seqs[index].pop()
                if len(dataset.all_seqs[index]) == 0:
                    dataset.all_seqs[index] = list(range(len(dataset.videos[index]) - 4))
                    random.shuffle(dataset.all_seqs[index])

            # generator input, target
            input = torch.cat([frame_1,frame_2, frame_3, frame_4], 1).cuda() # (n, 12, 256, 256) 
            target = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256) 

            # forward
            G_l, D_l, F_frame = forward(input=input.cuda(), target=target, input_last=frame_4, models=models, losses=losses) # (n, 3, 256, 256) 
            scores['g_loss_list'].append(G_l.item())
            scores['d_loss_list'].append(D_l.item())

            # backward
            opts['optimizer_G'].zero_grad()
            G_l.backward()
            opts['optimizer_G'].step()

            opts['optimizer_D'].zero_grad()
            D_l.backward()
            opts['optimizer_D'].step()

            # calculate time
            torch.cuda.synchronize()
            time_end = time.time()
            if scores['step'] > start_iter:  
                iter_t = time_end - temp
            temp = time_end


            if scores['step'] != start_iter:
                '''
                -----------------------------------
                check train status per 20 iteration
                -----------------------------------
                '''
                if scores['step'] % 20 == 0:
                    print(f"===========epoch:{epoch} (step:{scores['step']})============")

                    # calculate remained time
                    time_remain = (cfg.iters - scores['step']) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]

                    # calculate psnr
                    psnr = psnr_error(F_frame, target)

                    # print loss, psnr, auc
                    print(f"[{scores['step']}] G_l: {G_l:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "\
                    f"best_auc: {scores['best_auc']:.3f} | iter_t: {iter_t:.3f}s | remain_t: {eta}")

                    # view loss by graph
                    view_loss(cfg, scores)

                '''
                --------------------------------
                find Best model per val_interval
                --------------------------------
                '''
                if scores['step'] % cfg.val_interval == 0:
                    auc, scores = val(cfg=cfg, train_scores=scores, models=models, iter=scores['step'])
                    update_best_model(cfg, auc, scores['step'], models, opts, scores)

                '''
                ------------------------------------
                save current model per save_interval 
                ------------------------------------
                '''
                if scores['step'] % cfg.save_interval == 0:
                    model_dict = make_models_dict(models, opts, scores)
                    torch.save(model_dict, f'weights/latest_{cfg.dataset}.pth')
                    print(f"\nAlready saved: \'latest_{cfg.dataset}.pth\'.")

                # training complete!
                if scores['step'] == cfg.iters:
                    training = False

            # one iteration ok!
            scores['step'] += 1
            
        # one epoch ok!
        epoch += 1
        

def forward(input, target, input_last, models, losses):
    '''
    Return generator_loss, discriminator_loss, generated_frame
    '''
    generator = models['generator']
    discriminator = models['discriminator']
    flownet = models['flownet']

    discriminate_loss = losses['discriminate_loss']
    intensity_loss = losses['intensity_loss']
    gradient_loss = losses['gradient_loss']
    adversarial_loss = losses['adversarial_loss']
    flow_loss = losses['flow_loss']

    coefs = [1, 1, 0.05, 2] # inte_l, grad_l, adv_l, flow_l

    # future frame prediction and get loss
    pred  = generator(input)
    inte_l = intensity_loss(pred, target)
    grad_l = gradient_loss(pred, target)
    adv_l = adversarial_loss(discriminator(pred))

    # flowmap prediction and get loss
    gt_flow_input = torch.cat([input_last.unsqueeze(2), target.unsqueeze(2)], 2)
    pred_flow_input = torch.cat([input_last.unsqueeze(2), pred.unsqueeze(2)], 2)

    flow_gt = (flownet(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
    flow_pred = (flownet(pred_flow_input * 255.) / 255.).detach()
    flow_l = flow_loss(flow_pred, flow_gt)

    loss_gen = coefs[0] * inte_l + \
                coefs[1] * grad_l + \
                coefs[2] * adv_l + \
                coefs[3] * flow_l

    # discriminator
    loss_dis = discriminate_loss(discriminator(target),
                                 discriminator(pred.detach()))

    return loss_gen, loss_dis, pred
