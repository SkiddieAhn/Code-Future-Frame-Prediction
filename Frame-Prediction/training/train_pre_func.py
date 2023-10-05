import torch 
from utils import *
import argparse
from training.losses import *
from model.generator import UNet
from model.discriminator import PixelDiscriminator
from flownet.flownet2.models import FlowNet2SD
from utils import weights_init_normal


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def seed(seed_value):
    if seed_value == -1:
        return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def print_infor(cfg, dataloader):
    cfg.epoch_size = cfg.iters // len(dataloader)
    cfg.print_cfg() 

    print('\n===========================================================')
    print('Dataloader Ok!')
    print('-----------------------------------------------------------')
    print('[Data Size]:',len(dataloader.dataset))
    print('[Batch Size]:',cfg.batch_size)
    print('[One epoch]:',len(dataloader),'step   # (Data Size / Batch Size)')
    print('[Epoch & Iteration]:',cfg.epoch_size,'epoch &', cfg.iters,'step')
    print('-----------------------------------------------------------')
    print('===========================================================')


def def_models():
    # generator
    generator = UNet(12,3).cuda()
    # discriminator
    discriminator = PixelDiscriminator(input_nc=3).cuda()
    # flownet
    flownet = FlowNet2SD().cuda()

    return generator, discriminator, flownet


def def_losses():
    adversarial_loss = Adversarial_Loss().cuda()
    discriminate_loss = Discriminate_Loss().cuda()
    gradient_loss = Gradient_Loss(3).cuda()
    intensity_loss = Intensity_Loss().cuda()
    flow_loss = Flow_Loss().cuda()

    return adversarial_loss, discriminate_loss, gradient_loss, intensity_loss, flow_loss


def def_optim(cfg, gen, disc):
    optim_G = torch.optim.Adam(gen.parameters(), lr=cfg.g_lr)
    optim_D = torch.optim.Adam(disc.parameters(), lr=cfg.d_lr)
    return optim_G, optim_D


def load_models(cfg, generator, discriminator, flownet, optimizer_G, optimizer_D):
    if cfg.resume:
        generator.load_state_dict(torch.load(cfg.resume)['net_g'])
        discriminator.load_state_dict(torch.load(cfg.resume)['net_d'])
        optimizer_G.load_state_dict(torch.load(cfg.resume)['optimizer_g'])
        optimizer_D.load_state_dict(torch.load(cfg.resume)['optimizer_d'])
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # flownet
    flownet.load_state_dict(torch.load('flownet/pretrained/FlowNet2-SD.pth')['state_dict'])
    flownet.eval()


def load_scores(cfg):
    if cfg.resume:
        step = torch.load(cfg.resume)['step']
        iter_list = torch.load(cfg.resume)['iter_list']
        best_auc = torch.load(cfg.resume)['best_auc']
        auc_list = torch.load(cfg.resume)['auc_list']
        g_loss_list = torch.load(cfg.resume)['g_loss_list']
        d_loss_list = torch.load(cfg.resume)['d_loss_list']
    else:
        step = 0
        iter_list = []
        best_auc = 0
        auc_list= []
        g_loss_list = []
        d_loss_list = []

    scores = dict()
    scores['step'] = step
    scores['iter_list'] = iter_list
    scores['best_auc'] = best_auc
    scores['auc_list'] = auc_list
    scores['g_loss_list'] = g_loss_list
    scores['d_loss_list'] = d_loss_list

    return scores


def make_model_dict(generator, discriminator, flownet):
    models = dict()
    models['generator'] = generator
    models['discriminator'] = discriminator
    models['flownet'] = flownet
    return models
    

def make_loss_dict(discriminate_loss, intensity_loss, gradient_loss, adversarial_loss, flow_loss):
    losses = dict()
    losses['discriminate_loss'] = discriminate_loss
    losses['intensity_loss'] = intensity_loss
    losses['gradient_loss'] = gradient_loss
    losses['adversarial_loss'] = adversarial_loss
    losses['flow_loss'] = flow_loss
    return losses


def make_opt_dict(optimizer_G, optimizer_D):
    opts = dict()
    opts['optimizer_G'] = optimizer_G
    opts['optimizer_D'] = optimizer_D
    return opts
