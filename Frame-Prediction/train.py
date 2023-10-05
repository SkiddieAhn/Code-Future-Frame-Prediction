from torch.utils.data import DataLoader
import argparse
from utils import *
import dataset
from config import update_config
from training.train_pre_func import * 
from training.train_func import training

def main():
    parser = argparse.ArgumentParser(description='Future_Frame_Prediction')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dataset', default='ped2', type=str, help='The name of the dataset to train.')
    parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
    parser.add_argument('--resume', default=None, type=str,
                        help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
    parser.add_argument('--save_interval', default=10000, type=int, help='Save the model every [save_interval] iterations.')
    parser.add_argument('--val_interval', default=1000, type=int,
                        help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
    parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Pre-work for Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    
    # setup seed (for deterministic behavior)
    seed(seed_value=train_cfg.manualseed)

    # get dataset and loader
    train_dataset = dataset.train_dataset(train_cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    print_infor(cfg=train_cfg, dataloader=train_dataloader)

    # define models
    generator, discriminator, flownet = def_models()

    # define losses
    adversarial_loss, discriminate_loss, gradient_loss, intensity_loss, flow_loss = def_losses()

    # define optimizer 
    optimizer_G, optimizer_D = def_optim(train_cfg, generator, discriminator)

    # load models
    load_models(train_cfg, generator, discriminator, flownet, optimizer_G, optimizer_D)

    # load scores
    scores = load_scores(train_cfg)

    # make dict
    models = make_model_dict(generator, discriminator, flownet)
    losses = make_loss_dict(discriminate_loss, intensity_loss, gradient_loss, adversarial_loss, flow_loss)
    opts = make_opt_dict(optimizer_G, optimizer_D)


    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # train
    training(train_cfg, train_dataset, train_dataloader, models, losses, opts, scores)


if __name__=="__main__":
    main()