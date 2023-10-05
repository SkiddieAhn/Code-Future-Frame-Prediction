import torch 
import argparse
from evaluation.train_eval import val_train_eval
from evaluation.test_eval import val_test_eval

from config import update_config
from model.generator import UNet 

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to evaluate.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', default=True, type=bool, help='show curve')
parser.add_argument('--show_heatmap', default=True, type=bool, help='show heatmap')


def val(cfg, train_scores=None, models=None, iter=None):
    '''
    ========================================
    This is for evaluation during training.    
    ========================================
    '''
    if models:  
        generator = models['generator']
        generator.eval()

        # validation 
        auc, train_scores = val_train_eval(cfg, train_scores, generator, iter)

        generator.train()
        return auc, train_scores

    '''
    ========================================
    This is for evaluation during testing.    
    ========================================
    '''
    generator = UNet(12, 3).cuda().eval()

    if cfg.trained_model:
        generator.load_state_dict(torch.load(f'weights/' + cfg.trained_model + '.pth')['net_g'])
        iter = torch.load(f'weights/' + cfg.trained_model + '.pth')['step']
        val_test_eval(cfg, generator, iter)
    else:
        print('no trained model!')


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)