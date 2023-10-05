import torch 
import matplotlib.pyplot as plt

def view_loss(cfg, scores):
    g_loss_list = scores['g_loss_list']
    d_loss_list = scores['d_loss_list']

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss_list,label="G")
    plt.plot(d_loss_list,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    # save
    file_path = f'results/loss_graph_{cfg.dataset}.png'
    plt.savefig(file_path)


def make_models_dict(models, opts, scores):
    model_dict = {'net_g': models['generator'].state_dict(), 'optimizer_g': opts['optimizer_G'].state_dict(),
                'net_d': models['discriminator'].state_dict(), 'optimizer_d': opts['optimizer_D'].state_dict(),
                'step':int(scores['step']), 'iter_list':scores['iter_list'], 
                'best_auc':float(scores['best_auc']), 'auc_list':scores['auc_list'],
                'g_loss_list':scores['g_loss_list'], 'd_loss_list':scores['d_loss_list']}
    return model_dict


def update_best_model(cfg, auc, iteration, models, opts, scores):
    if auc > scores['best_auc']:
        scores['best_auc'] = auc
        model_dict = make_models_dict(models, opts, scores)
        save_path = f"weights/best_model_{cfg.dataset}.pth"
        torch.save(model_dict, save_path)

        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f"[best model] update! at {iteration} iteration!! [auc: {scores['best_auc']:.3f}]")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    return scores