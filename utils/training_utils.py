import random
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(TASK, root_dir, **kwargs):

    k_shot = kwargs['k_shot']
    dataset = kwargs['dataset']
    checkpoint = '_finetuned' if kwargs['checkpoint'] else ''

    csv_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}{checkpoint}', 'csv')
    check_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}{checkpoint}', 'checkpoint')
    csv_path = os.path.join(csv_dir, f"Seed_{kwargs['seed']}-results.csv")

    if dataset == 'brainmri' and kwargs['class_name'] == 't2w':
        check_path = os.path.join(check_dir, f"{TASK}-Seed_{kwargs['seed']}-normal_brain-check_point.pt")
        from_brainmri = 'brats_from_brainmri-'
    else:
        check_path = os.path.join(check_dir, f"{TASK}-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt")
        from_brainmri = ''

    folder = f"{from_brainmri}{k_shot}_shot-100_epochs{checkpoint}" if kwargs['inference'] else 'imgs'
    img_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}{checkpoint}', folder)

    

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return img_dir, csv_path, check_path
