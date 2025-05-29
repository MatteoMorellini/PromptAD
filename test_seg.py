import warnings
#warnings.simplefilter("ignore", category=FutureWarning)
import argparse
import torch.optim.lr_scheduler
from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'SEG'

def test(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        ):

    # change the model into eval mode
    model.eval_mode()

    if args.checkpoint:
        previous_checkpoint = './result/brainmri/k_-1/checkpoint/SEG-Seed_111-normal_brain-check_point.pt'
        torch.load(previous_checkpoint)
        model.load_state_dict(torch.load(previous_checkpoint), strict=False)
    else:
        torch.load(check_path)
        model.load_state_dict(torch.load(check_path), strict=False)

    score_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []
    image_scores = []
    with torch.no_grad():
        for (data, mask, label, name, img_type) in dataloader:
            data = [model.transform(Image.fromarray(f.numpy())) for f in data]
            data = torch.stack(data, dim=0)

            for d, n, l, m in zip(data, name, label, mask):
                test_imgs += [denormalization(d.cpu().numpy())]
                m = m.numpy()
                m[m > 0] = 1

                names += [n]
                gt_mask_list += [m] # mask tensor 
                
                gt_list += [l] # label

            data = data.to(device)
            if args.dataset == 'brainmri':
                image_score, score_map = model(data, 'cls')
                image_scores += image_score
            else:
                score_map = model(data, 'seg')
            score_maps += score_map
    test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
    print('qui ci sono')
    if args.vis:
        plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir, inference = True)



def main(args):
    kwargs = vars(args)
    brainmri_from_scratch = False

    if kwargs['seed'] is None:
        kwargs['seed'] = 222

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    if kwargs['dataset'] == 'brats':
        kwargs['distance_per_slice'] = 5
    else:
        kwargs['distance_per_slice'] = 0
    
    kwargs['inference'] = True

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the model
    if kwargs['dataset'] == 'brainmri' and kwargs['class_name'] == 't2w':
        kwargs['class_name'] = 'normal_brain'
        brainmri_from_scratch = True
    
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    model = PromptAD(**kwargs)
    model = model.to(device)

    # get the test dataloader
    if brainmri_from_scratch:
        kwargs['dataset'] = 'brats'
        kwargs['class_name'] = 't2w'
        kwargs['distance_per_slice'] = 1
        
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)
    print(len(test_dataloader))
    test(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path)

    #p_roc = round(metrics['p_roc'], 2)
    #object = kwargs['class_name']
    #print(f'Object:{object} =========================== Pixel-AUROC:{p_roc}\n')

    #save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
    #            kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='brats', choices=['mvtec', 'visa', 'brats', 'brainmri'])
    parser.add_argument('--class_name', type=str, default='t2w')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=240) #before was 400

    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=10)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=bool, default=False)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--left_slice", type=int, default=0)
    parser.add_argument("--right_slice",  type=int, default=20)

    parser.add_argument("--checkpoint", type=bool, default = False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
