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

import copy

TASK = 'SEG'

import torch.nn.functional as F

def compute_patchwise_cross_entropy(logits, target_v2t):
    """
    Calcola la cross-entropy per ciascuna patch.

    Args:
        logits (Tensor): forma (batch_size, num_classes, num_patches)
        target_v2t (Tensor): forma (batch_size, num_patches), con valori interi tra 0 e num_classes-1

    Returns:
        Tensor: cross-entropy per ciascuna patch, forma (batch_size, num_patches)
    """
    # Verifica forme
    assert logits.ndim == 3, f"logits deve avere 3 dimensioni, ricevuto {logits.shape}"
    assert target_v2t.ndim == 2, f"target_v2t deve avere 2 dimensioni, ricevuto {target_v2t.shape}"
    assert logits.shape[0] == target_v2t.shape[0], "Dimensione batch non corrisponde"
    assert logits.shape[2] == target_v2t.shape[1], "Numero di patch non corrisponde"

    # Trasformiamo logits per adattarsi a cross_entropy: (batch_size * num_patches, num_classes)
    batch_size, num_classes, num_patches = logits.shape
    logits = logits.permute(0, 2, 1).reshape(-1, num_classes)
    targets = target_v2t.reshape(-1)

    # Calcolo della cross-entropy
    # loss è una lista di lunghezza batch_size x #patch
    loss = F.cross_entropy(logits, targets, reduction='none')
    print(f"shape before unrolling is {loss.shape}")
    indice_massima = list(loss).index(max(list(loss)))
    print(indice_massima)
    print(f"foto nella batch nr {indice_massima // 225}")

    #for i, row in enumerate(loss): 
    #    print(i)

    print(f"the maximum value in the CE is {loss.max()}")
    print(f"the average of the CE is {loss.mean()}")
    return loss

def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)


def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        train_data: DataLoader
        ):
    # change the model into eval mode
    model.eval_mode()

    features1 = []
    features2 = []
    for (data, mask, label, name, img_type) in train_data:
        data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
        data = torch.stack(data, dim=0).to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)


    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)

    model.build_image_feature_gallery(features1, features2)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    first_image = None
    for epoch in range(args.Epoch):
        for (data, mask, label, name, img_type) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0).to(device)

            data = data.to(device)

            # normal, MAP, LAP
            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            # align LAP and MAP as squared l2 norm
            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0
            # extract 3rd and 8th layers' feature map to balance global and local information
            # the last layer isn't used since it's too abstract, good for classification but not for segmentation
            _, feature_map, _, _ = model.encode_image(data)
            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / (normal_text_features_ahchor.norm(dim=-1, keepdim=True) + 1e-8)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / (abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)  + 1e-8)
            abnormal_text_features = abnormal_text_features / (abnormal_text_features.norm(dim=-1, keepdim=True) + 1e-8)

            # compute similarity score
            # Similarity scores between the visual features and normal text embeddings
            l_pos = torch.einsum('nic,cj->nij', feature_map, normal_text_features_ahchor.transpose(0, 1))
            # Similarity scores between the visual features and abnormal text embeddings
            l_neg_v2t = torch.einsum('nic,cj->nij', feature_map, abnormal_text_features.transpose(0, 1))
            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scale
            
            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            target_v2t = torch.zeros([logits_v2t.shape[0], logits_v2t.shape[1]], dtype=torch.long).to(device)
            # CLIP and Prompt learning loss
            loss_v2t = criterion(logits_v2t.transpose(1, 2), target_v2t).mean()
            #compute_patchwise_cross_entropy(logits_v2t.transpose(1,2), target_v2t)
            # EAM loss
            trip_loss = criterion_tip(feature_map, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t.mean() + trip_loss + loss_match_abnormal * args.lambda1
            loss.backward()
            optimizer.step()
            #print(f"Loss {loss} divided in:", loss_v2t.mean().item(), trip_loss.item(), (loss_match_abnormal*args.lambda1).item())

        scheduler.step()
        model.build_text_feature_gallery()

        score_maps = []
        image_scores = []
        test_imgs = []
        gt_mask_list = []
        gt_list = []
        names = []
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
        
        
        """if first_image is None:
            first_image = torch.from_numpy(copy.deepcopy(score_maps[0]))
        else:
            print(torch.equal(first_image, torch.from_numpy(score_maps[0])))
        """
        test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
        if args.dataset=='brainmri':
            result_dict = metric_cal_img(image_scores, gt_list, np.array(score_maps))
        else:
            result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)
        print('metric on validation:', result_dict)

        if best_result_dict is None:
            print('no previous result')
            best_result_dict = result_dict
            save_check_point(model, check_path)
            if args.vis:
                plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)
        else:
            if args.dataset=='brainmri':
                if best_result_dict['i_roc'] < result_dict['i_roc']:
                    print('better result')
                    best_result_dict = result_dict
                    save_check_point(model, check_path)
                    if args.vis:
                        plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=(img_dir+str(epoch)))
            else:
                if best_result_dict['p_roc'] < result_dict['p_roc']:
                    print('better result')
                    best_result_dict = result_dict
                    save_check_point(model, check_path)
                    if args.vis:
                        plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)
                


    return best_result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if not kwargs['use_cpu']:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    if kwargs['dataset'] == 'brats':
        kwargs['distance_per_slice'] = 5
    else:
        kwargs['distance_per_slice'] = 0

    kwargs['inference'] = False

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the train dataloader
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    print(f"Test dataset size: {len(test_dataloader.dataset)}")
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path, train_data=train_dataloader)

    p_roc = round(metrics['p_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Pixel-AUROC:{p_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='brats', choices=['mvtec', 'visa', 'brats', 'brainmri'])
    parser.add_argument('--class_name', type=str, default='t2w')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    # ? shoud I have the same architecture ViT-L_14 of MediCLIP?
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use_cpu", type=bool, default=False)

    # prompt tuning hyper-parameter
    # number of context tokens for normal prompts, randomly initialized and optimized during training
    parser.add_argument("--n_ctx", type=int, default=4)
    # number of context tokens for abnormal prompts, concatenated to the normal prompt
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    # number of normal prompts to generate for each image
    parser.add_argument("--n_pro", type=int, default=1)
    # number of abnormal prompts to generate for each image
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=100)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--left_slice", type=int, default=0)
    parser.add_argument("--right_slice",  type=int, default=0)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
