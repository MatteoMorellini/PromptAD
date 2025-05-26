import glob
import os
import random

BRAINMRI_DIR = '../MediCLIP/data/brainmri/images'

brainmri_classes = ['normal_brain']

# * brainmri doesn't have masks, therefore I can't calculate the p-auroc and consequently 
# * I need to implement image auroc in order to save checkpoints

# category in this case is useless
def load_brainmri(category, k_shot, seed, *args):
    def load_phase(root_path, seed):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)
        for defect_type in defect_types:
            if defect_type == 'normal':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['normal'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.jpg")
                # ! remove gt_paths since I don't have masks 
                gt_tot_paths.extend([0]*len(img_paths)) 
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    test_img_path = os.path.join(BRAINMRI_DIR, 'test')
    train_img_path = os.path.join(BRAINMRI_DIR, 'train')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, seed)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, seed)

    # Remember: only normal samples available during training
    # * selected samples shuffled with the given seed, then selected the 1st one for 1-shot,
    # * from the 2nd to the 6th for 5-shot...
    seed_file = os.path.join('./datasets/seeds_brainmri', 'selected_samples_per_run.txt')
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    training_indx = []
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]
            index = strip_line.split(' ')
            training_indx = [int(item) for item in index]
            
    print(f"training indx: {training_indx}")
    selected_train_img_tot_paths = [train_img_tot_paths[k] for k in training_indx]
    selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in training_indx]
    selected_train_tot_labels = [train_tot_labels[k] for k in training_indx]
    selected_train_tot_types = [train_tot_types[k] for k in training_indx]
    print(f"selected train img paths: {selected_train_img_tot_paths}")
    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
