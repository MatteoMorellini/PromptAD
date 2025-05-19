import glob
import os
import random 
from PIL import Image
import nibabel as nib

brats_classes = ['t1c', 't1n', 't2f', 't2w']

BRATS_DIR = '../MediCLIP/data/brats-met/images'
#BRATS_DIR = './anomaly_detection/brats_anomaly_detection'

def load_brats(category, k_shot):

    def get_is_abnormal(mask_path):
        mask = Image.open(mask_path).convert('L')
        mask = mask.point(lambda p: p > 0)
        # getbbox() returns a 4-tuple defining the left, upper, right, and lower pixel coordinate of non-zero regions.
        return mask.getbbox() is not None
    
    def filter_and_merge(lista, training_indx):
        filtered_list = [lista[k] for k in \
                                    training_indx if k in lista]
        merged_list = [item for sublist in filtered_list for item in sublist]
        return merged_list
    
    def list_non_hidden_folders(root_path):
        return [f for f in os.listdir(root_path) if not f.startswith('.') and os.path.isdir(os.path.join(root_path, f))]

    def load_phase(root_path, train = False):
        img_tot_paths = {}
        gt_tot_paths = {}
        tot_labels = {}
        tot_types = {}

        patients = list_non_hidden_folders(root_path)
        # ! while debugginng, keep a low number of patients
        patients = patients[:2]

        for patient in patients:
            img_patient_paths = []
            gt_patient_paths = []
            patient_labels = []
            patient_types = []

            patient_id = patient.split('-')[-2]
            # ? Isn't faster to use samples.json instead of listing all the images?
            images = glob.glob(os.path.join(root_path, patient, category) + '/*.jpeg')
            masks = glob.glob(os.path.join(root_path, patient, 'seg') + '/*.jpeg')
            
            for  (img_path, mask_path) in zip(images, masks):
                
                if get_is_abnormal(mask_path):
                    if train: continue
                    gt_patient_paths.append(mask_path)
                    patient_labels.append(1)
                    patient_types.append('abnormal')
                else:
                    # ? are anomalous images kept
                    gt_patient_paths.append(0)
                    patient_labels.append(0)
                    patient_types.append('normal')
                img_patient_paths.append(img_path)
                # !!! 
                if len(img_patient_paths) == 2:
                    break

            assert len(img_patient_paths) == len(gt_patient_paths), "Something wrong with test and ground truth pair!"

            img_tot_paths[patient_id] = img_patient_paths
            gt_tot_paths[patient_id] = gt_patient_paths
            tot_labels[patient_id] = patient_labels
            tot_types[patient_id] = patient_types

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in brats_classes
    
    train_img_path = os.path.join(BRATS_DIR, 'train/abnormal')
    test_img_path = os.path.join(BRATS_DIR, 'test/abnormal')
    
    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, train = True)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path)

    seed_file = os.path.join('./datasets/seeds_brats', category, 'selected_samples_per_run.txt')
    with open(seed_file, 'r') as f:
        files = f.readlines()
    begin_str = f'#{k_shot}: '

    training_indx = []
    for line in files:
        if line.count(begin_str) > 0:
            strip_line = line[len(begin_str):-1]
            index = strip_line.split(' ')
            training_indx = [item for item in index]

    selected_train_img_tot_paths = filter_and_merge(train_img_tot_paths, training_indx)
    selected_train_gt_tot_paths = filter_and_merge(train_gt_tot_paths,training_indx)
    selected_train_tot_labels = filter_and_merge(train_tot_labels, training_indx)
    selected_train_tot_types = filter_and_merge(train_tot_types, training_indx)

    test_img_tot_paths = [item for sublist in test_img_tot_paths.values() for item in sublist]
    test_gt_tot_paths = [item for sublist in test_gt_tot_paths.values() for item in sublist]
    test_tot_labels = [item for sublist in test_tot_labels.values() for item in sublist]
    test_tot_types = [item for sublist in test_tot_types.values() for item in sublist]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
