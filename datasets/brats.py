import glob
import os
import random 
from PIL import Image
import nibabel as nib
from collections import defaultdict

brats_classes = ['t1c', 't1n', 't2f', 't2w']

BRATS_DIR = '../MediCLIP/data/brats-met/images'
#BRATS_DIR = './anomaly_detection/brats_anomaly_detection'

def load_brats(category, k_shot, seed, distance_per_slice):

    def get_is_abnormal(mask_path):
        mask = Image.open(mask_path).convert('L')
        mask = mask.point(lambda p: p > 0)
        # getbbox() returns a 4-tuple defining the left, upper, right, and lower pixel coordinate of non-zero regions.
        return mask.getbbox() is not None
    
    def list_non_hidden_folders(root_path):
        return [f for f in os.listdir(root_path) if not f.startswith('.') and os.path.isdir(os.path.join(root_path, f))]

    def load_phase(root_path, seed, distance_per_slice, train = False):
        img_tot_paths = {}
        gt_tot_paths = {}
        tot_labels = {}
        tot_types = {}

        patients = list_non_hidden_folders(root_path)
        random.shuffle(patients, random.seed(seed))
        if not train:
            patients = patients[:10]
            distance_per_slice *= 2
        for patient in patients:
            img_patient_paths = []
            gt_patient_paths = []
            patient_labels = []
            patient_types = []

            patient_id = patient.split('-')[-2]
            # ? Isn't faster to use samples.json instead of listing all the images?
            images = sorted(glob.glob(os.path.join(root_path, patient, category) + '/*.jpeg'))
            masks = sorted(glob.glob(os.path.join(root_path, patient, 'seg') + '/*.jpeg'))

            for  i, (img_path, mask_path) in enumerate(zip(images, masks)):
                if i % distance_per_slice != 0: continue
                if get_is_abnormal(mask_path):
                    if train: continue
                    gt_patient_paths.append(mask_path)
                    patient_labels.append(1)
                    patient_types.append('abnormal')
                else:
                    gt_patient_paths.append(0)
                    patient_labels.append(0)
                    patient_types.append('normal')
                img_patient_paths.append(img_path)

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
    train_tot_types = load_phase(train_img_path, seed, distance_per_slice, train = True)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, seed, distance_per_slice)

    keys = list(train_img_tot_paths.keys())
    random.shuffle(keys)

    selected_train_img_tot_paths = defaultdict(list)
    selected_train_gt_tot_paths = defaultdict(list)
    selected_train_tot_labels = defaultdict(list)
    selected_train_tot_types = defaultdict(list)

    for key in keys:
        slices = train_img_tot_paths[key]
        for i, slice in enumerate(slices):
            id_slice = slice.split('/')[-1].split('.')[0]
            if len(selected_train_img_tot_paths[id_slice]) < k_shot:
                selected_train_img_tot_paths[id_slice].append(slice)
                selected_train_gt_tot_paths[id_slice].append(train_gt_tot_paths[key][i])
                selected_train_tot_labels[id_slice].append(train_tot_labels[key][i])
                selected_train_tot_types[id_slice].append(train_tot_types[key][i])

    selected_train_img_tot_paths = [item for sublist in selected_train_img_tot_paths.values() for item in sublist]
    selected_train_gt_tot_paths = [item for sublist in selected_train_gt_tot_paths.values() for item in sublist]
    selected_train_tot_labels = [item for sublist in selected_train_tot_labels.values() for item in sublist]
    selected_train_tot_types = [item for sublist in selected_train_tot_types.values() for item in sublist]

    test_img_tot_paths = [item for sublist in test_img_tot_paths.values() for item in sublist]
    test_gt_tot_paths = [item for sublist in test_gt_tot_paths.values() for item in sublist]
    test_tot_labels = [item for sublist in test_tot_labels.values() for item in sublist]
    test_tot_types = [item for sublist in test_tot_types.values() for item in sublist]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels, selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
