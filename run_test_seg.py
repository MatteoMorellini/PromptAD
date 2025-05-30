import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)
    # combination for inference on brats after training on brainmri: 
    # datasets = [...,('brainmri', 't2w'),...]
    datasets = [('brainmri', 'normal_brain')]
    shots = [1, 5, 10]
    slices = [(0,20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 140), (140, 155)]
    for (dataset, class_name) in datasets:
        for shot in shots:
            for (left_slice, right_slice) in slices:
                sh_method = f'python test_seg.py ' \
                            f'--dataset {dataset} ' \
                            f'--class_name {class_name} ' \
                            f'--k-shot {shot} ' \
                            f'--left_slice {left_slice} ' \
                            f'--right_slice {right_slice} '

                print(sh_method)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




