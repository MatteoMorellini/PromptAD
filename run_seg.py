import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    datasets = [ ('brainmri', 'normal_brain')]
    shots = [1, 5, 10]

    for (dataset, classname) in datasets:
        #vis = False if dataset == 'brats' else True
        for shot in shots:
            sh_method = f'python train_seg.py ' \
                        f'--dataset {dataset} ' \
                        f'--class_name {classname} ' \
                        f'--k-shot {shot} ' \
                        f'--vis False' 

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()