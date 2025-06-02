import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    # TODO: check that the model is good also at predicting the training set, especially after 0 and 1 epochs

    pool = Pool(processes=1)
    datasets = [('brats', 't2w')]
    shots = [1, 5, 10]
    for (dataset, class_name) in datasets:
        for shot in shots:
            sh_method = f'python test_seg.py ' \
                            f'--dataset {dataset} ' \
                            f'--class_name {class_name} ' \
                            f'--k-shot {shot} '\
                            f'--seed 42 ' 
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




