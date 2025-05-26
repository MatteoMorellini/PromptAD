import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    datasets = ['brats']
    shots = [1, 5]
    slices = [(0,20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120), (120, 140), (140, 155)]

    for shot in shots:
        for (left_slice, right_slice) in slices:
            sh_method = f'python test_seg.py ' \
                        f'--k-shot {shot} ' \
                        f'--left_slice {left_slice} ' \
                        f'--right_slice {right_slice} '

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




