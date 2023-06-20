import argparse
import scipy.sparse as smat
import numpy as np
from pecos.utils import smat_util
import sklearn
import os
import sys
from xclib.data import data_utils

def main():
    parser = argparse.ArgumentParser(description='DataPrep_forXCrepo')
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='LF-Amazon-131K')
    args = parser.parse_args()
    print(args)

    cur_dir = f'{args.work_dir}/dataset/{args.dataset}'
    
    if args.dataset in ['LF-Amazon-131K','LF-WikiSeeAlso-320K','LF-Amazon-1.3M']:
        # Read files with features and labels (old format from XMLRepo)
        features, tabels, num_samples, num_features, num_labels = data_utils.read_data(f'{cur_dir}/train.txt')
        features = features.astype(np.float32)
        sklearn.preprocessing.normalize(features,copy=False)
        smat.save_npz(f'{cur_dir}/X_bow.trn.npz',features)
        smat.save_npz(f'{cur_dir}/normalized/Y.trn.npz',tabels)
        smat.save_npz(f'{cur_dir}/raw/Y.trn.npz',tabels)
        
        features, tabels, num_samples, num_features, num_labels = data_utils.read_data(f'{cur_dir}/test.txt')
        features = features.astype(np.float32)
        sklearn.preprocessing.normalize(features,copy=False)
        smat.save_npz(f'{cur_dir}/X_bow.tst.npz',features)
        smat.save_npz(f'{cur_dir}/normalized/Y.tst.npz',tabels)
        smat.save_npz(f'{cur_dir}/raw/Y.tst.npz',tabels)
        
        TEST = data_utils.read_sparse_file(f'{cur_dir}/Yf.txt',header=True)
        sklearn.preprocessing.normalize(TEST,copy=False)
        smat.save_npz(f"{cur_dir}/Y_bow.npz",TEST)
    elif args.dataset in ['LF-Wikipedia-500K']:
        # Read files with labels (old format from XMLRepo) 
        _, tabels, num_samples, num_features, num_labels = data_utils.read_data(f'{cur_dir}/train.txt')
        smat.save_npz(f'{cur_dir}/normalized/Y.trn.npz',tabels)
        smat.save_npz(f'{cur_dir}/raw/Y.trn.npz',tabels)
        _, tabels, num_samples, num_features, num_labels = data_utils.read_data(f'{cur_dir}/test.txt')
        smat.save_npz(f'{cur_dir}/normalized/Y.tst.npz',tabels)
        smat.save_npz(f'{cur_dir}/raw/Y.tst.npz',tabels)
        
        # Read files with features (BoW, dim = 500000. The feature in the old format has dim = 2381304.) 
        X_trn = data_utils.read_sparse_file(f"{cur_dir}/trn_X_Xf.txt", header=True)
        X_trn = sklearn.preprocessing.normalize(X_trn,norm='l2')

        X_tst = data_utils.read_sparse_file(f"{cur_dir}/tst_X_Xf.txt", header=True)
        X_tst = sklearn.preprocessing.normalize(X_tst,norm='l2')

        L_bow = data_utils.read_sparse_file(f"{cur_dir}/lbl_X_Xf.txt", header=True)
        L_bow = sklearn.preprocessing.normalize(L_bow,norm='l2')

        smat.save_npz(f"{cur_dir}/Y_bow.npz",L_bow)
        smat.save_npz(f"{cur_dir}/X_bow.trn.npz",X_trn)
        smat.save_npz(f"{cur_dir}/X_bow.tst.npz",X_tst)
    else:
        raise ValueError(f'Dataset {args.dataset} is not supported yet!')
    
    
if __name__ == "__main__":
    main()
