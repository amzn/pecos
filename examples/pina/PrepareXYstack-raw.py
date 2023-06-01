import scipy.sparse as smat
import numpy as np
from pecos.utils import smat_util
import sklearn.preprocessing
import os
import sys
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description='PrepareXYstack-raw')
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='LF-Amazon-131K')
    args = parser.parse_args()
    print(args)
    
    data_dir= f"{args.work_dir}/dataset/{args.dataset}/raw"
    feature_dir = f"{args.work_dir}/dataset/{args.dataset}"
    
    Y = smat.load_npz(f"{data_dir}/Y.trn.npz")
    num_X, num_L = Y.shape
    
    # Preparing Y_all.npz
    I_X = smat.identity(num_X,dtype=np.float32,format='csr')
    I_L = smat.identity(num_L,dtype=np.float32,format='csr')

    X_LX = smat_util.hstack_csr([Y,I_X])
    L_XL = smat_util.hstack_csr([I_L,Y.transpose().tocsr()]) # This should be improved in the future...
    Y_all = smat_util.vstack_csr([X_LX,L_XL])
    smat.save_npz("{}/Y_all.npz".format(data_dir),Y_all)
    del Y_all, Y, I_X, I_L, X_LX, L_XL
    
    # Stack bag of word feature
    X_bow_trn = smat.load_npz(f"{feature_dir}/X_bow.trn.npz")
    Y_bow = smat.load_npz(f"{feature_dir}/Y_bow.npz")

    # Normalization!
    sklearn.preprocessing.normalize(X_bow_trn,copy=False)
    sklearn.preprocessing.normalize(Y_bow,copy=False)

    X_bow_all = smat_util.vstack_csr([X_bow_trn,Y_bow])
    smat.save_npz(f"{feature_dir}/X_bow.all.npz",X_bow_all)
    del Y_bow, X_bow_trn, X_bow_all
    
    # Prepare text file (X_all.txt), note that the order should be X|L
    with open("{}/trn.txt".format(data_dir),'r') as f:
        X_lines = f.readlines()
        assert num_X == len(X_lines)
        print(num_X,len(X_lines))
    with open("{}/output-items.txt".format(data_dir),'r') as f:
        L_lines = f.readlines()
        assert num_L == len(L_lines)
        print(num_L,len(L_lines))

    with open("{}/X_all.txt".format(data_dir),"w") as f:
        for line in tqdm(X_lines):
            f.write(line)
        for line in tqdm(L_lines):
            f.write(line)
    
    print("All Done!")
    
if __name__ == "__main__":
    main()
    
