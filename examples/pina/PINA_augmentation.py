from pecos.xmc.xtransformer.model import XTransformer
import scipy.sparse as smat
import numpy as np
from pecos.utils import smat_util
from pecos.utils.featurization.text.preprocess import Preprocessor
import sklearn
import os
from pecos.xmc import Indexer, LabelEmbeddingFactory
import sys
from pecos.core import clib as pecos_clib
from tqdm import tqdm
import argparse



def CSR_rowwise_softmax(P):
    P.data = np.exp(P.data).astype(np.float32)
    P = sklearn.preprocessing.normalize(P, norm='l1')
    return P

def main():
    parser = argparse.ArgumentParser(description='PrepareXYstack')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--feature_name', type=str, default='BoW')
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='LF-AmazonTitles-131K')
    parser.add_argument('--L_option', type=str, default='Lft_xrt', choices=['Lf', 'Lft', 'Lf_xrt','Lft_xrt','Lxrt'])
    parser.add_argument('--Pk', type=int, default=5, help='Should be =< 20!!!')
    parser.add_argument('--Use_A', type=int, default=0, help='Use true neighbor for training data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size when applying XR-Transformer')
    parser.add_argument('--num_workers', type=int, default=48, help='number of workers XR-Transformer')
    parser.add_argument('--text_normalization', type=str, default="raw", help='Use raw or normalized text.')
    args = parser.parse_args()
    print(args)
    
    # !!! only_topk =<20 !!!
    topk = 5
    
    feature_dir=f"{args.work_dir}/dataset/{args.dataset}"
    params_path=f"{args.work_dir}/scripts/params/xtransformer/{args.dataset}/{args.model_name}.json"
    model_dir=f"{args.work_dir}/models_LF/xtransformer/{args.dataset}/{args.model_name}/{args.feature_name}/XYstack"
    
    # Remember to replace 20 with your own top k if you have modified it!
    P_trn = smat.load_npz("{}/P.20.trn.npz".format(model_dir))
    P_tst = smat.load_npz("{}/P.20.tst.npz".format(model_dir))
    
    if len(args.L_option)>2 and args.L_option[-3:]=="xrt":
        xtf = XTransformer.load(model_dir)
        
    # use softmax row-wise to turn it into a probability. Since P contains negative values...
    P_trn = smat_util.sorted_csr(P_trn,only_topk=topk)[:]
    P_tst = smat_util.sorted_csr(P_tst,only_topk=topk)[:]

    if P_trn.min()<0 or P_tst.min()<0:
        P_trn = CSR_rowwise_softmax(P_trn)
        P_tst = CSR_rowwise_softmax(P_tst)
    
    N_trn = P_trn.shape[0]
    print("{}/{}/Y_all.npz".format(feature_dir,args.text_normalization))
    Y_trn = smat.load_npz("{}/{}/Y_all.npz".format(feature_dir,args.text_normalization))[:N_trn, :]
        
    print(f"P_trn shape is {P_trn.shape}, max: {P_trn.max()}, min: {P_trn.min()}")
    print(f"P_tst shape is {P_tst.shape}, max: {P_tst.max()}, min: {P_tst.min()}")
    print(f"Y_trn shape is {Y_trn.shape}, max: {Y_trn.max()}, min: {Y_trn.min()}")
    
    
    # Get features of pretraining XMC output space
    if args.L_option == "Lft_xrt":
        # Generate xrt dense embedding for label text
        with open(f"{args.work_dir}/dataset/{args.dataset}/{args.text_normalization}/output-items.txt") as f:
            text = f.readlines()
        L_emb = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)
        
        # Generate xrt dense embedding for instance text
        with open("{}/dataset/{}/{}/X.trn.txt".format(args.work_dir,args.dataset,args.text_normalization)) as f:
            text = f.readlines()
        X_emb = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)
    
        # Prepare [L|X] for dense embedding.
        L_emb = smat.csr_matrix(L_emb,dtype=np.float32)
        X_emb = smat.csr_matrix(X_emb,dtype=np.float32)
        All_emb = smat_util.vstack_csr([L_emb,X_emb])
        # Row normalization
        All_emb = sklearn.preprocessing.normalize(All_emb,norm='l2')
        
        # Load stacked instance feature
        if args.feature_name in ['BoW']:
            X_all = smat.load_npz('{}/X_bow.all.npz'.format(feature_dir)).astype(np.float32)
        else:
            X_all = smat.load_npz('{}/X.tfidf.all.npz'.format(feature_dir)).astype(np.float32)
            
        X_all = sklearn.preprocessing.normalize(X_all,norm='l2')
        
        # Concat sparse and dense embedding
        Lf1 = smat_util.hstack_csr([X_all,All_emb])

    elif args.L_option == "Lf_xrt":
        # Use PIFA
        # # Load stacked instance feature and multilabel matrix
        if args.feature_name in ['BoW']:
            X_all = smat.load_npz('{}/X_bow.all.npz'.format(feature_dir)).astype(np.float32)
        else:
            X_all = smat.load_npz('{}/X.tfidf.all.npz'.format(feature_dir)).astype(np.float32)   
        X_all = sklearn.preprocessing.normalize(X_all,norm='l2')
        Y_all = smat.load_npz('{}/dataset/{}/{}/Y_all.npz'.format(args.work_dir,args.dataset,args.text_normalization)).astype(np.float32)
        
        # Produce PIFA embedding
        Lf1 = LabelEmbeddingFactory.create(Y_all, X_all, method="pifa")
        
        # Generate xrt dense embedding for label text and instnace text
        with open("{}/dataset/{}/{}/output-items.txt".format(args.work_dir,args.dataset,args.text_normalization)) as f:
            text = f.readlines()
        L_emb = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)
        with open("{}/dataset/{}/{}/X.trn.txt".format(args.work_dir,args.dataset,args.text_normalization)) as f:
            text = f.readlines()
        X_emb = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)

        L_emb = smat.csr_matrix(L_emb,dtype=np.float32)
        X_emb = smat.csr_matrix(X_emb,dtype=np.float32)
        All_emb = smat_util.vstack_csr([L_emb,X_emb])
        # Row normalization
        All_emb = sklearn.preprocessing.normalize(All_emb,norm='l2')
        # Concat, X|L
        Lf1 = smat_util.hstack_csr([All_emb,Lf1])

    elif args.L_option == "Lft":
        # Load stacked instance feature
        if args.feature_name in ['BoW']:
            X_all = smat.load_npz('{}/X_bow.all.npz'.format(feature_dir)).astype(np.float32)
        else:
            X_all = smat.load_npz('{}/X.tfidf.all.npz'.format(feature_dir)).astype(np.float32)
            
        Lf1 = sklearn.preprocessing.normalize(X_all,norm='l2')
        
    elif args.L_option == "Lf":
        # Load stacked instance feature
        if args.feature_name in ['BoW']:
            X_all = smat.load_npz('{}/X_bow.all.npz'.format(feature_dir)).astype(np.float32)
        else:
            X_all = smat.load_npz('{}/X.tfidf.all.npz'.format(feature_dir)).astype(np.float32)
            
        X_all = sklearn.preprocessing.normalize(X_all,norm='l2')
        Y_all = smat.load_npz('{}/dataset/{}/{}/Y_all.npz'.format(args.work_dir,args.dataset,args.text_normalization)).astype(np.float32)
        
        # Produce PIFA embedding
        Lf1 = LabelEmbeddingFactory.create(Y_all, X_all, method="pifa")
        
    elif args.L_option == "Lxrt":
        # Generate xrt dense embedding for label text and instnace text
        with open("{}/dataset/{}/{}/output-items.txt".format(args.work_dir,args.dataset,args.text_normalization)) as f:
            text = f.readlines()
        L_emb = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)
        with open("{}/dataset/{}/{}/X.trn.txt".format(args.work_dir,args.dataset,args.text_normalization)) as f:
            text = f.readlines()
        X_emb = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)

        L_emb = smat.csr_matrix(L_emb,dtype=np.float32)
        X_emb = smat.csr_matrix(X_emb,dtype=np.float32)
        Lf1 = smat_util.vstack_csr([L_emb,X_emb])

    else:
        print("Not implemented")
    
    # Apply row wise l2 normalization
    Lf1 = sklearn.preprocessing.normalize(Lf1,norm='l2')
    print(f"Feature shape for the pretraining XMC output space: {Lf1.shape}")
    
    # Prepare PINA augmentation
    
    # This allows for multi-hop generalization in the future...
    Hops_trn = []
    Hops_tst = []
    Hops_true = []
    
    # 0-Hop, also include xrt emb!!!
    if args.feature_name in ['BoW']:
        X_trn = smat.load_npz("{}/X_bow.trn.npz".format(feature_dir))
        X_tst = smat.load_npz("{}/X_bow.tst.npz".format(feature_dir))
    else:
        X_trn = smat.load_npz('{}/X.tfidf.trn.npz'.format(feature_dir)).astype(np.float32)
        X_tst = smat.load_npz('{}/X.tfidf.tst.npz'.format(feature_dir)).astype(np.float32)
        
    with open(f"{args.work_dir}/dataset/{args.dataset}/{args.text_normalization}/X.trn.txt") as f:
        text = f.readlines()
    X_emb_trn = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)

    with open(f"{args.work_dir}/dataset/{args.dataset}/{args.text_normalization}/X.tst.txt") as f:
        text = f.readlines()
    X_emb_tst = xtf.encode(text, batch_size=args.batch_size, batch_gen_workers=args.num_workers)

    X_emb_trn = smat.csr_matrix(X_emb_trn,dtype=np.float32)
    X_emb_tst = smat.csr_matrix(X_emb_tst,dtype=np.float32)
    X_emb_trn = sklearn.preprocessing.normalize(X_emb_trn,norm='l2')
    X_emb_tst = sklearn.preprocessing.normalize(X_emb_tst,norm='l2')

    X_trn = sklearn.preprocessing.normalize(X_trn,norm='l2')
    X_tst = sklearn.preprocessing.normalize(X_tst,norm='l2')

    X_trn = smat_util.hstack_csr([X_emb_trn,X_trn])
    X_tst = smat_util.hstack_csr([X_emb_tst,X_tst])
    
    X_trn = sklearn.preprocessing.normalize(X_trn,norm='l2')
    X_tst = sklearn.preprocessing.normalize(X_tst,norm='l2')
    
    Hops_trn.append(X_trn)
    Hops_tst.append(X_tst)
    Hops_true.append(X_trn)
    
    # 1-Hop
    X_trn = pecos_clib.sparse_matmul(P_trn,Lf1)
    X_tst = pecos_clib.sparse_matmul(P_tst,Lf1)
    X_true = pecos_clib.sparse_matmul(Y_trn,Lf1)

    # Apply row wise l2 normalization 
    X_trn = sklearn.preprocessing.normalize(X_trn,norm='l2')
    X_tst = sklearn.preprocessing.normalize(X_tst,norm='l2')
    X_true = sklearn.preprocessing.normalize(X_true,norm='l2')
    
    Hops_trn.append(X_trn)
    Hops_tst.append(X_tst)
    Hops_true.append(X_true)
    
    # Concat all hops.
    X_cat_trn = smat_util.hstack_csr(Hops_trn)
    X_cat_tst = smat_util.hstack_csr(Hops_tst)
    X_cat_true = smat_util.hstack_csr(Hops_true)

    # Apply row wise l2 normalization 
    X_cat_trn = sklearn.preprocessing.normalize(X_cat_trn,norm='l2')
    X_cat_tst = sklearn.preprocessing.normalize(X_cat_tst,norm='l2')
    X_cat_true = sklearn.preprocessing.normalize(X_cat_true,norm='l2')
    
    print(f"X_trn shape is {X_cat_trn.shape}, max: {X_cat_trn.max()}, min: {X_cat_trn.min()}")
    print(f"X_tst shape is {X_cat_tst.shape}, max: {X_cat_tst.max()}, min: {X_cat_tst.min()}")
    print(f"X_true shape is {X_cat_true.shape}, max: {X_cat_true.max()}, min: {X_cat_true.min()}")

    smat.save_npz(f'{model_dir}/X_trn_P{args.Pk}{args.L_option}.npz',X_cat_trn.astype(np.float32))
    smat.save_npz(f'{model_dir}/X_tst_P{args.Pk}{args.L_option}.npz',X_cat_tst.astype(np.float32))
    smat.save_npz(f'{model_dir}/X_true_{args.L_option}.npz',X_cat_true.astype(np.float32))

    print("All Set!!")
    
if __name__ == "__main__":
    main()
