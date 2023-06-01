from pecos.utils.smat_util import sorted_csr, CsrEnsembler, load_matrix, Metrics
import scipy.sparse as smat
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='PrepareXYstack')
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='LF-Amazon-131K')
    parser.add_argument('--model_name', type=str, default='v0')
    parser.add_argument('--DS_model_names', type=str, default='v0,v0-s1,v0-s2', help="The DS_model_name should be seperated by ','. For example: 'v0,v0-s1,v0-s2'.")
    parser.add_argument('--feature_name', type=str, default='BoW')
    parser.add_argument('--ens_name', type=str, default='softmax', choices = ['rank', 'softmax', 'sigmoid'])
    parser.add_argument('--L_option', type=str, default='Lft_xrt')
    parser.add_argument('--Pk', type=str, default='5')
    parser.add_argument('--Use_A', type=str, default='false')
    args = parser.parse_args()
    print(args)
    

    feature_dir=f"{args.work_dir}/dataset/{args.dataset}"
    TAGS = args.DS_model_names.split(',')
    assert len(TAGS)>1 # Assume to ensemble at least 2 models!
    
    P_paths = []
    for tag in TAGS:
        P_paths.append(f"{args.work_dir}/models_LF/xtransformer/{args.dataset}/{args.model_name}/{args.feature_name}/XYstack/downstream/{tag}/{args.Pk}/{args.L_option}/P.20.npz")
        
    Y_true = sorted_csr(load_matrix(f"{args.work_dir}/dataset/{args.dataset}/raw/Y.tst.npz").tocsr())
    Y_pred = [sorted_csr(load_matrix(pp).tocsr()) for pp in P_paths]
    print("==== evaluation results ====")
    ens = getattr(CsrEnsembler, f"{args.ens_name}_average")
    cur_pred = ens(*Y_pred)
    print(Metrics.generate(Y_true, cur_pred, topk=10))
    PATH = f"{args.work_dir}/models_LF/xtransformer/{args.dataset}/{args.model_name}/{args.feature_name}/XYstack/downstream/{args.DS_model_names}/{args.Pk}/{args.L_option}"
    os.makedirs(PATH,exist_ok=True)
    smat.save_npz(f"{PATH}/P.20.{args.ens_name}.npz",cur_pred)
    print("Ensembled P matrix saved!")
    print(f"Saved model path: {PATH}")
    print(f"To evaluate, please use this path with ./scripts/Ensemble_evaluations.sh")
if __name__ == "__main__":
    main()
