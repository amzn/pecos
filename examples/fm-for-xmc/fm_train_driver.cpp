#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#if defined USEOMP
#include <omp.h>
#endif

#include "utils/matrix.hpp"
#include "utils/scipy_loader.hpp"
#include "xmc/fm_solver.hpp"

typedef float32_t value_type;
typedef uint64_t mem_index_type;
typedef pecos::NpyArray<value_type> scipy_npy_t;
typedef pecos::ScipySparseNpz<true, value_type> scipy_npz_t;

using namespace std;

auto npz_to_csr = [](scipy_npz_t& X_npz) -> pecos::csr_t {
    pecos::csr_t X;
    X.rows = X_npz.shape[0];
    X.cols = X_npz.shape[1];
    X.indices = X_npz.indices.array.data();
    X.indptr = X_npz.indptr.array.data();
    X.val = X_npz.data.array.data();
    return X;
};

auto npy_to_drm = [](scipy_npy_t& X_npy) -> pecos::drm_t {
    pecos::drm_t X;
    X.rows = X_npy.shape[0];
    X.cols = X_npy.shape[1];
    X.val = X_npy.array.data();
    return X;
};

int main(int argc, char** argv) {
    // parse CLI arguments.
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    pecos::fm_solver::FMParameter param;
    uint64_t k_size = 4;
    std::string X_trn_path, Y_trn_path, X_tst_path, Y_tst_path, Z_path;
    std::string model_path = "";
    std::string Y_prefix = "";
    bool is_emb_dense = false;

#ifdef USEOMP
    int n_threads = 1;
#endif

    int i = 1;
    for(; i < argc; i++) {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            param.max_iter = atoi(args[i].c_str());
            if(param.max_iter <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        } else if(args[i].compare("-k") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            k_size = atoi(args[i].c_str());
            if(k_size <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        } else if(args[i].compare("-r") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify eta after -r");
            i++;
            param.eta = atof(args[i].c_str());
            if(param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        } else if(args[i].compare("--prefix") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify prefix of Y after --prefix");
            i++;
            Y_prefix = "." + args[i];
        } else if(args[i].compare("-l") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify lambda after -l");
            i++;
            param.reg = atof(args[i].c_str());
            if(param.reg < 0)
                throw invalid_argument("regularization cost should not be smaller than zero");
        } else if(args[i].compare("--auto-stop") == 0) {
            param.auto_stop = true;
        } else if(args[i].compare("--factorized") == 0) {
            param.factorized = true;
        } else if(args[i].compare("--identity_biased_init") == 0) {
            param.identity_biased_init = true;
        } else if(args[i].compare("--dense") == 0) {
            is_emb_dense = true;
        }
#ifdef USEOMP
        else if(args[i].compare("--n_threads") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify number of threads after --n_threads");
            i++;
            n_threads = atoi(args[i].c_str());
            if(n_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        }
#endif
        else {
            break;
        }
    }

    X_trn_path = string(args[i]);
    i++;
    Y_trn_path = string(args[i]);
    i++;
    X_tst_path = string(args[i]);
    i++;
    Y_tst_path = string(args[i]);
    i++;
    Z_path = string(args[i]);
    i++;
    
    model_path = string(args[i]);
    
#ifdef USEOMP
    // set threads
    omp_set_num_threads(n_threads);
#endif
    
    typedef typename pecos::fm_solver::FMWorker<uint32_t, float> fm_worker_t;
    fm_worker_t fmw;
    
    const int seed = 1126;
    
    if (is_emb_dense) {
        // load training data.
        scipy_npy_t X_trn_npy(X_trn_path);
        scipy_npz_t Y_trn_npz(Y_trn_path);

        auto X_trn = npy_to_drm(X_trn_npy); // pecos::drm_t, [n_trn, d]
        // auto Z_trn = npy_to_drm(Z_trn_npy);
        auto Y_trn = npz_to_csr(Y_trn_npz);

        std::cout << "X_trn.shape = (" << X_trn.rows << "," << X_trn.cols << ")" << std::endl;
        std::cout << "Y_trn.shape = (" << Y_trn.rows << "," << Y_trn.cols << ")" << std::endl;
        
        // load testing data.
        scipy_npy_t X_val_npy(X_tst_path);
        scipy_npz_t Y_val_npz(Y_tst_path);

        auto X_val = npy_to_drm(X_val_npy); // pecos::csr_t, [n_trn, d]
        auto Y_val = npz_to_csr(Y_val_npz);

        std::cout << "X_val.shape = (" << X_val.rows << "," << X_val.cols << ")" << std::endl;
        std::cout << "Y_val.shape = (" << Y_val.rows << "," << Y_val.cols << ")" << std::endl;
        
        scipy_npy_t Z_npy(Z_path);
        auto Z = npy_to_drm(Z_npy);
        std::cout << "Z.shape = (" << Z.rows << "," << Z.cols << ")" << std::endl;
        
        // train
        const uint64_t wx_size = X_trn.cols;
        const uint64_t wz_size = Z.cols;
        fmw.init(wx_size, wz_size, k_size, &param);
        fmw.solve(X_trn, Z, Y_trn, X_val, Z, Y_val, seed);
    }
    else {
        // load training data.
        scipy_npz_t X_trn_npz(X_trn_path);
        // scipy_npz_t Z_trn_npz(data_dir + "/Z.trn.npz");
        scipy_npz_t Y_trn_npz(Y_trn_path);

        auto X_trn = npz_to_csr(X_trn_npz); // pecos::csr_t, [n_trn, d]
        auto Y_trn = npz_to_csr(Y_trn_npz);

        std::cout << "X_trn.shape = (" << X_trn.rows << "," << X_trn.cols << ")" << std::endl;
        std::cout << "Y_trn.shape = (" << Y_trn.rows << "," << Y_trn.cols << ")" << std::endl;

        // load testing data.
        scipy_npz_t X_val_npz(X_tst_path);
        scipy_npz_t Y_val_npz(Y_tst_path);

        auto X_val = npz_to_csr(X_val_npz); // pecos::csr_t, [n_trn, d]
        auto Y_val = npz_to_csr(Y_val_npz);

        std::cout << "X_val.shape = (" << X_val.rows << "," << X_val.cols << ")" << std::endl;
        std::cout << "Y_val.shape = (" << Y_val.rows << "," << Y_val.cols << ")" << std::endl;
        
        scipy_npz_t Z_npz(Z_path);
        auto Z = npz_to_csr(Z_npz);
        std::cout << "Z.shape = (" << Z.rows << "," << Z.cols << ")" << std::endl;
        
        // train
        const uint64_t wx_size = X_trn.cols;
        const uint64_t wz_size = Z.cols;
        fmw.init(wx_size, wz_size, k_size, &param);
        fmw.solve(X_trn, Z, Y_trn, X_val, Z, Y_val, seed);
    }
    
    // save model
    FILE *fp;
    fp = fopen(&model_path[0], "w");
    fmw.save(fp);
    
    return 0;
}
