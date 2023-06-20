#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include "utils/matrix.hpp"
#include "utils/scipy_loader.hpp"

#include "xmc/fm_solver.hpp"
#include "xmc/fm_inference.hpp"

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

int main(int argc, char** argv) {
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));
    
    // arguments
    std::string model_path = args[1];
    std::string X_data_path = args[2];
    std::string Z_data_path = args[3];
    std::string emb_dir = args[4];
    
    std::string X_emb_path = emb_dir + "/X.emb";
    std::string Z_emb_path = emb_dir + "/Z.emb";
    std::string X_bias_path = emb_dir + "/X.bias";
    std::string Z_bias_path = emb_dir + "/Z.bias";
    
    // data loading.
    scipy_npz_t X_npz(X_data_path);
    scipy_npz_t Z_npz(Z_data_path);
    
    auto X = npz_to_csr(X_npz); // pecos::csr_t, [n_trn, d]
    auto Z = npz_to_csr(Z_npz);
    
    std::cout << X.rows << " " << X.cols << std::endl;
    std::cout << Z.rows << " " << Z.cols << std::endl;
    
    // load inference model.
    typedef typename pecos::FactorizationMachineModel<uint32_t, float> fm_t;
    fm_t fm;
    
    FILE *fp;
    fp = fopen(&model_path[0], "rb");
    fm.load(fp);
    fclose(fp);
    
    fm.build_index(Z);
    
    // build X embs and bias
    std::vector<float> X_embs;
    std::vector<float> X_bias;
    
    X_embs.resize(X.rows * fm.k_size, 0);
    X_bias.resize(X.rows, 0);

    pecos::drm_t X_embs_;
    X_embs_.rows = X.rows;
    X_embs_.cols = fm.k_size;
    X_embs_.val = X_embs.data();

    pecos::drm_t Wx_;
    Wx_.rows = fm.wx_size;
    Wx_.cols = fm.k_size;
    Wx_.val = fm.Wx.data();

    fm.smat_x_dmat(X, Wx_, X_embs_);

    for (size_t i = 0; i < X.rows; ++i) {
        const auto& xi = X.get_row(i);
        X_bias[i] = fm.get_bias(xi, Wx_);
    }
    
    // test for correctness
    pecos::drm_t Z_embs_;
    Z_embs_.rows = Z.rows;
    Z_embs_.cols = fm.k_size;
    Z_embs_.val = fm.Z_embs.data();
    
    // cout << pecos::do_dot_product(X_embs_.get_row(0), Z_embs_.get_row(0)) + X_bias[0] + fm.Z_bias[0] << endl;
    // cout << fm.inference(X.get_row(0), 0) << endl;
    
    // write.
    fp = fopen(&X_emb_path[0], "wb");
    pecos::file_util::fput_multiple<uint32_t>(&(X_embs_.rows), 1, fp);
    pecos::file_util::fput_multiple<uint32_t>(&(X_embs_.cols), 1, fp);

    pecos::file_util::fput_multiple<float>(&(X_embs[0]), X_embs.size(), fp);
    fclose(fp);
    
    fp = fopen(&X_bias_path[0], "wb");
    pecos::file_util::fput_multiple<uint32_t>(&(X_embs_.rows), 1, fp);

    pecos::file_util::fput_multiple<float>(&(X_bias[0]), X_bias.size(), fp);
    fclose(fp);
    
    fp = fopen(&Z_emb_path[0], "wb");
    pecos::file_util::fput_multiple<uint32_t>(&(fm.num_zs), 1, fp);
    pecos::file_util::fput_multiple<uint32_t>(&(fm.k_size), 1, fp);

    pecos::file_util::fput_multiple<float>(&(fm.Z_embs[0]), fm.Z_embs.size(), fp);
    fclose(fp);
    
    fp = fopen(&Z_bias_path[0], "wb");
    pecos::file_util::fput_multiple<uint32_t>(&(fm.num_zs), 1, fp);

    pecos::file_util::fput_multiple<float>(&(fm.Z_bias[0]), fm.Z_bias.size(), fp);
    fclose(fp);
}
