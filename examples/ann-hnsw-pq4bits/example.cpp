#include <iostream>
#include <string>
#include <unordered_set>
#include "utils/matrix.hpp"
#include "utils/scipy_loader.hpp"
#include "ann/hnsw.hpp"



class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};


int num_rerank;
int sub_dimension;
using pecos::ann::index_type;

typedef float32_t value_type;
typedef uint64_t mem_index_type;
typedef pecos::NpyArray<value_type> scipy_npy_t;


auto npy_to_drm = [](scipy_npy_t& X_npy) -> pecos::drm_t {
    pecos::drm_t X;
    X.rows = X_npy.shape[0];
    X.cols = X_npy.shape[1];
    X.val = X_npy.array.data();
    return X;
};


template<typename MAT, typename feat_vec_t>
void run_dense(std::string data_dir , char* model_path, index_type M, index_type efC, index_type max_level, int threads, int efs) {
    // data prepare
    scipy_npy_t X_trn_npy(data_dir + "/X.trn.npy");
    scipy_npy_t X_tst_npy(data_dir + "/X.tst.npy");
    scipy_npy_t Y_tst_npy(data_dir + "/Y.tst.npy");
    auto X_trn = npy_to_drm(X_trn_npy);
    auto X_tst = npy_to_drm(X_tst_npy);
    auto Y_tst = npy_to_drm(Y_tst_npy);
    // model prepare
    index_type topk = 10;
    pecos::ann::HNSWProductQuantizer4Bits<float, feat_vec_t> indexer;
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        // if subspace_dimension is set to 0, it will use default scheme. That is,
        // if dimension <= 400, we use subspace_dimension 1, otherwise we use 2.
        indexer.train(X_trn, M, efC, 0, 200, threads, max_level);
        std::cout<< "After train" <<std::endl;
        indexer.save(model_path);
        std::cout<< "After save" <<std::endl;
        indexer.load(model_path);
    } else {
        indexer.load(model_path);
        fclose(fp);
    }

    // prepare searcher for inference
    index_type num_data = X_tst.rows;
    auto searcher = indexer.create_searcher();
    searcher.prepare_inference();


    double latency = std::numeric_limits<double>::max();
    // REPEAT 10 times and report the best result
    for (int repeat = 0; repeat < 10; repeat++) {
        double inner_latency = 0.0;
        for (index_type idx = 0; idx < num_data; idx++) {
            StopW stopw = StopW();
            auto ret_pairs = indexer.predict_single(X_tst.get_row(idx), efs, topk, searcher, num_rerank);
            double ss = stopw.getElapsedTimeMicro();
            inner_latency += ss;
        }
        latency = std::min(latency, inner_latency);
    }
    // inference and calculate recalls
    double recall = 0.0;
    for (index_type idx = 0; idx < num_data; idx++) {
        auto ret_pairs = indexer.predict_single(X_tst.get_row(idx), efs, topk, searcher, num_rerank);
        std::unordered_set<pecos::csr_t::index_type> true_indices;

        for (auto k = 0u; k < topk; k++) {
            true_indices.insert(Y_tst.get_row(idx).val[k]);  // assume Y_tst is ascendingly sorted by distance
        }
        for (auto dist_idx_pair : ret_pairs) {
            if (true_indices.find(dist_idx_pair.node_id) != true_indices.end()) {
                recall += 1.0;
            }
        }
    }
    recall = recall / num_data / topk;
    latency = latency / num_data / 1000.;
    std::cout<< recall << " : " << 1.0 / latency * 1e3 << "," <<std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = argv[1];
    std::string model_dir = argv[2];
    std::string space_name = argv[3];
    index_type M = (index_type) atoi(argv[4]);
    index_type efC = (index_type) atoi(argv[5]);
    int threads = atoi(argv[6]);
    int efs = atoi(argv[7]);
    num_rerank = atoi(argv[8]);
    sub_dimension = atoi(argv[9]);
    index_type max_level = 8;
    char model_path[2048];
    sprintf(model_path, "%s/pecos.%s.M-%d_efC-%d_t-%d_d-%d.bin", model_dir.c_str(), space_name.c_str(), M, efC, threads, sub_dimension);
    // currently only support l2
    if (space_name.compare("l2") == 0) {
        run_dense<pecos::drm_t, pecos::ann::FeatVecDenseL2Simd<float>>(data_dir, model_path, M, efC, max_level, threads, efs);
    }
}
