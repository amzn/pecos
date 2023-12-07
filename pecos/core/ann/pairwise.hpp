/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

#ifndef __PAIRWISE_ANN_H__
#define __PAIRWISE_ANN_H__

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ann/hnsw.hpp"
#include "third_party/nlohmann_json/json.hpp"
#include "utils/file_util.hpp"
#include "utils/mmap_util.hpp"
#include "utils/matrix.hpp"
#include "utils/random.hpp"
#include "utils/type_util.hpp"

namespace pecos {

namespace ann {

    typedef uint32_t index_type;
    typedef uint64_t mem_index_type;
    typedef float32_t value_type;

    template<typename T>
    struct is_sparse_mat {
        static const bool value = false; // compile-time constant
    };
    template<>
    struct is_sparse_mat<pecos::csr_t> {
        static const bool value = true; // compile-time constant
    };
    template<>
    struct is_sparse_mat<pecos::csc_t> {
        static const bool value = true; // compile-time constant
    };

    template<class MAT_T>
    void save_mat(
        const MAT_T &X, mmap_util::MmapStore& mmap_s,
        typename std::enable_if<is_sparse_mat<MAT_T>::value, MAT_T>::type* = 0
    ) {
        auto nnz = X.get_nnz();
        mmap_s.fput_one<index_type>(X.rows);
        mmap_s.fput_one<index_type>(X.cols);
        mmap_s.fput_one<mem_index_type>(nnz);
        mmap_s.fput_multiple<mem_index_type>(X.indptr, (X.IS_COLUMN_MAJORED ? X.cols : X.rows) + 1);
        mmap_s.fput_multiple<index_type>(X.indices, nnz);
        mmap_s.fput_multiple<value_type>(X.data, nnz);
    }

    template<class MAT_T>
    void save_mat(
        const MAT_T &X, mmap_util::MmapStore& mmap_s,
        typename std::enable_if<!is_sparse_mat<MAT_T>::value, MAT_T>::type* = 0
    ) {
        auto nnz = X.get_nnz();
        mmap_s.fput_one<index_type>(X.rows);
        mmap_s.fput_one<index_type>(X.cols);
        mmap_s.fput_one<mem_index_type>(nnz);
        mmap_s.fput_multiple<value_type>(X.val, nnz);
    }

    template<class MAT_T>
    void load_mat(MAT_T &X, mmap_util::MmapStore& mmap_s) {
        X.rows = mmap_s.fget_one<index_type>();
        X.cols = mmap_s.fget_one<index_type>();
        auto nnz = mmap_s.fget_one<mem_index_type>();
        X.indptr = mmap_s.fget_multiple<mem_index_type>((X.IS_COLUMN_MAJORED ? X.cols : X.rows) + 1);
        X.indices = mmap_s.fget_multiple<index_type>(nnz);
        X.data = mmap_s.fget_multiple<value_type>(nnz);
    }

    template<>
    void load_mat(pecos::drm_t &X, mmap_util::MmapStore& mmap_s) {
        X.rows = mmap_s.fget_one<index_type>();
        X.cols = mmap_s.fget_one<index_type>();
        auto nnz = mmap_s.fget_one<mem_index_type>();
        X.val = mmap_s.fget_multiple<value_type>(nnz);
    }

    template <typename T1, typename T2>
    struct KeyValPair {
        T1 input_key_idx;
        T2 input_key_dist;
        T2 input_label_val;
        KeyValPair(const T1& input_key_idx=T1(), const T2& input_key_dist=T2(), const T2& input_label_val=T2()):
            input_key_idx(input_key_idx), input_key_dist(input_key_dist), input_label_val(input_label_val) {}
        bool operator<(const KeyValPair<T1, T2>& other) const { return input_key_dist < other.input_key_dist; }
        bool operator>(const KeyValPair<T1, T2>& other) const { return input_key_dist > other.input_key_dist; }
    };

    // PairwiseANN Interface
    template<class FeatVec_T, class MAT_T>
    struct PairwiseANN {
        typedef FeatVec_T feat_vec_t;
        typedef MAT_T mat_t;
        typedef pecos::ann::KeyValPair<index_type, value_type> pair_t;
        typedef pecos::ann::heap_t<pair_t, std::less<pair_t>> max_heap_t;

        struct Searcher {
            typedef PairwiseANN<feat_vec_t, mat_t> pairwise_ann_t;

            const pairwise_ann_t* pairwise_ann;
            max_heap_t topk_queue;

            Searcher(const pairwise_ann_t* _pairwise_ann=nullptr): pairwise_ann(_pairwise_ann) {}

            void reset() { topk_queue.clear(); }

            max_heap_t& predict_single(const feat_vec_t& query_vec, const index_type label_key, index_type topk) {
                return pairwise_ann->predict_single(query_vec, label_key, topk, *this);
            }
        };

        Searcher create_searcher() const {
            return Searcher(this);
        }

        // scalar variables
        index_type num_input_keys;  // N
        index_type num_label_keys;  // L
        index_type feat_dim;        // d

        // matrices
        pecos::csc_t Y_csc;         // shape of [N, L]
        mat_t X_trn;                // shape of [N, d]

        // for loading memory-mapped file
        pecos::mmap_util::MmapStore mmap_store;

        // destructor
        ~PairwiseANN() {
            // If mmap_store is not open for read, then the memory of Y/X is owned by this class
            // Thus, we need to explicitly free the underlying memory of Y/X during destructor
            if (!mmap_store.is_open_for_read()) {
                this->Y_csc.free_underlying_memory();
                this->X_trn.free_underlying_memory();
            }
        }

        static nlohmann::json load_config(const std::string& filepath) {
            std::ifstream loadfile(filepath);
            std::string json_str;
            if (loadfile.is_open()) {
                json_str.assign(
                    std::istreambuf_iterator<char>(loadfile),
                    std::istreambuf_iterator<char>()
                );
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            auto j_params = nlohmann::json::parse(json_str);
            std::string cur_pairwise_ann_t = pecos::type_util::full_name<PairwiseANN>();
            std::string inp_pairwise_ann_t = j_params["pairwise_ann_t"];
            if (cur_pairwise_ann_t != inp_pairwise_ann_t) {
                throw std::invalid_argument("Inconsistent PairwiseANN_T: cur = " + cur_pairwise_ann_t  + " inp = " + inp_pairwise_ann_t);
            }
            return j_params;
        }

        void save_config(const std::string& filepath) const {
            nlohmann::json j_params = {
                {"pairwise_ann_t", pecos::type_util::full_name<PairwiseANN>()},
                {"version", "v1.0"},
                {"train_params", {
                    {"num_input_keys", num_input_keys},
                    {"num_label_keys", num_label_keys},
                    {"feat_dim", feat_dim},
                    {"nnz_of_Y", Y_csc.get_nnz()},
                    {"nnz_of_X", X_trn.get_nnz()}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if (savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void save(const std::string& model_dir) const {
            if (mkdir(model_dir.c_str(), 0777) == -1) {
                if (errno != EEXIST) {
                    throw std::runtime_error("Unable to create save folder at " + model_dir);
                }
            }
            save_config(model_dir + "/config.json");
            std::string index_path = model_dir + "/index.mmap_store";
            mmap_util::MmapStore mmap_s = mmap_util::MmapStore();
            mmap_s.open(index_path.c_str(), "w");
            // save scalar variables
            mmap_s.fput_one<index_type>(num_input_keys);
            mmap_s.fput_one<index_type>(num_label_keys);
            mmap_s.fput_one<index_type>(feat_dim);
            // save matrices
            save_mat(Y_csc, mmap_s);
            save_mat(X_trn, mmap_s);
            mmap_s.close();
        }

        void load(const std::string& model_dir, bool lazy_load = false) {
            auto config = load_config(model_dir + "/config.json");
            std::string version = config.find("version") != config.end() ? config["version"] : "not found";
            if (version == "v1.0") {
                std::string index_path = model_dir + "/index.mmap_store";
                mmap_store.open(index_path.c_str(), lazy_load ? "r_lazy" : "r");
                // load scalar variables
                num_input_keys = mmap_store.fget_one<index_type>();
                num_label_keys = mmap_store.fget_one<index_type>();
                feat_dim = mmap_store.fget_one<index_type>();
                // load matrices
                load_mat<pecos::csc_t>(Y_csc, mmap_store);
                load_mat<mat_t>(X_trn, mmap_store);
                // DO NOT call mmap_store.close() as the actual memory is held by this->mmap_store object.
            } else {
                throw std::runtime_error("Unable to load memory-mapped file with version = " + version);
            }
        }

        void train(const mat_t &X_trn, const pecos::csc_t &Y_csc) {
            // sanity check
            std::string mat_t_str = pecos::type_util::full_name<mat_t>();
            if (mat_t_str != "pecos::csr_t" && mat_t_str != "pecos::drm_t") {
                throw std::runtime_error("X_trn should be either csr_t or drm_t!");
            }
            if (X_trn.rows != Y_csc.rows) {
                throw std::runtime_error("X_trn.rows != Y_csc.rows");
            }
            // scalar variables
            this->num_input_keys = Y_csc.rows;
            this->num_label_keys = Y_csc.cols;
            this->feat_dim = X_trn.cols;
            // Deepcopy the memory of X/Y.
            // Otherwise, after Python API of PairwiseANN.train(),
            // the input matrices pX/pY can be modified or deleted.
            this->Y_csc = Y_csc.deep_copy();
            this->X_trn = X_trn.deep_copy();
        }

        max_heap_t& predict_single(
            const feat_vec_t& query_vec,
            const index_type label_key,
            index_type topk,
            Searcher& searcher
        ) const {
            searcher.reset();
            max_heap_t& topk_queue = searcher.topk_queue;

            const auto& rid_vec = this->Y_csc.get_col(label_key);
            for (index_type idx = 0; idx < rid_vec.nnz; idx++) {
                const auto input_key_idx = rid_vec.idx[idx];
                const auto input_label_val = rid_vec.val[idx];
                value_type input_key_dist = feat_vec_t::distance(query_vec, X_trn.get_row(input_key_idx));
                topk_queue.emplace(input_key_idx, input_key_dist, input_label_val);
            }
            if (topk < rid_vec.nnz) {
                while (topk_queue.size() > topk) {
                    topk_queue.pop();
                }
            }
            std::sort_heap(topk_queue.begin(), topk_queue.end());
            return topk_queue;
        }
    };

}  // end of namespace ann
}  // end of namespace pecos

#endif // end of __PAIRWISE_ANN_H__

