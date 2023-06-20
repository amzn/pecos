/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/*
* File: fm_inference.hpp
*
* Description: Provides functionality for performing PECOS FM prediction.
*
* Note about memory management: Any function that returns a matrix type has allocated memory
* and it is incumbent upon the user to deallocate that memory by calling the free_underlying_memory
* method of the matrix in question.
*/

#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <vector>
#include "utils/matrix.hpp"
#include "ann/hnsw.hpp"
#include "third_party/nlohmann_json/json.hpp"
#include "third_party/robin_hood_hashing/robin_hood.h"

namespace pecos {
    
template<class index_type, class value_type>
struct FactorizationMachineModel {
    
    typedef std::vector<value_type> dvec_t;
    typedef dense_vec_t<value_type> dvec_wrapper_t;
    typedef sparse_vec_t<index_type, value_type> svec_wrapper_t;
    typedef pecos::ann::Pair<value_type, index_type> pair_t;
    typedef pecos::ann::heap_t<pair_t, std::less<pair_t>> max_heap_t;
    
    index_type wx_size, wz_size, k_size, num_zs = 0;
    dvec_t Wx; // query feature matrix of size wx_size * k_size;
    dvec_t Wz; //  item feature matrix of size wz_size * k_size;

    dvec_t Z_embs; // [# of items, k_size]
    dvec_t Z_bias; // [#indexed_items, 1]
    
    void init(index_type wx_size, index_type wz_size, index_type k_size) {
        this->wx_size = wx_size;
        this->wz_size = wz_size;
        this->k_size = k_size;
        
        this->Wx.resize(wx_size * k_size, 0);
        this->Wz.resize(wz_size * k_size, 0);
    }
    
    void load(FILE *fp) {
        pecos::file_util::fget_multiple<index_type>(&(this->wx_size), 1, fp);
        pecos::file_util::fget_multiple<index_type>(&(this->wz_size), 1, fp);
        pecos::file_util::fget_multiple<index_type>(&(this->k_size), 1, fp);
        
        this->Wx.resize(this->wx_size * this->k_size, 0);
        pecos::file_util::fget_multiple<value_type>(&(this->Wx[0]), this->wx_size * this->k_size, fp);
        
        this->Wz.resize(this->wz_size * this->k_size, 0);
        pecos::file_util::fget_multiple<value_type>(&(this->Wz[0]), this->wz_size * this->k_size, fp);
    }
    
    void svec_x_dmat(svec_wrapper_t x, const pecos::drm_t& Y, dvec_wrapper_t xY) const {
        size_t nnz = x.get_nnz();
        // TODO: parallelize.
        for (size_t i = 0; i < nnz; ++i) {
            const auto& yi = Y.get_row(x.idx[i]);
            pecos::do_axpy(x.val[i], yi, xY);
        }
    }
    
    void smat_x_dmat(const pecos::csr_t& X, const pecos::drm_t& Y, pecos::drm_t& XY) const {
        // TODO: parallelize.
        for (size_t i = 0; i < X.rows; ++i) {
            svec_wrapper_t xi = X.get_row(i); // must add const.
            dvec_wrapper_t xyi = XY.get_row(i); // cannot use auto& here. must specify type.
            
            svec_x_dmat(xi, Y, xyi);
        }
    }
    
    float get_bias(const svec_wrapper_t& x, const pecos::drm_t& W) const {
        float bias = 0;
        
        // add ||xW||^2 term.
        dvec_t xW;
        xW.resize(this->k_size, 0);
        dvec_wrapper_t xW_(xW);
        svec_x_dmat(x, W, xW_);
        bias += pecos::do_dot_product(xW_, xW_);
        
        // subtract diagonal term.
        size_t x_nnz = x.get_nnz();
        for (size_t i = 0; i < x_nnz; ++i) {
            const auto& wi = W.get_row(x.idx[i]);
            bias -= x.val[i] * x.val[i] * pecos::do_dot_product(wi, wi);
        }
        
        bias /= 2.;
        return bias;
    }
    
    template<typename MAT>
    void build_index(const MAT& Z) {
        // Z.shape = [# of items, wz_size] -> sparse
        // Wz.shape = [wz_size, k_size] -> dense
        this->num_zs = Z.rows;
        this->Z_embs.resize(this->num_zs * this->k_size, 0);
        this->Z_bias.resize(this->num_zs, 0);
        
        // wrap shit in matrix
        pecos::drm_t Z_embs_;
        Z_embs_.rows = this->num_zs;
        Z_embs_.cols = this->k_size;
        Z_embs_.val = this->Z_embs.data();
        
        pecos::drm_t Wz_;
        Wz_.rows = this->wz_size;
        Wz_.cols = this->k_size;
        Wz_.val = this->Wz.data();
        
        // build embeddings.
        smat_x_dmat(Z, Wz_, Z_embs_);
        
        // build bias.
        for (size_t i = 0; i < this->num_zs; ++i) {
            const auto& zi = Z.get_row(i);
            this->Z_bias[i] = get_bias(zi, Wz_);
        }
    }
    
    template<typename VEC>
    value_type inference(const VEC& x, const index_type z_idx) {
        pecos::drm_t Wx_;
        Wx_.rows = this->wx_size;
        Wx_.cols = this->k_size;
        Wx_.val = this->Wx.data();
        
        // get x embedding.
        dvec_t x_emb;
        x_emb.resize(this->k_size, 0);
        dvec_wrapper_t x_emb_(x_emb);
        svec_x_dmat(x, Wx_, x_emb_);
        
        // get x bias.
        float x_bias = get_bias(x, Wx_);
        
        // wrap Z_embs.
        pecos::drm_t Z_embs_;
        Z_embs_.rows = this->num_zs;
        Z_embs_.cols = this->k_size;
        Z_embs_.val = this->Z_embs.data();
        
        dvec_wrapper_t z_emb_(Z_embs_.get_row(z_idx));
        const float z_bias = this->Z_bias[z_idx];
        
        value_type score = pecos::do_dot_product(x_emb_, z_emb_) + x_bias + z_bias;
        return score;
    }
    
    template<typename VEC>
    max_heap_t& ranking(const VEC& x, const std::vector<index_type> item_ids, const index_type topk) {
        pecos::drm_t Wx_;
        Wx_.rows = this->wx_size;
        Wx_.cols = this->k_size;
        Wx_.val = this->Wx.data();
        
        // get x embedding.
        dvec_t x_emb;
        x_emb.resize(this->k_size, 0);
        dvec_wrapper_t x_emb_(x_emb);
        svec_x_dmat(x, Wx_, x_emb_);
        
        // get x bias.
        float x_bias = get_bias(x, Wx_);
        
        // wrap Z_embs.
        pecos::drm_t Z_embs_;
        Z_embs_.rows = this->num_zs;
        Z_embs_.cols = this->k_size;
        Z_embs_.val = this->Z_embs.data();
        
        max_heap_t topk_queue;
        
        for (auto i : item_ids) {
            if (i >= this->num_zs) {
                throw std::runtime_error("Item ids should be less then total number of items.");
            }
            dvec_wrapper_t z_emb_(Z_embs_.get_row(i));
            const value_type z_bias = this->Z_bias[i];
            
            value_type score = pecos::do_dot_product(x_emb_, z_emb_) + x_bias + z_bias;
            topk_queue.emplace(score, i);
        }
        
        while (topk_queue.size() > topk) {
            topk_queue.pop();
        }
        
        return topk_queue;
    }
    
};
}  // end namespace pecos

#endif // end of __INFERENCE_H__
