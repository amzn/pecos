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

#ifndef __FM_UTILS_H__
#define __FM_UTILS_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "utils/parallel.hpp"
#include "utils/scipy_loader.hpp"
#include "utils/matrix.hpp"

namespace pecos {

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
 
    /*
    template <class VX, class M_MAT, class VY>
    void dmat_x_dvec(const M_MAT& M, const dense_vec_t<VX>& x, dense_vec_t<VY>& Mx) {
        // TODO: parallelize.
        for (size_t i = 0; i < M.cols; ++i) {
            const auto& mi = M.get_col(i);
            Mx.val[i] = pecos::do_dot_product(mi.val, x.val, M.rows);
        }
    }
    
    template <class VX, class Y_MAT, class VY>
    void svec_x_dmat(const sparse_vec_t<VX>& x, const Y_MAT& Y, dense_vec_t<VY>& xY) {
        // TODO: parallelize.
        for (size_t i = 0; i < x.nnz; ++i) {
            const auto& yi = Y.get_row(x.idx[i]);
            pecos::do_axpy(x.val[i], yi.val, xY.val, Y.cols);
        }
    }
    
    template <class X_MAT, class Y_MAT>
    void smat_x_dmat(const X_MAT& X, const Y_MAT& Y, const Y_MAT& XY) {
        // TODO: parallelize.
        for (size_t i = 0; i < X.rows; ++i) {
            auto& xi = X.get_row(i);
            auto& xyi = XY.get_row(i);
            
            svec_x_dmat(xi, Y, xyi);
        }
    }
    */
    
    template <class VX, class IX>
    inline IX get_ind(const dense_vec_t<VX>& x, const IX i) {
        return i;
    }
    
    template <class VX, class IX, class II>
    inline IX get_ind(const sparse_vec_t<IX, VX>& x, const II i) {
        return x.idx[i];
    }
    
    template <class X_VEC, class MX_VEC>
    void mat_x_vec(const drm_t& M, const X_VEC& x, MX_VEC& Mx) {
        // TODO: parallelize.
        for (size_t i = 0; i < x.get_nnz(); ++i) {
            const auto& mi = M.get_row(get_ind(x, i));
            pecos::do_axpy(x.val[i], mi.val, Mx.val, M.cols);
        }
    }
    
    template <class MAT>
    void mat_x_mat(const drm_t& M, const MAT& X, const drm_t& MX) {
        // TODO: parallelize.
        for (size_t i = 0; i < X.rows; ++i) {
            auto& xi = X.get_row(i);
            auto& mxi = MX.get_row(i);
            
            mat_x_vec(M, xi, mxi);
        }
    }
    
    template <class VEC>
    float get_bias(const VEC& x, const drm_t& M) {
        float bias = 0;
        
        // add ||xW||^2 term.
        std::vector<float> Mx;
        Mx.resize(M.cols, 0);
        
        dense_vec_t<float> Mx_(Mx);
        mat_x_vec(M, x, Mx_);
        bias += pecos::do_dot_product(Mx_.val, Mx_.val, M.cols);
        
        // subtract diagonal term.
        for (size_t i = 0; i < x.get_nnz(); ++i) {
            const auto& mi = M.get_row(get_ind(x, i));
            bias -= x.val[i] * x.val[i] * pecos::do_dot_product(mi.val, mi.val, M.cols);
        }
        
        bias /= 2.;
        return bias;
    }
    
    /*
    template <class IX, class VX, class M_MAT>
    VX get_bias(const sparse_vec_t<IX, VX>& x, const M_MAT& M) {
        VX bias = 0;
        
        // add ||xW||^2 term.
        std::vector<VX> xM;
        xM.resize(M.cols, 0);
        
        dense_vec_t<VX> xM_(xM);
        svec_x_dmat(x, M, xM_);
        bias += pecos::do_dot_product(xM_.val, xM_.val, M.cols);
        
        // subtract diagonal term.
        for (size_t i = 0; i < x.nnz; ++i) {
            const auto& mi = M.get_row(x.idx[i]);
            bias -= x.val[i] * x.val[i] * pecos::do_dot_product(mi.val, mi.val, M.cols);
        }
        
        bias /= 2.;
        return bias;
    }
    
    
    template <class VX, class M_MAT>
    VX get_bias(const dense_vec_t<VX>& x, const M_MAT& M) {
        VX bias = 0;
        
        // add ||xW||^2 term.
        std::vector<VX> Mx;
        Mx.resize(M.cols, 0);
        
        dense_vec_t<VX> Mx_(Mx);
        dmat_x_dvec(M, x, Mx_);
        bias += pecos::do_dot_product(Mx_.val, Mx_.val, M.cols);
        
        // subtract diagonal term.
        for (size_t i = 0; i < M.cols; ++i) { // k
            const auto& mi = M.get_col(i);
            for (size_t j = 0; j < M.rows; ++j) { // d
                bias -= x.val[j] * x.val[j] * mi.val[j] * mi.val[j];
            }
        }
        
        bias /= 2.;
        return bias;
    }
    */
    
} // end namespace pecos

#endif // end of __FM_UTILS_H__
