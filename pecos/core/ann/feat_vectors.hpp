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

#pragma once
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#include "distance.hpp"

namespace pecos {

namespace ann {

    // assuming the num_node < 2^31-1 (~4.2 billion)
    typedef uint32_t index_type;

    struct FixedSizedVec {
        typedef std::true_type is_fixed_size;
    };

    struct VariableSizedVec {
        typedef std::false_type is_fixed_size;
    };

    // =============== Various Storage ================
    template<class VAL_T>
    struct FeatVecDense : FixedSizedVec {
        typedef VAL_T value_type;
        typedef FeatVecDense<VAL_T> feat_vec_t;

        index_type len;
        value_type* val;

        // A wrapper to interpret a dens_vec_t
        FeatVecDense(dense_vec_t<value_type> v): len(v.len), val(v.val) {}

        // A wrapper to interpret memory starting from memory_ptr
        FeatVecDense(void* memory_ptr) {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            len = *reinterpret_cast<index_type*>(curr_ptr);
            curr_ptr += sizeof(index_type);
            val = reinterpret_cast<value_type*>(curr_ptr);
        }

        // number of bytes required to store this feature vector
        size_t memory_size() const { return sizeof(value_type) * (len) + sizeof(index_type); }

        // copy the content of this feature vector to this memory address
        void copy_to(void* memory_ptr) const {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            std::memcpy(curr_ptr, &len, sizeof(index_type));
            curr_ptr += sizeof(index_type);
            std::memcpy(curr_ptr, val, sizeof(value_type) * (len));
        }
    };

    template<class VAL_T>
    struct FeatVecDensePtr : FixedSizedVec {
        typedef VAL_T value_type;
        typedef FeatVecDensePtr<VAL_T> feat_vec_t;

        index_type len;
        value_type* val;

        FeatVecDensePtr(dense_vec_t<value_type> v): len(v.len), val(v.val) {}

        FeatVecDensePtr(void* memory_ptr) {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            len = *reinterpret_cast<index_type*>(curr_ptr);
            curr_ptr += sizeof(index_type);
            val = *reinterpret_cast<value_type**>(curr_ptr);
        }

        size_t memory_size() const { return sizeof(index_type) + sizeof(value_type*); }

        void copy_to(void* memory_ptr) const {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            std::memcpy(curr_ptr, &len, sizeof(index_type));
            curr_ptr += sizeof(index_type);
            std::memcpy(curr_ptr, &val, sizeof(value_type*));
        }
    };

    template<class IDX_T, class VAL_T>
    struct FeatVecSparse : VariableSizedVec {
        typedef VAL_T value_type;
        typedef IDX_T index_type;
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;

        index_type len;
        value_type* val;
        index_type* idx;

        FeatVecSparse(pecos::sparse_vec_t<index_type, value_type> v): len(v.nnz), val(v.val), idx(v.idx) {}

        FeatVecSparse(void* memory_ptr) {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            len = *reinterpret_cast<index_type*>(curr_ptr);
            curr_ptr += sizeof(index_type);
            val = reinterpret_cast<value_type*>(curr_ptr);
            curr_ptr += sizeof(value_type) * (len);
            idx = reinterpret_cast<index_type*>(curr_ptr);
        }

        size_t memory_size() const { return (sizeof(value_type) + sizeof(index_type)) * (len) + sizeof(index_type); }

        void copy_to(void* memory_ptr) const {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            std::memcpy(curr_ptr, &len, sizeof(index_type));
            curr_ptr += sizeof(index_type);
            std::memcpy(curr_ptr, val, sizeof(value_type) * (len));
            curr_ptr +=  sizeof(value_type) * (len);
            std::memcpy(curr_ptr, idx, sizeof(index_type) * (len));
        }
    };

    // =============== Various Distance Functions defined for FeatVecDense/FeatVecDensePtr ================
    template<class VAL_T>
    struct FeatVecDenseIPSimd : FeatVecDense<VAL_T> {
        typedef FeatVecDense<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return 1.0 - do_dot_product_simd(x.val, y.val, feat_dim);
        }
    };

    template<class VAL_T>
    struct FeatVecDensePtrIPSimd : FeatVecDensePtr<VAL_T> {
        typedef FeatVecDensePtr<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return 1.0 - do_dot_product_simd(x.val, y.val, feat_dim);
        }
    };

    template<class VAL_T>
    struct FeatVecDenseL2Simd : FeatVecDense<VAL_T> {
        typedef FeatVecDense<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return do_l2_distance_simd(x.val, y.val, feat_dim);
        }
    };

    template<class VAL_T>
    struct FeatVecDensePtrL2Simd : FeatVecDensePtr<VAL_T> {
        typedef FeatVecDensePtr<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return do_l2_distance_simd(x.val, y.val, feat_dim);
        }
    };

    // =============== Various Distance Functions defined for FeatVecSparse/FeatVecSparsePtr ================

    template<uint64_t step=4>
    inline float do_l2_distance_sparse_block(
        const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
        const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {

        float x_sq = do_l2_distance_simd(x, x, s_a);
        float y_sq = do_l2_distance_simd(y, y, s_b);
        return x_sq + y_sq - 2.0 * do_dot_product_sparse_block<step>(s_a, x, A, s_b, y, B);
    }

    inline float do_l2_distance_sparse_simd(
        const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
        const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {
        float x_sq = do_l2_distance_simd(x, x, s_a);
        float y_sq = do_l2_distance_simd(y, y, s_b);
        return x_sq + y_sq - 2.0 * do_dot_product_sparse_simd(s_a, x, A, s_b, y, B);
    }

    template<class IDX_T, class VAL_T>
    struct FeatVecSparseIPSimd : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            return 1.0 - do_dot_product_sparse_simd(x.len, x.val, x.idx, y.len, y.val, y.idx);
        }
    };

    template<class IDX_T, class VAL_T>
    struct FeatVecSparseL2Simd : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            return do_l2_distance_sparse_simd(x.len, x.val, x.idx, y.len, y.val, y.idx);
        }
    };

    template<class IDX_T, class VAL_T>
    struct FeatVecSparseIPBlock : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            return 1.0 - do_dot_product_sparse_block<4>(x.len, x.val, x.idx, y.len, y.val, y.idx);
        }
    };

    template<class IDX_T, class VAL_T>
    struct FeatVecSparseL2Block : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            return do_l2_distance_sparse_block<4>(x.len, x.val, x.idx, y.len, y.val, y.idx);
        }
    };

    // Sparse Inner Product by Marching Pointers
    template<class IDX_T, class VAL_T>
    struct FeatVecSparseIPMp : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            auto x_end = x.idx + (x.len);
            auto y_end = y.idx + (y.len);
            auto x_cur = x.idx;
            auto y_cur = y.idx;
            float32_t ret = 0;

            while(1) {
                while(*x_cur < *y_cur) {
                    if(++x_cur == x_end) {
                        return 1.0 - ret;
                    }
                }
                while(*y_cur < *x_cur) {
                    if(++y_cur == y_end) {
                        return 1.0 - ret;
                    }
                }
                if(*x_cur == *y_cur) {
                    ret += x.val[std::distance(x.idx, x_cur)] * y.val[std::distance(y.idx, y_cur)];
                    if(++x_cur == x_end || ++y_cur == y_end) {
                        return 1.0 - ret;
                    }
                }
            }
            return 1.0 - ret;
        }
    };

    template<class IDX_T, class VAL_T>
    struct FeatVecSparseL2Mp : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            float x_sq = do_l2_distance_simd(x.val, x.val, x.len);
            float y_sq = do_l2_distance_simd(y.val, y.val, y.len);
            return x_sq + y_sq + 2 * FeatVecSparseIPMp<IDX_T, VAL_T>::distance(x, y) - 2.0;
        }
    };

    // Sparse Inner Product by Binary Search
    template<class IDX_T, class VAL_T>
    struct FeatVecSparseIPBs : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const FeatVecSparse<IDX_T, VAL_T>& x, const FeatVecSparse<IDX_T, VAL_T>& y) {
            auto x_end = x.idx + (x.len);
            auto y_end = y.idx + (y.len);
            auto x_cur = x.idx;
            auto y_cur = y.idx;
            float32_t ret = 0;

            while(x_cur != x_end && y_cur != y_end) {
                x_cur = lower_bound(x_cur, x_end, *y_cur);
                if(x_cur == x_end) {
                    return 1.0 - ret;
                }
                y_cur = lower_bound(y_cur, y_end, *x_cur);
                if(y_cur == y_end) {
                    return 1.0 - ret;
                }
                if(*x_cur == *y_cur) {
                    ret += x.val[std::distance(x.idx, x_cur)] * y.val[std::distance(y.idx, y_cur)];
                    if(++x_cur == x_end || ++y_cur == y_end) {
                        return 1.0 - ret;
                    }
                }
            }
            return 1.0 - ret;
        }

        // an implementation faster than std::lower_bound
        template<class T>
        static T* lower_bound(T* first, T* last, const T& key) {
            intptr_t n = last - first;
            intptr_t s = -1;
            intptr_t t = n;
            intptr_t step = 1;
            while(t - s > 1) {
                intptr_t s_step = s + step;
                t = (first[s_step] < key)? t : s_step;
                s = (first[s_step] < key)? s_step : s;
                step = (first[s_step] < key)? (step << 1) : 1;
                step = ((s + step) < t) ? step : 1;
            }

            return first + s + 1;
        }
    };

    template<class IDX_T, class VAL_T>
    struct FeatVecSparseL2Bs : FeatVecSparse<IDX_T, VAL_T> {
        typedef FeatVecSparse<IDX_T, VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const FeatVecSparse<IDX_T, VAL_T>& x, const FeatVecSparse<IDX_T, VAL_T>& y) {
            float x_sq = do_l2_distance_simd(x.val, x.val, x.len);
            float y_sq = do_l2_distance_simd(y.val, y.val, y.len);
            return x_sq + y_sq + 2 * FeatVecSparseIPBs<IDX_T, VAL_T>::distance(x, y) - 2.0;
        }
    };


} // end of namespace ann

} // end of namespace pecos
