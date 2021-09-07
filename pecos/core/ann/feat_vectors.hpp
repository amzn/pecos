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
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

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

    inline float do_dot_product_simd_sse(const float *x, const float *y, size_t len) {
        size_t len16  = len / 16;
        size_t len4  = len / 4;
        const float *x_end16 = x + 16 * len16;
        const float *x_end4 = x + 4 * len4;
        const float *x_end = x + len;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while(x < x_end16) {
            v1 = _mm_loadu_ps(x); x += 4;
            v2 = _mm_loadu_ps(y); y += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(x); x += 4;
            v2 = _mm_loadu_ps(y); y += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(x); x += 4;
            v2 = _mm_loadu_ps(y); y += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(x); x += 4;
            v2 = _mm_loadu_ps(y); y += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        while(x < x_end4) {
            v1 = _mm_loadu_ps(x); x += 4;
            v2 = _mm_loadu_ps(y); y += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        float PORTABLE_ALIGN32 tmp_sum[4];
        _mm_store_ps(tmp_sum, sum_prod);
        float sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3];
        while(x < x_end) {
            sum += (*x) * (*y);
            x++;
            y++;
        }
        return sum;
    }

    inline float do_dot_product_simd_avx(const float *x, const float *y, size_t len) {
        size_t len16  = len / 16;
        size_t len4  = len / 4;
        const float *x_end16 = x + 16 * len16;
        const float *x_end4 = x + 4 * len4;
        const float *x_end = x + len;

        __m256  v1_256, v2_256;
        __m256  sum_prod_256 = _mm256_set1_ps(0);

        while(x < x_end16) {
            v1_256 = _mm256_loadu_ps(x); x += 8;
            v2_256 = _mm256_loadu_ps(y); y += 8;
            sum_prod_256 = _mm256_add_ps(sum_prod_256, _mm256_mul_ps(v1_256, v2_256));

            v1_256 = _mm256_loadu_ps(x); x += 8;
            v2_256 = _mm256_loadu_ps(y); y += 8;
            sum_prod_256 = _mm256_add_ps(sum_prod_256, _mm256_mul_ps(v1_256, v2_256));
        }
        __m128 v1_128, v2_128;
        __m128 sum_prod_128 = _mm_add_ps(_mm256_extractf128_ps(sum_prod_256, 0), _mm256_extractf128_ps(sum_prod_256, 1));
        while(x < x_end4) {
            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(v1_128, v2_128));
        }
        float PORTABLE_ALIGN32 tmp_sum[4];
        _mm_store_ps(tmp_sum, sum_prod_128);
        float sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3];
        while(x < x_end) {
            sum += (*x) * (*y);
            x++;
            y++;
        }
        return sum;
    }

    inline float do_l2_distance_simd_sse(const float *x, const float *y, size_t len) {
        size_t len16  = len / 16;
        size_t len4  = len / 4;
        const float *x_end16 = x + 16 * len16;
        const float *x_end4 = x + 4 * len4;
        const float *x_end = x + len;

        __m128 v1_128, v2_128, diff_128;
        __m128 sum_prod_128 = _mm_set1_ps(0);

        while(x < x_end16) {
            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            diff_128 = _mm_sub_ps(v1_128, v2_128);
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(diff_128, diff_128));

            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            diff_128 = _mm_sub_ps(v1_128, v2_128);
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(diff_128, diff_128));

            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            diff_128 = _mm_sub_ps(v1_128, v2_128);
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(diff_128, diff_128));

            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            diff_128 = _mm_sub_ps(v1_128, v2_128);
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(diff_128, diff_128));

        }

        while(x < x_end4) {
            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            diff_128 = _mm_sub_ps(v1_128, v2_128);
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(diff_128, diff_128));
        }
        float PORTABLE_ALIGN32 tmp_sum[4];
        _mm_store_ps(tmp_sum, sum_prod_128);
        float sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3];
        while(x < x_end) {
            float diff = (*x) - (*y);
            sum += diff * diff;
            x++;
            y++;
        }
        return sum;
    }

    inline float do_l2_distance_simd_avx(const float *x, const float *y, size_t len) {
        size_t len16  = len / 16;
        size_t len4  = len / 4;
        const float *x_end16 = x + 16 * len16;
        const float *x_end4 = x + 4 * len4;
        const float *x_end = x + len;

        __m256  v1_256, v2_256, diff_256;
        __m256  sum_prod_256 = _mm256_set1_ps(0);

        while(x < x_end16) {
            v1_256 = _mm256_loadu_ps(x); x += 8;
            v2_256 = _mm256_loadu_ps(y); y += 8;
            diff_256 = _mm256_sub_ps(v1_256, v2_256);
            sum_prod_256 = _mm256_add_ps(sum_prod_256, _mm256_mul_ps(diff_256, diff_256));

            v1_256 = _mm256_loadu_ps(x); x += 8;
            v2_256 = _mm256_loadu_ps(y); y += 8;
            diff_256 = _mm256_sub_ps(v1_256, v2_256);
            sum_prod_256 = _mm256_add_ps(sum_prod_256, _mm256_mul_ps(diff_256, diff_256));
        }

        __m128 v1_128, v2_128, diff_128;
        __m128 sum_prod_128 = _mm_add_ps(_mm256_extractf128_ps(sum_prod_256, 0), _mm256_extractf128_ps(sum_prod_256, 1));
        while(x < x_end4) {
            v1_128 = _mm_loadu_ps(x); x += 4;
            v2_128 = _mm_loadu_ps(y); y += 4;
            diff_128 = _mm_sub_ps(v1_128, v2_128);
            sum_prod_128 = _mm_add_ps(sum_prod_128, _mm_mul_ps(diff_128, diff_128));
        }
        float PORTABLE_ALIGN32 tmp_sum[4];
        _mm_store_ps(tmp_sum, sum_prod_128);
        float sum = tmp_sum[0] + tmp_sum[1] + tmp_sum[2] + tmp_sum[3];
        while(x < x_end) {
            float diff = (*x) - (*y);
            sum += diff * diff;
            x++;
            y++;
        }
        return sum;
    }

    template<class VAL_T>
    struct FeatVecDenseIPSimd : FeatVecDense<VAL_T> {
        typedef FeatVecDense<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return 1.0 - do_dot_product_simd_avx(x.val, y.val, feat_dim);
        }
    };

    template<class VAL_T>
    struct FeatVecDensePtrIPSimd : FeatVecDensePtr<VAL_T> {
        typedef FeatVecDensePtr<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return 1.0 - do_dot_product_simd_avx(x.val, y.val, feat_dim);
        }
    };

    template<class VAL_T>
    struct FeatVecDenseL2Simd : FeatVecDense<VAL_T> {
        typedef FeatVecDense<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return do_l2_distance_simd_avx(x.val, y.val, feat_dim);
        }
    };

    template<class VAL_T>
    struct FeatVecDensePtrL2Simd : FeatVecDensePtr<VAL_T> {
        typedef FeatVecDensePtr<VAL_T> feat_vec_t;
        using feat_vec_t::feat_vec_t;
        static VAL_T distance(const feat_vec_t& x, const feat_vec_t& y) {
            size_t feat_dim = x.len;
            return do_l2_distance_simd_avx(x.val, y.val, feat_dim);
        }
    };

    // =============== Various Distance Functions defined for FeatVecSparse/FeatVecSparsePtr ================
    inline float do_dot_product_sparse_simd(
        const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
        const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {
        /**
         * More or less from
         * http://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
         * precomputed dictionary
         */
        const static __m128i shuffle_mask[16] = {
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 7, 6, 5, 4),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 11, 10, 9, 8),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 11, 10, 9, 8, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 11, 10, 9, 8, 7, 6, 5, 4),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 15, 14, 13, 12),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 7, 6, 5, 4),
            _mm_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 8),
            _mm_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 11, 10, 9, 8, 3, 2, 1, 0),
            _mm_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4),
            _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        };

#ifdef USE_ALIGNED
#define MM_LOAD_SI_128 _mm_load_si128
#define MM_STORE_SI_128 _mm_store_si128
#else
#define MM_LOAD_SI_128 _mm_loadu_si128
#define MM_STORE_SI_128 _mm_storeu_si128
#endif

        size_t i_a = 0, i_b = 0;
        float PORTABLE_ALIGN32 buf_x[4];
        float PORTABLE_ALIGN32 buf_y[4];
        float ret = 0;
        const static uint32_t cyclic_shift1 = _MM_SHUFFLE(0, 3, 2, 1);
        const static uint32_t cyclic_shift2 = _MM_SHUFFLE(1, 0, 3, 2);
        const static uint32_t cyclic_shift3 = _MM_SHUFFLE(2, 1, 0, 3);

        // trim lengths to be a multiple of 4
        size_t st_a = (s_a / 4) * 4;
        size_t st_b = (s_b / 4) * 4;
        if (i_a < st_a && i_b < st_b) {
            __m128i v_a = MM_LOAD_SI_128((__m128i *)&A[i_a]);
            __m128i v_b = MM_LOAD_SI_128((__m128i *)&B[i_b]);
            while (true) {
                __m128i cmp_mask_a1 = _mm_cmpeq_epi32(v_a, v_b); // pairwise comparison
                __m128i cmp_mask_a2 = _mm_cmpeq_epi32(v_a, _mm_shuffle_epi32(v_b, cyclic_shift1)); // again...
                __m128i cmp_mask_a = _mm_or_si128(cmp_mask_a1, cmp_mask_a2);
                __m128i cmp_mask_a3 = _mm_cmpeq_epi32(v_a, _mm_shuffle_epi32(v_b, cyclic_shift2)); // and again...
                cmp_mask_a = _mm_or_si128(cmp_mask_a, cmp_mask_a3);
                __m128i cmp_mask_a4 = _mm_cmpeq_epi32(v_a, _mm_shuffle_epi32(v_b, cyclic_shift3)); // and again.
                cmp_mask_a = _mm_or_si128(cmp_mask_a, cmp_mask_a4);
                // convert the 128-bit mask to the 4-bit mask
                const int mask_a = _mm_movemask_ps(*reinterpret_cast<__m128 *>(&cmp_mask_a));

                if(mask_a) {
                    __m128i cmp_mask_b1 = cmp_mask_a1; // pairwise comparison
                    __m128i cmp_mask_b2 = _mm_cmpeq_epi32(v_b, _mm_shuffle_epi32(v_a, cyclic_shift1)); // again...
                    __m128i cmp_mask_b = _mm_or_si128(cmp_mask_b1, cmp_mask_b2);
                    __m128i cmp_mask_b3 = _mm_cmpeq_epi32(v_b, _mm_shuffle_epi32(v_a, cyclic_shift2)); // and again...
                    cmp_mask_b = _mm_or_si128(cmp_mask_b, cmp_mask_b3);
                    __m128i cmp_mask_b4 = _mm_cmpeq_epi32(v_b, _mm_shuffle_epi32(v_a, cyclic_shift3)); // and again.
                    cmp_mask_b = _mm_or_si128(cmp_mask_b, cmp_mask_b4);

                    // convert the 128-bit mask to the 4-bit mask
                    const int mask_b = _mm_movemask_ps(*reinterpret_cast<__m128 *>(&cmp_mask_b));

                    const __m128i v_x = MM_LOAD_SI_128((__m128i *)&x[i_a]);
                    const __m128i v_y = MM_LOAD_SI_128((__m128i *)&y[i_b]);

                    // copy out common elements
                    const __m128i p_x = (_mm_shuffle_epi8(v_x, shuffle_mask[mask_a]));
                    const __m128i p_y = (_mm_shuffle_epi8(v_y, shuffle_mask[mask_b]));

                    _mm_storeu_si128(reinterpret_cast<__m128i *>(&buf_x[0]), p_x);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(&buf_y[0]), p_y);
                    for(int i = 0; i < _mm_popcnt_u32(mask_a); i++) {
                        ret += buf_x[i] * buf_y[i];
                    }
                }

                const uint32_t a_max = A[i_a + 3];
                if (a_max <= B[i_b + 3]) {
                    i_a += 4;
                    if (i_a >= st_a) { break; }
                    v_a = MM_LOAD_SI_128((__m128i *)&A[i_a]);
                }
                if (a_max >= B[i_b + 3]) {
                    i_b += 4;
                    if (i_b >= st_b) { break; }
                    v_b = MM_LOAD_SI_128((__m128i *)&B[i_b]);
                }
            }
        }

        // intersect the tail using scalar intersection
        if(i_a < s_a && i_b < s_b) {
            // intersect the tail using scalar intersection
            while(1) {
                while(A[i_a] < B[i_b]) {
                    if(++i_a == s_a) {
                        return ret;
                    }
                }
                while(B[i_b] < A[i_a]) {
                    if(++i_b == s_b) {
                        return ret;
                    }
                }
                if(A[i_a] == B[i_b]) {
                    ret += x[i_a] * y[i_b];
                    if(++i_a == s_a || ++i_b == s_b) {
                        return ret;
                    }
                }
            }
        }
        return ret;
    }

    inline float do_l2_distance_sparse_simd(
        const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
        const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {
        float x_sq = do_l2_distance_simd_avx(x, x, s_a);
        float y_sq = do_l2_distance_simd_avx(y, y, s_b);
        return x_sq + y_sq - 2.0 * do_dot_product_sparse_simd(s_a, x, A, s_b, y, B);
    }

    template<uint64_t step=4>
    inline float do_dot_product_sparse_block(
        const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
        const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {

        size_t i_a = 0, i_b = 0;
        float ret = 0;

        // trim lengths to be a multiple of step
        size_t st_a = (s_a / step) * step;
        size_t st_b = (s_b / step) * step;

        if(i_a < st_a && i_b < st_b) {
            while (true) {
                uint64_t PORTABLE_ALIGN32 a[step];
                uint64_t PORTABLE_ALIGN32 b[step];

                for(auto i = 0u; i < step; i++) {
                    a[i] = A[i_a + i];
                    b[i] = B[i_b + i];
                }

                auto* xx = x + i_a;
                auto* yy = y + i_b;

                for(auto i = 0u; i < step; i++){
                    for(auto j = 0u; j < step; j++){
                        if(a[i] == b[j]) {
                            ret += xx[i] * yy[j];
                            break;
                        }
                    }
                }


                // move block based on the last element in both sub arrays
                if (a[step - 1] <= b[step - 1]) {
                    i_a += step;
                    if (i_a == st_a) {
                        break;
                    }
                }
                if (a[step - 1] >= b[step - 1]) {
                    i_b += step;
                    if (i_b == st_b) {
                        break;
                    }
                }
            }
        }

        if(i_a < s_a && i_b < s_b) {
            // intersect the tail using scalar intersection
            while(1) {
                while(A[i_a] < B[i_b]) {
                    if(++i_a == s_a) {
                        return ret;
                    }
                }
                while(B[i_b] < A[i_a]) {
                    if(++i_b == s_b) {
                        return ret;
                    }
                }
                if(A[i_a] == B[i_b]) {
                    ret += x[i_a] * y[i_b];
                    if(++i_a == s_a || ++i_b == s_b) {
                        return ret;
                    }
                }
            }
        }

        return ret;
    }

    template<uint64_t step=4>
    inline float do_l2_distance_sparse_block(
        const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
        const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {

        float x_sq = do_l2_distance_simd_avx(x, x, s_a);
        float y_sq = do_l2_distance_simd_avx(y, y, s_b);
        return x_sq + y_sq - 2.0 * do_dot_product_sparse_block<step>(s_a, x, A, s_b, y, B);
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
            float x_sq = do_l2_distance_simd_avx(x.val, x.val, x.len);
            float y_sq = do_l2_distance_simd_avx(y.val, y.val, y.len);
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

        // a implementation faster than std::lower_bound
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
            float x_sq = do_l2_distance_simd_avx(x.val, x.val, x.len);
            float y_sq = do_l2_distance_simd_avx(y.val, y.val, y.len);
            return x_sq + y_sq + 2 * FeatVecSparseIPBs<IDX_T, VAL_T>::distance(x, y) - 2.0;
        }
    };


} // end of namespace ann

} // end of namespace pecos
