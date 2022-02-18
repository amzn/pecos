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
#endif

#include "common.hpp"


__attribute__((__target__("default")))
inline float do_dot_product_simd(const float *x, const float *y, size_t len) {
    return do_dot_product_simd_default(x, y, len);
}
__attribute__((__target__("sse")))
inline float do_dot_product_simd(const float *x, const float *y, size_t len) {
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
__attribute__((__target__("avx")))
inline float do_dot_product_simd(const float *x, const float *y, size_t len) {
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
__attribute__((__target__("avx512f")))
inline float do_dot_product_simd(const float *x, const float *y, size_t len) {
    size_t len16  = len / 16;
    size_t len4  = len / 4;
    const float *x_end16 = x + 16 * len16;
    const float *x_end4 = x + 4 * len4;
    const float *x_end = x + len;

    __m512  v1_512, v2_512;
    __m512  sum_prod_512 = _mm512_set1_ps(0);

    while(x < x_end16) {
        v1_512 = _mm512_loadu_ps(x); x += 16;
        v2_512 = _mm512_loadu_ps(y); y += 16;
        sum_prod_512 = _mm512_add_ps(sum_prod_512, _mm512_mul_ps(v1_512, v2_512));
    }
    __m128 v1_128, v2_128;
    __m128 sum_prod_128 = _mm_add_ps(
        _mm_add_ps(_mm512_extractf32x4_ps(sum_prod_512, 0), _mm512_extractf32x4_ps(sum_prod_512, 1)),
        _mm_add_ps(_mm512_extractf32x4_ps(sum_prod_512, 2), _mm512_extractf32x4_ps(sum_prod_512, 3))
    );

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

__attribute__((__target__("default")))
inline float do_l2_distance_simd(const float *x, const float *y, size_t len) {
    return do_l2_distance_simd_default(x, y, len);
}
__attribute__((__target__("sse")))
inline float do_l2_distance_simd(const float *x, const float *y, size_t len) {
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
__attribute__((__target__("avx")))
inline float do_l2_distance_simd(const float *x, const float *y, size_t len) {
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
__attribute__((__target__("avx512f")))
inline float do_l2_distance_simd(const float *x, const float *y, size_t len) {
    size_t len16  = len / 16;
    size_t len4  = len / 4;
    const float *x_end16 = x + 16 * len16;
    const float *x_end4 = x + 4 * len4;
    const float *x_end = x + len;

    __m512  v1_512, v2_512, diff_512;
    __m512  sum_prod_512 = _mm512_set1_ps(0);

    while(x < x_end16) {
        v1_512 = _mm512_loadu_ps(x); x += 16;
        v2_512 = _mm512_loadu_ps(y); y += 16;
        diff_512 = _mm512_sub_ps(v1_512, v2_512);
        sum_prod_512 = _mm512_add_ps(sum_prod_512, _mm512_mul_ps(diff_512, diff_512));
    }

    __m128 v1_128, v2_128, diff_128;
    __m128 sum_prod_128 = _mm_add_ps(
        _mm_add_ps(_mm512_extractf32x4_ps(sum_prod_512, 0), _mm512_extractf32x4_ps(sum_prod_512, 1)),
        _mm_add_ps(_mm512_extractf32x4_ps(sum_prod_512, 2), _mm512_extractf32x4_ps(sum_prod_512, 3))
    );

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

__attribute__((__target__("default")))
inline float do_dot_product_sparse_simd(
    const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
    const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {
    return do_dot_product_sparse_simd_default(s_a, x, A, s_b, y, B);
}

__attribute__((__target__("avx")))
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
