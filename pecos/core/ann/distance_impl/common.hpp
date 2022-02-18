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

// SIMD default functions
inline float do_dot_product_simd_default(const float *x, const float *y, size_t len) {
    float sum = 0;
    for(size_t i = 0; i < len; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

inline float do_l2_distance_simd_default(const float *x, const float *y, size_t len) {
    float sum = 0.0;
    for(size_t i = 0; i < len; i++) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}

inline float do_dot_product_sparse_simd_default(
    const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
    const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {
    return do_dot_product_sparse_block<4>(s_a, x, A, s_b, y, B);
}
