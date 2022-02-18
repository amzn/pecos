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
#include "common.hpp"


inline float do_dot_product_simd(const float *x, const float *y, size_t len) {
    return do_dot_product_simd_default(x, y, len);
}

inline float do_l2_distance_simd(const float *x, const float *y, size_t len) {
    return do_l2_distance_simd_default(x, y, len);
}

inline float do_dot_product_sparse_simd(
    const size_t s_a, const float * __restrict__ x, const uint32_t * __restrict__ A,
    const size_t s_b, const float * __restrict__ y, const uint32_t * __restrict__ B) {
    return do_dot_product_sparse_simd_default(s_a, x, A, s_b, y, B);
}
