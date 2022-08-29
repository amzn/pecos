/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#include <algorithm>
#include <limits>
#include <vector>
#pragma once
#include "common.hpp"
#include "utils/clustering.hpp"

namespace pecos {

namespace ann {

    struct ProductQuantizer4Bits : ProductQuantizer4BitsBase {

        __attribute__((__target__("default")))
        void pack_codebook_for_inference() {
            pack_codebook_for_inference_default();
        }

        __attribute__((__target__("avx512f")))
        void pack_codebook_for_inference() {
            local_codebooks.resize(original_local_codebooks.size(), 0);
            for (index_type i = 0; i < num_local_codebooks; i++) {
                for (size_t j = 0; j < num_of_local_centroids; j++) {
                    for (int k = 0; k < local_dimension; k++) {
                        local_codebooks[i * num_of_local_centroids * local_dimension + k * num_of_local_centroids + j]
                            = original_local_codebooks[i * num_of_local_centroids * local_dimension + j * local_dimension + k];
                    }
                }
            }
        }

        __attribute__((__target__("default")))
        void pad_parameters(index_type& max_degree, size_t& code_dimension) {
            pad_parameters_default(max_degree, code_dimension);
        }

        __attribute__((__target__("avx512f"))) 
        void pad_parameters(index_type& max_degree, size_t& code_dimension) {
            //  When using AVX512f, we have 16 centroids per local codebook, and each of it uses 8 bits to represent quantized
            //  distance value. Thus, we will have 128 bits to load 1 set of local codebooks. Thus, a loadu_si512 will load
            //  512 / 128 == 4 local codebooks at a time. Thus, will need to be adjust the code_dimension to make it divisible
            //  by 4. If it's not, we have to pad 0 to extended dimensions.
            code_dimension = code_dimension % 4 == 0 ? code_dimension : (code_dimension / 4 + 1) * 4;

            // AVX512f will process 16 operations in parallel, and due to memory layout issues, to achieve fast memory load,
            // we have to process all dimensions of 16 members first then move to next 16 neighbors. Therefore, we have to make
            // sure the max_degree is a multiple of 16 so there will be no segmentation faults when reading codes from graph.
            max_degree = max_degree % 16 == 0 ? max_degree : (max_degree / 16 + 1) * 16;
        }

        __attribute__((__target__("avx512f")))
        inline void approximate_neighbor_group_distance(size_t neighbor_size, float* ds, const char* neighbor_codes, uint8_t* lut_ptr, float scale, float bias) const {
            //  When using AVX512f, we have 16 centroids per local codebook, and each of it uses 8 bits to represent quantized
            //  distance value. Thus, we will have 128 bits to load 1 set of local codebooks. Thus, a loadu_si512 will load
            //  512 / 128 == 4 local codebooks at a time. Thus, number of loadu_si512 perfomred on a group will need to be
            //  adjusted baed on if num_local_codebooks is divisible by 4.
            index_type num_dimension_block = num_local_codebooks % 4 == 0 ? num_local_codebooks / 4 : num_local_codebooks / 4 + 1;

            // Similarly, we have to parse every 16 neighbors at a time to maximally leverage avx512f.
            // Thus we can also calculate the number of group iterations based on if neighbor_size id
            // divisible by 16.
            index_type num_groups = neighbor_size % 16 == 0 ? neighbor_size / 16 : neighbor_size / 16 + 1;

            const uint8_t *localID = reinterpret_cast<const uint8_t*>(neighbor_codes);
            float* d = ds;

            for (index_type iters = 0; iters < num_groups; iters++) {
                uint8_t *local_lut_ptr = lut_ptr;
                __m512i sum_result = _mm512_setzero_si512();

                for (index_type r = 0; r < num_dimension_block; r++) {
                    __m512i lookup_table = _mm512_loadu_si512((__m512i const*)local_lut_ptr);
                    // Each time, avx512f will load 4 x 16 entries from lookup table. We
                    // prefech the position which will be accessed 8 rounds later.
                    _mm_prefetch(&localID[0] + 64 * 8, _MM_HINT_T0);
                    __m512i packedobj = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i const*)localID));
                    __m512i mask512x0F = _mm512_set1_epi16(0x000f);  // #32 of 16 bits mask
                    __m512i mask512xF0 = _mm512_set1_epi16(0x00f0);
                    __m512i lo = _mm512_and_si512(packedobj, mask512x0F);
                    __m512i hi = _mm512_slli_epi16(_mm512_and_si512(packedobj, mask512xF0), 4);
                    __m512i obj = _mm512_or_si512(lo, hi);
                    __m512i vtmp = _mm512_shuffle_epi8(lookup_table, obj);

                    sum_result = _mm512_adds_epu16(sum_result, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(vtmp, 0)));
                    sum_result = _mm512_adds_epu16(sum_result, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(vtmp, 1)));

                    // Essentially, avx512 will process 4 x 16 entries of lookup table at a time
                    // Thus, we want to move 64 positions after a round of process.
                    // The reason why neighbor id only moves 32 positions is due to the fact that
                    // 2 neighbor code is saved in a byte, and will be expanded via avx512f operations.
                    // So we only need to move 32 positions and effectively extract 64 codes.
                    local_lut_ptr += 64;
                    localID += 32;
                }

                __m512i lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(sum_result, 0));
                __m512i hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(sum_result, 1));
                __m512 distance = _mm512_cvtepi32_ps(_mm512_add_epi32(lo, hi));

                __m512 _scale = _mm512_set1_ps(scale);
                __m512 _bias = _mm512_set1_ps(bias);
                distance = _mm512_mul_ps(distance, _scale);
                distance = _mm512_add_ps(distance, _bias);

                _mm512_storeu_ps(d, distance);
                d += 16;
            }
        }

        __attribute__((__target__("default")))
        inline void approximate_neighbor_group_distance(size_t neighbor_size, float* ds, const char* neighbor_codes, uint8_t* lut_ptr, float scale, float bias) const {
            approximate_neighbor_group_distance_default(neighbor_size, ds, neighbor_codes, lut_ptr, scale, bias);
        }

        __attribute__((__target__("default")))
        inline void setup_lut(float* query, uint8_t* lut_ptr, float& scale, float& bias) const {
            setup_lut_default(query, lut_ptr, scale, bias);
        }

        __attribute__((__target__("avx512f")))
        inline void setup_lut(float* query, uint8_t* lut_ptr, float& scale, float& bias) const {
            int original_dimension = num_local_codebooks * local_dimension;

            // avx512f can load 16 float32 in parallel. Thus we need to calculate
            // how many round of computation needed to get redisual vector
            int global_blocks = original_dimension % 16 == 0 ? original_dimension / 16 : original_dimension / 16 + 1;
            int extend_dimension = global_blocks * 16;
            std::vector<float> centered_query(extend_dimension, 0);
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();

            // first iteration to calculate raw distance and max,min values for quantized lut
            const float *globalID = global_centroid.data();
            const float *localID = local_codebooks.data();
            const float *query_ptr = &centered_query[0];
            std::vector<float> raw_dist(num_local_codebooks * num_of_local_centroids, 0);

            // AVX512 on centering query
            for (int i = 0; i < global_blocks; i++) {
                __m512 q = _mm512_loadu_ps(query);
                __m512 g = _mm512_loadu_ps(globalID);
                _mm512_storeu_ps(&centered_query[i * 16], _mm512_sub_ps(q, g));
                // AVX512 read 16 floats at a time so query and globalID will move 16 positions after a round
                query += 16;
                globalID += 16;
            }

            for (index_type d = 0; d < num_local_codebooks; d++) {
                __m512 tmp_v = _mm512_setzero_ps();
                // prefect data used in the next round. It's currently decided by experience and observance of good empirical
                // results. The best prefetch position could be determined by a more complete empirical study.
                _mm_prefetch(localID + local_dimension * 16, _MM_HINT_T0);
                _mm_prefetch(raw_dist.data() + d * num_of_local_centroids, _MM_HINT_T0);

                for (int j = 0; j < local_dimension; j++) {
                    __m512 q = _mm512_set1_ps(*query_ptr++);
                    __m512 l = _mm512_loadu_ps(localID);
                    __m512 v = _mm512_sub_ps(q, l);
                    v = _mm512_mul_ps(v, v);
                    tmp_v = _mm512_add_ps(tmp_v, v);
                    // AVX512 read 16 floats at a time so locaiID will move 16 positions after a round
                    localID += 16;
                }
                _mm512_storeu_ps(&raw_dist[d * num_of_local_centroids], tmp_v);
                max = std::max(max, _mm512_reduce_max_ps(tmp_v));
                min = std::min(min, _mm512_reduce_min_ps(tmp_v));
            }

            bias = min;
            scale = (max - min) / 255.0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _bias = _mm512_set1_ps(bias);
            auto *raw_ptr = raw_dist.data();
            for (index_type d = 0; d < num_local_codebooks; d++) {
                __m512 raw_table = _mm512_loadu_ps(raw_ptr);
                raw_table = _mm512_sub_ps(raw_table, _bias);
                raw_table = _mm512_div_ps(raw_table, _scale);
                raw_table = _mm512_roundscale_ps(raw_table, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(lut_ptr), _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(raw_table)));
                // AVX512 read 16 floats at a time so raw distance table and final lookup table  will move 16 positions after a round
                raw_ptr += 16;
                lut_ptr += 16;
            }
        }
    };

}  // end of namespace ann
}  // end of namespace pecos

