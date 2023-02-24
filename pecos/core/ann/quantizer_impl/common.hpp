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

#include <algorithm>
#include <limits>
#include <vector>
#include "utils/clustering.hpp"
#include "utils/mmap_util.hpp"

namespace pecos {

namespace ann {

    typedef uint32_t index_type;
    typedef uint64_t mem_index_type;

    struct ProductQuantizer4BitsBase {
        // num_of_local_centroids denotes number of cluster centers used in quantization
        // In 4 Bit case, it's a fixed to be 16
        const index_type num_of_local_centroids = 16;
        // num_local_codebooks denotes number of local codebooks we have or in other words,
        // number of subspace we have in Product Quantization.
        // Supposedly, num_local_codebooks * local_dimension equals dimension of original data vector
        index_type num_local_codebooks;
        // local dimension denotes the dimensionality of subspace in Product Quantization
        index_type local_dimension;
        mmap_util::MmapableVector<float> global_centroid;
        mmap_util::MmapableVector<float> local_codebooks;
        mmap_util::MmapableVector<float> original_local_codebooks;

        inline void save(mmap_util::MmapStore& mmap_s) const {
            mmap_s.fput_one<index_type>(this->num_local_codebooks);
            mmap_s.fput_one<index_type>(this->local_dimension);
            this->global_centroid.save_to_mmap_store(mmap_s);
            this->local_codebooks.save_to_mmap_store(mmap_s);
            this->original_local_codebooks.save_to_mmap_store(mmap_s);
        }

        inline void load(mmap_util::MmapStore& mmap_s) {
            this->num_local_codebooks = mmap_s.fget_one<index_type>();
            this->local_dimension = mmap_s.fget_one<index_type>();
            this->global_centroid.load_from_mmap_store(mmap_s);
            this->local_codebooks.load_from_mmap_store(mmap_s);
            this->original_local_codebooks.load_from_mmap_store(mmap_s);
        }

        inline void pack_codebook_for_inference_default() {
            local_codebooks = original_local_codebooks;
        }

        inline void pad_parameters_default(index_type& max_degree, size_t& code_dimension) {}

        inline void approximate_neighbor_group_distance_default(size_t neighbor_size, float* ds, const char* neighbor_codes, uint8_t* lut_ptr, float scale, float bias) const {
            index_type num_groups = neighbor_size % 16 == 0 ? neighbor_size / 16 : neighbor_size / 16 + 1;

            std::vector<uint32_t> d(num_of_local_centroids);
            int ptr = 0;

            const uint8_t *localID = reinterpret_cast<const uint8_t*>(neighbor_codes);
            for (index_type iters = 0; iters < num_groups; iters++) {
                memset(d.data(), 0, sizeof(uint32_t) * num_of_local_centroids);
                uint8_t* local_lut_ptr = lut_ptr;
                for (index_type m = 0; m < num_local_codebooks; m++) {
                    for (index_type k = 0; k < num_of_local_centroids; k++) {
                        uint8_t obj = *localID;
                        if (k % 2 == 0) {
                            obj &= 0x0f;
                        } else {
                            obj >>= 4;
                            localID++;
                        }
                        d[k] += *(local_lut_ptr + obj);
                    }

                    local_lut_ptr += num_of_local_centroids;
                }
                for (index_type k = 0; k < num_of_local_centroids; k++) {
                    ds[k + ptr] =  d[k] * scale + bias;
                }
                ptr += num_of_local_centroids;
            }
        }

        inline void setup_lut_default(float* query, uint8_t* lut_ptr, float& scale, float& bias) const {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::min();
            mem_index_type buf_size, offset1, offset2, offset3;
            // first iteration to calculate raw distance and max,min values for quantized lut
            buf_size = (mem_index_type) num_local_codebooks * num_of_local_centroids;
            std::vector<float> raw_dist(buf_size, 0);
            std::vector<float> qs(local_dimension);
            for (index_type m = 0; m < num_local_codebooks; m++) {
                offset1 = (mem_index_type) m * num_of_local_centroids * local_dimension;
                offset2 = (mem_index_type) m * num_of_local_centroids;
                offset3 = (mem_index_type) m * local_dimension;
                for (index_type d = 0; d < local_dimension; d++) {
                    qs[d] = query[offset3 + d] - global_centroid[offset3 + d];
                }
                for (index_type k = 0; k < num_of_local_centroids; k++) {
                    float tmp_v = 0;
                    offset3 = (mem_index_type) k * local_dimension;
                    for (index_type d = 0; d < local_dimension; d++) {
                        float v = (qs[d] - local_codebooks[offset1 + offset3 + d]);
                        tmp_v += (v * v);
                    }
                    raw_dist[offset2 + k] = tmp_v;
                    max = std::max(max, tmp_v);
                    min = std::min(min, tmp_v);
                }
            }

            bias = min;
            scale = (max - min) / 255.0;
            // second iteration to calculate quantized distnace and put it into lut
            for (index_type m = 0; m < num_local_codebooks; m++) {
                offset2 = (mem_index_type) m * num_of_local_centroids;
                for (index_type k = 0; k < num_of_local_centroids; k++) {
                    lut_ptr[offset2 + k] = std::round((raw_dist[offset2 + k] - bias) / scale);
                }
            }
        }

        inline void encode(float* query, uint8_t* codes) {
            mem_index_type offset1, offset2, offset3;
            for (index_type m = 0; m < num_local_codebooks; m++) {
                std::vector<float> results;
                offset1 = (mem_index_type) m * num_of_local_centroids * local_dimension;
                offset2 = (mem_index_type) m * local_dimension;
                for (index_type k = 0; k < num_of_local_centroids; k++) {
                    float v = 0;
                    offset3 = (mem_index_type) k * local_dimension;
                    for (index_type d = 0; d < local_dimension; d++) {
                        float tmp_v = original_local_codebooks[offset1 + offset3 + d]
                            - (query[offset2 + d] - global_centroid[offset2 + d]);
                        v += (tmp_v * tmp_v);
                    }
                    results.push_back(v);
                }
                std::vector<float>::iterator argmin_result = std::min_element(results.begin(), results.end());
                codes[m] = std::distance(results.begin(), argmin_result);
            }
        }

        inline void compute_centroids(
            pecos::drm_t& X,
            index_type dsub,
            index_type ksub,
            index_type *assign,
            float *centroids,
            int threads=1
        ) {
            // zero initialization for later do_axpy
            mem_index_type buf_size = (mem_index_type) ksub * dsub;
            memset(centroids, 0, buf_size * sizeof(*centroids));
            std::vector<float> centroids_size(ksub);
            #pragma omp parallel num_threads(threads)
            {
                // each thread takes care of [c_l, c_r)
                int rank = omp_get_thread_num();
                size_t c_l = ((mem_index_type) ksub * rank) / threads;
                size_t c_r = ((mem_index_type) ksub * (rank + 1)) / threads;
                for (index_type i = 0; i < X.rows; i++) {
                    auto ci = assign[i];
                    if (ci >= c_l && ci < c_r) {
                        float* y = centroids + (mem_index_type) ci * dsub;
                        const auto& xi = X.get_row(i);
                        pecos::do_axpy(1.0, xi.val, y, dsub);
                        centroids_size[ci] += 1;
                    }
                }
                // normalize center vector
                for (size_t ci = c_l; ci < c_r; ci++) {
                    float* y = centroids + (mem_index_type) ci * dsub;
                    pecos::do_scale(1.0 / centroids_size[ci], y, dsub);
                }
            }
        }

        inline void train(
            const pecos::drm_t& X_trn,
            index_type num_local_codebooks,
            index_type sub_sample_points=0,
            int seed=0,
            size_t max_iter=10,
            int threads=32
        ) {
            mem_index_type buf_size; // for allocating memory of vectors
            mem_index_type offset;   // for offsetting pointer of vectors
            index_type n_data = X_trn.rows;
            index_type global_dimension = X_trn.cols;
            if (global_dimension % num_local_codebooks != 0) {
                throw std::runtime_error("Original dimension must be divided by subspace dimension");
            }
            this->num_local_codebooks = num_local_codebooks;
            this->local_dimension = global_dimension / num_local_codebooks;
            if (sub_sample_points == 0) {
                sub_sample_points = n_data;
            }

            buf_size = (mem_index_type) num_local_codebooks * num_of_local_centroids * local_dimension;
            original_local_codebooks.resize(buf_size, 0);
            global_centroid.resize(global_dimension, 0);

            buf_size = (mem_index_type) sub_sample_points * local_dimension;
            std::vector<float> xslice(buf_size);
            for (index_type m = 0; m < num_local_codebooks; m++) {
                std::vector<index_type> indices(n_data, 0);
                std::iota(indices.data(), indices.data() + n_data, 0);
                std::random_shuffle(indices.data(), indices.data() + n_data);
                for (index_type i = 0; i < sub_sample_points; i++) {
                    offset = (mem_index_type) indices[i] * global_dimension;
                    std::memcpy(
                        xslice.data() + (mem_index_type) i * local_dimension,
                        X_trn.val + offset  + (mem_index_type) m * local_dimension,
                        local_dimension * sizeof(float)
                    );
                }
                pecos::drm_t Xsub;
                Xsub.rows = sub_sample_points;
                Xsub.cols = local_dimension;
                Xsub.val = xslice.data();

                // fit HLT or flat-Kmeans for each sub-space
                std::vector<index_type> assignments(sub_sample_points);
                index_type hlt_depth = (index_type) std::log2(num_of_local_centroids);
                float max_sample_rate = 1.0;
                float min_sample_rate = 1.0;
                float warmup_ratio = 1.0;
                pecos::clustering::Tree hlt(hlt_depth);
                pecos::clustering::ClusteringParam clustering_param(0, seed, max_iter, threads, max_sample_rate, min_sample_rate, warmup_ratio);
                hlt.run_clustering<pecos::drm_t, index_type>(
                    Xsub,
                    &clustering_param,
                    assignments.data()
                );

                offset = (mem_index_type) m * num_of_local_centroids * local_dimension;
                compute_centroids(
                    Xsub,
                    local_dimension,
                    num_of_local_centroids,
                    assignments.data(),
                    &original_local_codebooks[offset],
                    threads
                );
            }
        }
    };

}  // end of namespace ann
}  // end of namespace pecos
