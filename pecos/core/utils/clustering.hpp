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

#ifndef __CLUSTERING_H__
#define  __CLUSTERING_H__

#include <algorithm>
#include <vector>

#include "matrix.hpp"
#include "random.hpp"
#include "parallel.hpp"

namespace pecos {


namespace clustering {

enum {
    KMEANS=0,
    SKMEANS=5,
}; /* partition_algo */

extern "C" {
    struct ClusteringParam {
        size_t partition_algo;
        int seed;
        size_t kmeans_max_iter;
        int threads;
        float max_sample_rate;
        float min_sample_rate;
        float warmup_ratio;

        ClusteringParam(
            size_t partition_algo,
            int seed,
            size_t kmeans_max_iter,
            int threads,
            float max_sample_rate,
            float min_sample_rate,
            float warmup_ratio
        ): partition_algo(partition_algo),
        seed(seed),
        kmeans_max_iter(kmeans_max_iter),
        threads(threads),
        max_sample_rate(max_sample_rate),
        min_sample_rate(min_sample_rate),
        warmup_ratio(warmup_ratio) {
            if(min_sample_rate <= 0 || min_sample_rate > 1) {
                throw std::invalid_argument("expect 0 < min_sample_rate <= 1.0");
            }
            if(max_sample_rate <= 0 || max_sample_rate > 1) {
                throw std::invalid_argument("expect 0 < max_sample_rate <= 1.0");
            }
            if(min_sample_rate > max_sample_rate) {
                throw std::invalid_argument("expect min_sample_rate <= max_sample_rate");
            }
            if(warmup_ratio < 0 || warmup_ratio > 1) {
                throw std::invalid_argument("expect 0 <= warmup_ratio <= 1.0");
            }
        }
    };
} // end of extern C

struct Node {
    size_t start;
    size_t end;

    Node(size_t start=0, size_t end=0): start(start), end(end) {}

    void set(size_t start, size_t end) {
        this->start = start;
        this->end = end;
    }

    size_t size() const { return end - start; }
};


/*
 * Each node is a cluster of elements
 * #leaf nodes = 2^{depth}
 * #internal nodes = 2^{depth} (where we have a dummy node with node Id = 0)
 * #nodes = 2^{depth + 1}
 */
struct Tree {

    template<class Cont, class Wrap>
    struct center_buffer_t {
        typedef Cont container_t;
        typedef Wrap wrapper_t;

        std::vector<Cont> center1; // need to be duplicated to handle parallel clustering
        std::vector<Cont> center2; // need to be duplicated to handle parallel clustering
        std::vector<Cont> center_tmp_thread; // Temporary working spaces for function update_center, will be cleared

        void resize(int threads, size_t feat_num) {
            center1.resize(threads, Cont(feat_num));
            center2.resize(threads, Cont(feat_num));
            center_tmp_thread.resize(threads, Cont(feat_num));
        }

        void deep_clear() {
            center1.clear();
            center1.shrink_to_fit();
            center2.clear();
            center2.shrink_to_fit();
            center_tmp_thread.clear();
            center_tmp_thread.shrink_to_fit();
        }
    };

    typedef random_number_generator<> rng_t;
    typedef dense_vec_t<float32_t> dvec_wrapper_t;
    typedef sdvec_t<uint32_t, float32_t> sdvec_wrapper_t;
    typedef center_buffer_t<f32_dvec_t, dense_vec_t<float32_t>> dvec_center_buffer_t;
    typedef center_buffer_t<typename sdvec_wrapper_t::container_t, sdvec_wrapper_t> sdvec_center_buffer_t;

    size_t depth;     // # leaf nodes = 2^depth
    std::vector<Node> nodes;

    // used for balanced 2-means
    u64_dvec_t elements;
    u64_dvec_t previous_elements;
    u32_dvec_t seed_for_nodes; // random seeds used for each node
    f32_dvec_t scores;

    dvec_center_buffer_t dvec_center_buffer;
    sdvec_center_buffer_t sdvec_center_buffer;
    bool use_sdvec;

    Tree(size_t depth=0) { this->reset_depth(depth); }

    void reset_depth(size_t depth) {
        this->depth = depth;
        nodes.resize(1 << (depth + 1));
        seed_for_nodes.resize(nodes.size());
    }

    struct ClusteringSampler {
        // scheduler for sampling
        size_t depth;
        ClusteringParam* param_ptr;
        size_t warmup_layers;

        ClusteringSampler(size_t depth, ClusteringParam* param_ptr): depth(depth), param_ptr(param_ptr) {
            warmup_layers = size_t(depth * param_ptr->warmup_ratio);
        }

        float get_sample_rate(size_t layer) const {
            // If input `layer` < `warmup_layers`, return `min_sample_rate`.
            // Otherwise, linearly increase the current sample rate from `min_sample_rate` to `max_sample_rate` until the last layer.
            if(layer < warmup_layers) {
                return param_ptr->min_sample_rate;
            }
            return param_ptr->min_sample_rate + (param_ptr->max_sample_rate - param_ptr->min_sample_rate) * (layer + 1 - warmup_layers) / (depth - warmup_layers);
        }
    };

    struct comparator_by_value_t {
        const float32_t *pred_val;
        bool increasing;
        comparator_by_value_t(const float32_t *val, bool increasing=true):
            pred_val(val), increasing(increasing) {}
        bool operator()(const size_t i, const size_t j) const {
            if(increasing) {
                return (pred_val[i] < pred_val[j]) || (pred_val[i] == pred_val[j] && i < j);
            } else {
                return (pred_val[i] > pred_val[j]) || (pred_val[i] == pred_val[j] && i < j);
            }
        }
    };


    Node& root_of(size_t nid) { return nodes[nid]; }
    Node& left_of(size_t nid) { return nodes[nid << 1]; }
    Node& right_of(size_t nid) { return nodes[(nid << 1) + 1]; }

    void partition_elements(Node& root, Node& left, Node& right) {
        size_t middle = (root.start + root.end) >> 1;
        left.set(root.start, middle);
        right.set(middle, root.end);
    }

    void sample_elements(Node& root, rng_t& rng, float cur_sample_rate) {
        rng.shuffle(elements.begin() + root.start, elements.begin() + root.end);
        size_t n_sp_elements = size_t(cur_sample_rate * root.size());
        root.set(root.start, root.start + n_sp_elements);
    }

    // Sort elements by scores on node and return if this function changes the assignment
    bool sort_elements_by_scores_on_node(const Node& root, int threads=1, bool increasing=true) {
        auto prev_start_it = previous_elements.begin() + root.start;
        auto start_it = elements.begin() + root.start;
        auto middle_it = elements.begin() + ((root.start + root.end) >> 1);
        auto end_it = elements.begin() + root.end;
        std::copy(start_it, middle_it, prev_start_it);
        parallel_sort(start_it, end_it, comparator_by_value_t(scores.data(), increasing), threads);
        parallel_sort(start_it, middle_it, std::less<size_t>(), threads);
        parallel_sort(middle_it, end_it, std::less<size_t>(), threads);
        return !std::equal(start_it, middle_it, prev_start_it);
    }

    // X = [x_1, ..., x_L]^T
    // c_1 = e_1^T X / |e_1|_0, where \be_1 is the indicator for first half elements
    // c_2 = e_2^T X / |e_2|_0, where \be_2 is the indicator for second half elements
    // e = e_2/|e_2|_0 - e_1/|e_1|_0
    // c = c_2 - c_1 = X^T e
    // score(i) = <c, x_i>
    // works for both cosine similarity if feat_mat is with unit-length rows
    //                euclidean similarity

    // Loop through node's elements and update current center
    template<typename MAT, typename CenterWrapper, typename CenterBuffer>
    void update_center(const MAT& feat_mat, Node& cur_node, CenterWrapper& cur_center, float32_t alpha, int threads, CenterBuffer& center_buffer) {
        if(threads == 1) {
           for(size_t i = cur_node.start; i < cur_node.end; i++) {
                size_t eid = elements[i];
                const auto& feat = feat_mat.get_row(eid);
                do_axpy(alpha, feat, cur_center);
            }
        } else {
            #pragma omp parallel num_threads(threads)
            {
                int thread_id = omp_get_thread_num();
                typename CenterBuffer::wrapper_t cur_center_tmp_thread(center_buffer.center_tmp_thread[thread_id]);
                cur_center_tmp_thread.fill_zeros();
                // use static for reproducibility under multi-trials with same seed.
                #pragma omp for schedule(static)
                for(size_t i = cur_node.start; i < cur_node.end; i++) {
                    size_t eid = elements[i];
                    const auto& feat = feat_mat.get_row(eid);
                    do_axpy(alpha, feat, cur_center_tmp_thread);
                }
            }

            if(std::is_same<typename CenterBuffer::wrapper_t, dvec_wrapper_t>::value) {
                // global parallel reduction for dense vector center
#pragma omp parallel for schedule(static)
                for(size_t i = 0; i < cur_center.len; ++i) {
                    for(int thread_id = 0; thread_id < threads; thread_id++) {
                        typename CenterBuffer::wrapper_t cur_center_tmp_thread(center_buffer.center_tmp_thread[thread_id]);
                        cur_center[i] += cur_center_tmp_thread[i];
                    }
                }
            } else {
                for(int thread_id = 0; thread_id < threads; thread_id++) {
                    typename CenterBuffer::wrapper_t cur_center_tmp_thread(center_buffer.center_tmp_thread[thread_id]);
                    do_axpy(1.0, cur_center_tmp_thread, cur_center);
                }
            }
        }
    }

    template<typename MAT, typename CenterWrapper>
    bool do_assignment(MAT* feat_mat_ptr, Node& root, CenterWrapper* center_ptr, int threads) {
        u64_dvec_t *elements_ptr = &elements;
        auto *scores_ptr = &scores;
        if(threads == 1) {
            for(size_t i = root.start; i < root.end; i++) {
                size_t eid = elements_ptr->at(i);
                const auto& feat = feat_mat_ptr->get_row(eid);
                scores_ptr->at(eid) = do_dot_product(*center_ptr, feat);
            }
        } else {
#pragma omp parallel for shared(elements_ptr, scores_ptr, center_ptr, feat_mat_ptr)
            for(size_t i = root.start; i < root.end; i++) {
                size_t eid = elements_ptr->at(i);
                const auto& feat = feat_mat_ptr->get_row(eid);
                scores_ptr->at(eid) = do_dot_product(*center_ptr, feat);
            }
        }
        bool assignment_changed = sort_elements_by_scores_on_node(root, threads);
        return assignment_changed;
    }

    template<typename MAT, typename CenterBuffer>
    void partition_kmeans(size_t nid, size_t depth, const MAT& feat_mat, CenterBuffer& center_buffer, rng_t& rng, size_t max_iter=10, int threads=1, int thread_id=0, float cur_sample_rate=1.0) {
        // copy nodes rather than reference for sampling
        Node root = root_of(nid);
        Node left = left_of(nid);
        Node right = right_of(nid);

        // modify nodes' start and end based on cur_sample_rate 
        if(cur_sample_rate < 1.0) {
            sample_elements(root, rng, cur_sample_rate);
        }
        partition_elements(root, left, right);

        typename CenterBuffer::wrapper_t cur_center(center_buffer.center1[thread_id]);

        // perform the clustering and sorting
        for(size_t iter = 0; iter < max_iter; iter++) {
            // construct cur_center (for right child)
            cur_center.fill_zeros();
            if(iter == 0) {
                auto right_idx = rng.randint(0, root.size() - 1);
                auto left_idx = (right_idx + rng.randint(1, root.size() - 1)) % root.size();
                right_idx += root.start;
                left_idx  += root.start;

                const auto& feat_right = feat_mat.get_row(elements[right_idx]);
                const auto& feat_left = feat_mat.get_row(elements[left_idx]);
                do_axpy(1.0, feat_right, cur_center);
                do_axpy(-1.0, feat_left, cur_center);

            } else {
                float32_t alpha = 0;
                alpha = +1.0 / right.size();
                update_center(feat_mat, right, cur_center, alpha, threads, center_buffer);

                alpha = -1.0 / left.size();
                update_center(feat_mat, left, cur_center, alpha, threads, center_buffer);
            }
            bool assignment_changed = do_assignment(&feat_mat, root, &cur_center, threads);
            if(!assignment_changed) {
                break;
            }
        }

        // set indices for reference nodes
        partition_elements(root_of(nid), left_of(nid), right_of(nid));

        // perform inference on all elements
        if(cur_sample_rate < 1.0) {
            do_assignment(&feat_mat, root_of(nid), &cur_center, threads);
        }
    }

    template<typename MAT, typename CenterBuffer>
    void partition_skmeans(size_t nid, size_t depth, const MAT& feat_mat, CenterBuffer& center_buffer, rng_t& rng, size_t max_iter=10, int threads=1, int thread_id=0, float cur_sample_rate=1.0) {
        // copy nodes rather than reference for sampling
        Node root = root_of(nid);
        Node left = left_of(nid);
        Node right = right_of(nid);

        // modify nodes' start and end based on cur_sample_rate 
        if(cur_sample_rate < 1.0) {
            sample_elements(root, rng, cur_sample_rate);
        }
        partition_elements(root, left, right);

        typename CenterBuffer::wrapper_t cur_center1(center_buffer.center1[thread_id]);
        typename CenterBuffer::wrapper_t cur_center2(center_buffer.center2[thread_id]);

        // perform the clustering and sorting
        for(size_t iter = 0; iter < max_iter; iter++) {
            float32_t one = 1.0;
            // construct center1 (for right child)
            cur_center1.fill_zeros();
            cur_center2.fill_zeros();
            if(iter == 0) {
                auto right_idx = rng.randint(0, root.size() - 1);
                auto left_idx = (right_idx + rng.randint(1, root.size() - 1)) % root.size();
                right_idx += root.start;
                left_idx  += root.start;

                const auto& feat_right = feat_mat.get_row(elements[right_idx]);
                const auto& feat_left = feat_mat.get_row(elements[left_idx]);
                do_axpy(1.0, feat_right, cur_center1);
                do_axpy(1.0, feat_left, cur_center2);
                do_axpy(-1.0, cur_center2, cur_center1);
            } else {
                update_center(feat_mat, right, cur_center1, one, threads, center_buffer);
                float32_t alpha = do_dot_product(cur_center1, cur_center1);
                if(alpha > 0) {
                    do_scale(1.0 / sqrt(alpha), cur_center1);
                }

                update_center(feat_mat, left, cur_center2, one, threads, center_buffer);
                alpha = do_dot_product(cur_center2, cur_center2);
                if(alpha > 0) {
                    do_scale(1.0 / sqrt(alpha), cur_center2);
                }

                do_axpy(-1.0, cur_center2, cur_center1);
            }
            bool assignment_changed = do_assignment(&feat_mat, root, &cur_center1, threads);
            if(!assignment_changed) {
                break;
            }
        }

        // set indices for reference nodes
        partition_elements(root_of(nid), left_of(nid), right_of(nid));

        // perform inference on all elements
        if(cur_sample_rate < 1.0) {
            do_assignment(&feat_mat, root_of(nid), &cur_center1, threads);
        }
    }

    template<typename MAT, typename IND=unsigned>
    void run_clustering(const MAT& feat_mat, ClusteringParam* param_ptr=NULL, IND *label_codes=NULL) {
        size_t nr_elements = feat_mat.rows;
        elements.resize(nr_elements);
        previous_elements.resize(nr_elements);
        for(size_t i = 0; i < nr_elements; i++) {
            elements[i] = i;
        }
        rng_t rng(param_ptr->seed);
        for(size_t nid = 0; nid < nodes.size(); nid++) {
            seed_for_nodes[nid] = rng.randint<unsigned>();
        }

        use_sdvec = false; // use dense vector by default

        // no need to allocate sdvec and dvec at the same time!
        int threads = set_threads(param_ptr->threads);

        scores.resize(feat_mat.rows, 0);
        nodes[0].set(0, nr_elements);
        nodes[1].set(0, nr_elements);

        ClusteringSampler sample_scheduler(depth, param_ptr);

        // let's do it layer by layer so we can parallelize it
        for(size_t d = 0; d < depth; d++) {
            size_t layer_start = 1U << d;
            size_t layer_end = 1U << (d + 1);
            float cur_sample_rate = sample_scheduler.get_sample_rate(d);

            // switch from dense vec to sdvec if matrix worst max density is below threshold
            if(double(feat_mat.get_nnz()) / (double(layer_end - layer_start) * double(feat_mat.cols))  < 1) {
                use_sdvec = true;
            }

            if(use_sdvec) {
                dvec_center_buffer.deep_clear();
                sdvec_center_buffer.resize(threads, feat_mat.cols);
            } else {
                sdvec_center_buffer.deep_clear();
                dvec_center_buffer.resize(threads, feat_mat.cols);
            }

            if((layer_end - layer_start) >= (size_t) threads) {
#pragma omp parallel for schedule(dynamic)
                for(size_t nid = layer_start; nid < layer_end; nid++) {
                    rng_t rng(seed_for_nodes[nid]);
                    int local_threads = 1;
                    int thread_id = omp_get_thread_num();

                    if(use_sdvec) {
                        if(param_ptr->partition_algo == KMEANS) {
                            partition_kmeans(nid, d, feat_mat, sdvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        } else if(param_ptr->partition_algo == SKMEANS) {
                            partition_skmeans(nid, d, feat_mat, sdvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        }
                    } else {
                        if(param_ptr->partition_algo == KMEANS) {
                            partition_kmeans(nid, d, feat_mat, dvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        } else if(param_ptr->partition_algo == SKMEANS) {
                            partition_skmeans(nid, d, feat_mat, dvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        }
                    }

                }
            } else {
                for(size_t nid = layer_start; nid < layer_end; nid++) {
                    rng_t rng(seed_for_nodes[nid]);
                    int local_threads = threads;
                    int thread_id = 0;
                    if(use_sdvec) {
                        if(param_ptr->partition_algo == KMEANS) {
                            partition_kmeans(nid, d, feat_mat, sdvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        } else if(param_ptr->partition_algo == SKMEANS) {
                            partition_skmeans(nid, d, feat_mat, sdvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        }
                    } else {
                        if(param_ptr->partition_algo == KMEANS) {
                            partition_kmeans(nid, d, feat_mat, dvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        } else if(param_ptr->partition_algo == SKMEANS) {
                            partition_skmeans(nid, d, feat_mat, dvec_center_buffer, rng, param_ptr->kmeans_max_iter, local_threads, thread_id, cur_sample_rate);
                        }
                    }

                }
            }
        }

        if(label_codes != NULL) {
            size_t leaf_start = 1U << depth;
            size_t leaf_end = 1U << (depth + 1);
            for(size_t nid = leaf_start; nid < leaf_end; nid++) {
                for(size_t idx = nodes[nid].start; idx < nodes[nid].end; idx++) {
                    label_codes[elements[idx]] = nid - leaf_start;
                }
            }
        }

        dvec_center_buffer.deep_clear();
        sdvec_center_buffer.deep_clear();
    }

    void output() {
        size_t nr_internal_nodes = nodes.size() >> 1;
        for(size_t nid = nr_internal_nodes; nid < nodes.size(); nid++) {
            const Node& node = nodes[nid];
            printf("node(%ld): ", nid);
            for(size_t idx = node.start; idx < node.end; idx++) {
                printf(" %ld", elements[idx]);
            }
            puts("");
        }
    }
};

} // end of namespace clustering
} // end of namespace pecos

#endif // end of __CLUSTERING_H__
