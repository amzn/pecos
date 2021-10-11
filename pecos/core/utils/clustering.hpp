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

namespace pecos {


namespace clustering {

enum {
    KMEANS=0,
    SKMEANS=5,
}; /* partition_algo */

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
    typedef random_number_generator<> rng_t;
    typedef dense_vec_t<float32_t> dvec_wrapper_t;

    size_t depth;     // # leaf nodes = 2^depth
    std::vector<Node> nodes;

    // used for balanced 2-means
    u64_dvec_t elements;
    u64_dvec_t previous_elements;
    std::vector<f32_dvec_t> center1; // need to be duplicated to handle parallel clustering
    std::vector<f32_dvec_t> center2; // for spherical kmeans
    u32_dvec_t seed_for_nodes; // random seeds used for each node
    f32_dvec_t scores;

    // Temporary working spaces for function update_center, will be cleared after clustering to release space
    std::vector<f32_dvec_t> center_tmp_thread; // thread-private working array for parallel updating center

    Tree(size_t depth=0) { this->reset_depth(depth); }

    void reset_depth(size_t depth) {
        this->depth = depth;
        nodes.resize(1 << (depth + 1));
        seed_for_nodes.resize(nodes.size());
    }

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

    // Sort elements by scores on node and return if this function changes the assignment
    bool sort_elements_by_scores_on_node(const Node& root, bool increasing=true) {
        auto prev_start_it = previous_elements.begin() + root.start;
        auto start_it = elements.begin() + root.start;
        auto middle_it = elements.begin() + ((root.start + root.end) >> 1);
        auto end_it = elements.begin() + root.end;
        std::copy(start_it, middle_it, prev_start_it);
        std::sort(start_it, end_it, comparator_by_value_t(scores.data(), increasing));
        std::sort(start_it, middle_it);
        std::sort(middle_it, end_it);
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
    template<typename MAT>
    void update_center(const MAT& feat_mat, Node& cur_node, dvec_wrapper_t& cur_center, float32_t alpha, int threads=1) {
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
                std::fill(center_tmp_thread[thread_id].begin(), center_tmp_thread[thread_id].end(), 0);
                dvec_wrapper_t cur_center_tmp_thread(center_tmp_thread[thread_id]);
                // use static for reproducibility under multi-trials with same seed.
                #pragma omp for schedule(static)
                for(size_t i = cur_node.start; i < cur_node.end; i++) {
                    size_t eid = elements[i];
                    const auto& feat = feat_mat.get_row(eid);
                    do_axpy(alpha, feat, cur_center_tmp_thread);
                }
            }

            // global parallel reduction
            #pragma omp parallel for schedule(static)
            for(size_t i=0; i<cur_center.len; ++i) {
                for(int thread_id = 0; thread_id < threads; thread_id++) {
                        cur_center[i] += center_tmp_thread[thread_id][i];
                }
            }
        }
    }

    template<typename MAT>
    void partition_kmeans(size_t nid, size_t depth, const MAT& feat_mat, rng_t& rng, size_t max_iter=10, int threads=1, int thread_id=0) {
        Node& root = root_of(nid);
        Node& left = left_of(nid);
        Node& right = right_of(nid);
        partition_elements(root, left, right);

        dvec_wrapper_t cur_center(center1[thread_id]);

        // perform the clustering and sorting
        for(size_t iter = 0; iter < max_iter; iter++) {
            // construct cur_center (for right child)
            std::fill(center1[thread_id].begin(), center1[thread_id].end(), 0);
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
                update_center(feat_mat, right, cur_center, alpha, threads);

                alpha = -1.0 / left.size();
                update_center(feat_mat, left, cur_center, alpha, threads);
            }
            u64_dvec_t *elements_ptr = &elements;
            auto *scores_ptr = &scores;
            auto *center_ptr = &cur_center;
            const MAT* feat_mat_ptr = &feat_mat;
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
            bool assignment_changed = sort_elements_by_scores_on_node(root);
            if(!assignment_changed) {
                break;
            }
        }
    }

    template<typename MAT>
    void partition_skmeans(size_t nid, size_t depth, const MAT& feat_mat, rng_t& rng, size_t max_iter=10, int threads=1, int thread_id=0) {
        Node& root = root_of(nid);
        Node& left = left_of(nid);
        Node& right = right_of(nid);
        partition_elements(root, left, right);

        dvec_wrapper_t cur_center1(center1[thread_id]);
        dvec_wrapper_t cur_center2(center2[thread_id]);

        // perform the clustering and sorting
        for(size_t iter = 0; iter < max_iter; iter++) {
            float32_t one = 1.0;
            // construct center1 (for right child)
            std::fill(center1[thread_id].begin(), center1[thread_id].end(), 0);
            std::fill(center2[thread_id].begin(), center2[thread_id].end(), 0);
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
                update_center(feat_mat, right, cur_center1, one, threads);
                float32_t alpha = do_dot_product(cur_center1, cur_center1);
                if(alpha > 0) {
                    do_scale(1.0 / sqrt(alpha), cur_center1);
                }

                update_center(feat_mat, left, cur_center2, one, threads);
                alpha = do_dot_product(cur_center2, cur_center2);
                if(alpha > 0) {
                    do_scale(1.0 / sqrt(alpha), cur_center2);
                }

                do_axpy(-1.0, cur_center2, cur_center1);
            }


            u64_dvec_t *elements_ptr = &elements;
            auto *scores_ptr = &scores;
            auto *center_ptr = &cur_center1;
            const MAT* feat_mat_ptr = &feat_mat;
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
            bool assignment_changed = sort_elements_by_scores_on_node(root);
            if(!assignment_changed) {
                break;
            }
        }
    }

    template<typename MAT, typename IND=unsigned>
    void run_clustering(const MAT& feat_mat, int partition_algo, int seed=0, IND *label_codes=NULL, size_t max_iter=10, int threads=1) {
        size_t nr_elements = feat_mat.rows;
        elements.resize(nr_elements);
        previous_elements.resize(nr_elements);
        for(size_t i = 0; i < nr_elements; i++) {
            elements[i] = i;
        }
        rng_t rng(seed);
        for(size_t nid = 0; nid < nodes.size(); nid++) {
            seed_for_nodes[nid] = rng.randint<unsigned>();
        }

        threads = set_threads(threads);
        center1.resize(threads, f32_dvec_t(feat_mat.cols, 0));
        center2.resize(threads, f32_dvec_t(feat_mat.cols, 0));
        scores.resize(feat_mat.rows, 0);
        nodes[0].set(0, nr_elements);
        nodes[1].set(0, nr_elements);

        // Allocate tmp arrays for parallel update center
        center_tmp_thread.resize(threads, f32_dvec_t(feat_mat.cols, 0));


        // let's do it layer by layer so we can parallelize it
        for(size_t d = 0; d < depth; d++) {
            size_t layer_start = 1U << d;
            size_t layer_end = 1U << (d + 1);
            if((layer_end - layer_start) >= (size_t) threads) {
#pragma omp parallel for schedule(dynamic)
                for(size_t nid = layer_start; nid < layer_end; nid++) {
                    rng_t rng(seed_for_nodes[nid]);
                    int local_threads = 1;
                    int thread_id = omp_get_thread_num();
                    if(partition_algo == KMEANS) {
                        partition_kmeans(nid, d, feat_mat, rng, max_iter, local_threads, thread_id);
                    } else if(partition_algo == SKMEANS) {
                        partition_skmeans(nid, d, feat_mat, rng, max_iter, local_threads, thread_id);
                    }
                }
            } else {
                for(size_t nid = layer_start; nid < layer_end; nid++) {
                    rng_t rng(seed_for_nodes[nid]);
                    int local_threads = threads;
                    int thread_id = 0;
                    if(partition_algo == KMEANS) {
                        partition_kmeans(nid, d, feat_mat, rng, max_iter, local_threads, thread_id);
                    } else if(partition_algo == SKMEANS) {
                        partition_skmeans(nid, d, feat_mat, rng, max_iter, local_threads, thread_id);
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

        // clear tmp arrays
        center_tmp_thread.clear();
        center_tmp_thread.shrink_to_fit();
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
