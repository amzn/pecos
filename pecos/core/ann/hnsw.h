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

#include <cmath>
#include <cstdlib>
#include <queue>
#include <random>
#include <vector>

#include "utils/matrix.hpp"

namespace pecos {

namespace ann {
    // assuming the num_node < 2^31-1 (~4.2 billion)
    typedef uint32_t index_type;

    // for calling InnerProductBLAS/L2SqrBLAS
    // arg1 is the ptr for vect1
    // arg2 is the ptr for vect2
    // arg3 is the size of feat_dim
    template<typename MTYPE>
    using DistFn = MTYPE(*)(const void *, const void *, const void *);

    static float InnerProductBLAS(const void *x_ptr, const void *y_ptr, const void *feat_dim) {
        return 1.0 - do_dot_product((float *) x_ptr, (float *) y_ptr, *((size_t *) feat_dim));
    }

    static float L2SqrBLAS(const void *x_ptr, const void *y_ptr, const void *feat_dim) {
        float *x = (float *) x_ptr;
        float *y = (float *) y_ptr;
        size_t d = *((size_t *) feat_dim);
        float x_norm2 = do_dot_product(x, x, d);
        float y_norm2 = do_dot_product(y, y, d);
        return x_norm2 + y_norm2 - 2.0 * do_dot_product(x, y, d);
    }

    template<typename MTYPE>
    class DistanceBase {
        public:

            virtual DistFn<MTYPE> get_dist_fn() = 0;

            virtual void *get_dist_feat_dim() = 0;

            virtual ~DistanceBase() {}
    };

    class InnerProductSpace : public DistanceBase<float> {
        DistFn<float> dist_fn;
        size_t feat_dim;
    public:
        InnerProductSpace(size_t feat_dim) {
            this->dist_fn = InnerProductBLAS;
            this->feat_dim = feat_dim;
        }

        DistFn<float> get_dist_fn() {
            return dist_fn;
        }

        void *get_dist_feat_dim() {
            return &feat_dim;
        }

    	~InnerProductSpace() {}
    };

    class L2Space : public DistanceBase<float> {
        DistFn<float> dist_fn;
        size_t feat_dim;
    public:
        L2Space(size_t feat_dim) {
            this->dist_fn = L2SqrBLAS;
            this->feat_dim = feat_dim;
        }

        DistFn<float> get_dist_fn() {
            return dist_fn;
        }

        void *get_dist_feat_dim() {
            return &feat_dim;
        }

        ~L2Space() {}
    };

    struct GraphBase {
        virtual index_type* get_node_degree_ptr(index_type node_id, index_type dummy_level_id=0) = 0;
        virtual index_type* get_node_neighbor_ptr(index_type node_id, index_type dummy_level_id=0) = 0;
    };

    struct GraphL0 : GraphBase {
        index_type num_node;
        index_type feat_dim;
        index_type max_degree;
        index_type node_mem_size;
        std::vector<char> buffer;

        void resize(index_type num_node, index_type feat_dim, index_type max_degree) {
            this->num_node = num_node;
            this->feat_dim = feat_dim;
            this->max_degree = max_degree;
            this->node_mem_size = feat_dim * sizeof(float) + (1 + max_degree) * sizeof(index_type);
            buffer.resize(num_node * this->node_mem_size);
        }

        inline float* get_node_feat(index_type node_id) {
            return (float *) &buffer.data()[node_id * node_mem_size + (1 + max_degree) * sizeof(index_type)];
        }

        inline index_type* get_node_degree_ptr(index_type node_id, index_type dummy_level_id=0) {
            return (index_type *) &buffer.data()[node_id * node_mem_size];
        }

        inline index_type* get_node_neighbor_ptr(index_type node_id, index_type dummy_level_id=0) {
            return (index_type *) &buffer.data()[node_id * node_mem_size + 1 * sizeof(index_type)];
        }
    };

    struct GraphL1 : GraphBase {
        index_type num_node;
        index_type max_level;
        index_type max_degree;
        index_type node_mem_size;
        index_type level_mem_size;
        std::vector<index_type> buffer;

        void resize(index_type num_node, index_type max_level, index_type max_degree) {
            this->num_node = num_node;
            this->max_level = max_level;
            this->max_degree = max_degree;
            this->level_mem_size = 1 + max_degree;
            this->node_mem_size = max_level * this->level_mem_size;
            buffer.resize(num_node * this->node_mem_size);
        }

        inline index_type* get_node_degree_ptr(index_type node_id, index_type level_id=0) {
            if (level_id == 0) {
                throw std::runtime_error("get_node_degree_ptr can not have level_id == 0!");
            }
            return &buffer[node_id * this->node_mem_size + (level_id - 1) * this->level_mem_size];
        }

        inline index_type* get_node_neighbor_ptr(index_type node_id, index_type level_id=0) {
            return (get_node_degree_ptr(node_id, level_id) + 1);
        }
    };

    // PECOS-HNSW Interface
    template<typename dist_t>
    struct HNSW {
        typedef typename std::pair<dist_t, index_type> pair_t;
        struct CompareByFirst {
            constexpr bool operator()(pair_t const &a, pair_t const &b) const noexcept { return a.first < b.first; }
        };
        typedef typename std::priority_queue<pair_t, std::vector<pair_t>, CompareByFirst> max_heap_t;

        template<class T>
        struct SetOfVistedNodes {
            T init_token, curr_token;
            std::vector<T> buffer;
            SetOfVistedNodes(int num_nodes) :
                init_token(0),
                curr_token(init_token),
                buffer(num_nodes, T(init_token)) { }

            void mark_visited(unsigned node_id) { buffer[node_id] = curr_token; }

            bool is_visited(unsigned node_id) { return buffer[node_id] == curr_token; }

            // need to reset() for every new query search
            // amortized time complexity is O(num_nodes / std::numeric_limits<T>::max())
            void reset() {
                curr_token += 1;
                if(curr_token == init_token) {
                    std::fill(buffer.begin(), buffer.end(), init_token);
                    curr_token = init_token + 1;
                }
            }
        };

        // scalar variables
        index_type num_node;
        index_type feat_dim;
        index_type maxM;  // number of max out degree for layer l=1,...,L
        index_type maxM0; // number of max out degree for layer l=0
        index_type efC; // size of priority queue for construction time
        index_type efS; // size of priority queue for search time
        index_type max_level;
        index_type init_node;

        // distance function
        DistFn<dist_t> dist_fn;
        void *dist_feat_dim;

        // data structures for multi-level graph
        GraphL1 graph_l1;
        GraphL0 graph_l0;
        SetOfVistedNodes<unsigned short int> set_of_visited_nodes;
        std::vector<index_type> node2level_vec;
        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        // constructor
        HNSW(DistanceBase<dist_t> *s) {}

        // constructor
        HNSW(
            DistanceBase<dist_t> *s,
            index_type num_node,
            index_type feat_dim,
            index_type M=16,
            index_type efC=200
        ) : set_of_visited_nodes(num_node) {
            this->num_node = num_node;
            this->feat_dim = feat_dim;
            this->dist_fn = s->get_dist_fn();
            this->dist_feat_dim = s->get_dist_feat_dim();
            this->maxM = M;
            this->maxM0 = 2 * M;
            this->efC = efC;
        }

        // line 4 of Algorithm 1 in HNSW paper
        int get_random_level(double mult_l) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * mult_l;
            return (int) r;
        }

        // Algorithm 4 of HNSW paper
        void get_neighbors_heuristic(max_heap_t &top_candidates, const index_type M) {
            if (top_candidates.size() < M) { return; }

            std::priority_queue<pair_t> queue_closest;
            std::vector<pair_t> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M) {
                    break;
                }
                auto curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (auto second_pair : return_list) {
                    dist_t curdist = dist_fn(
                        graph_l0.get_node_feat(second_pair.second),
                        graph_l0.get_node_feat(curent_pair.second),
                        dist_feat_dim
                    );
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (auto curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        // line 10-17, Algorithm 1 of HNSW paper
        index_type mutually_connect(index_type query_id, max_heap_t &top_candidates, index_type level) {
            index_type Mcurmax = level ? this->maxM : this->maxM0;
            get_neighbors_heuristic(top_candidates, this->maxM);
            if (top_candidates.size() > this->maxM) {
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");
            }

            std::vector<index_type> selected_neighbors;
            selected_neighbors.reserve(this->maxM);
            while (top_candidates.size() > 0) {
                selected_neighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }
            index_type next_closest_entry_point = selected_neighbors.back();

            GraphBase *G;
            if (level == 0) {
                G = &graph_l0;
            } else {
                G = &graph_l1;
            }

            // for node in selected_neighbors, connect query_id to node
            auto degree_ptr = G->get_node_degree_ptr(query_id, level);
            *degree_ptr = selected_neighbors.size();
            auto neighbors = degree_ptr + 1;
            for (index_type idx = 0; idx < selected_neighbors.size(); idx++) {
                if (neighbors[idx]) {
                    throw std::runtime_error("Possible memory corruption");
                }
                if (level > node2level_vec[selected_neighbors[idx]]) {
                    throw std::runtime_error("Trying to make a link on a non-existent level");
                }
                neighbors[idx] = selected_neighbors[idx];
            }

            // for node in selected_neighbors, connect node to query_id
            for (index_type idx = 0; idx < selected_neighbors.size(); idx++) {
                auto degree_ptr = G->get_node_degree_ptr(selected_neighbors[idx], level);
                auto neighbors = degree_ptr + 1;
                auto num_edges = *degree_ptr;
                if (num_edges > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selected_neighbors[idx] == query_id)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > node2level_vec[selected_neighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                if (num_edges < Mcurmax) {
                    neighbors[num_edges] = query_id;
                    *degree_ptr = num_edges + 1;
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = dist_fn(
                        graph_l0.get_node_feat(query_id),
                        graph_l0.get_node_feat(selected_neighbors[idx]),
                        dist_feat_dim
                    );
                    // Heuristic:
                    max_heap_t candidates;
                    candidates.emplace(d_max, query_id);
                    for (index_type j = 0; j < num_edges; j++) {
                        dist_t dist_j = dist_fn(
                            graph_l0.get_node_feat(neighbors[j]),
                            graph_l0.get_node_feat(selected_neighbors[idx]),
                            dist_feat_dim
                        );
                        candidates.emplace(dist_j, neighbors[j]);
                    }
                    get_neighbors_heuristic(candidates, Mcurmax);

                    index_type indx = 0;
                    while (candidates.size() > 0) {
                        neighbors[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    *degree_ptr = indx;
                }
            }

            return next_closest_entry_point;
        }

        // train, Algorithm 1 of HNSW paper (i.e., construct HNSW graph)
        void train(pecos::drm_t &X_trn, index_type max_level_upper_bound) {
            this->num_node = X_trn.rows;
            this->feat_dim = X_trn.cols;
            // this is m_l defined in Sec 4.1 of HNSW paper
            float mult_l = 1.0 / log(1.0 * this->maxM);

            graph_l1.resize(this->num_node, max_level_upper_bound, this->maxM);
            graph_l0.resize(this->num_node, this->feat_dim, this->maxM0);
            node2level_vec.resize(num_node);

            index_type entrypoint_id = 0;
            index_type max_level = 0;
            // add point sequentially
            for (index_type query_id = 0; query_id < num_node; query_id++) {
                bool is_first_node = (query_id == 0);

                // assign query node features to graph_l0
                dist_t *query_feat_ptr = X_trn.get_row(query_id).val;
                std::memcpy(graph_l0.get_node_feat(query_id), query_feat_ptr, sizeof(float) * this->feat_dim);

                // sample the query node's level
                index_type query_level = get_random_level(mult_l);
                if (query_level > max_level_upper_bound) {
                    query_level = max_level_upper_bound;
                }
                node2level_vec[query_id] = query_level;

                index_type curr_node = entrypoint_id;
                if (is_first_node) {
                    entrypoint_id = 0;
                    max_level = query_level;
                } else {
                    // find entrypoint with ef=1 for layer > 1
                    if (query_level < max_level) {
                        dist_t curr_dist = dist_fn(query_feat_ptr, graph_l0.get_node_feat(curr_node), dist_feat_dim);
                        for (auto level = max_level; level > query_level; level--) {
                            bool changed = true;
                            while (changed) {
                                changed = false;
                                auto degree_ptr = graph_l1.get_node_degree_ptr(curr_node, level);
                                auto neighbors = degree_ptr + 1;
                                auto num_edges = *degree_ptr;
                                for (index_type j = 0; j < num_edges; j++) {
                                    index_type next_node = neighbors[j];
                                    dist_t next_dist = dist_fn(query_feat_ptr, graph_l0.get_node_feat(next_node), dist_feat_dim);
                                    if (next_dist < curr_dist) {
                                        curr_dist = next_dist;
                                        curr_node = next_node;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    for (auto level=std::min(query_level, max_level); ; level--) {
                        auto top_candidates = search_layer(query_feat_ptr, curr_node, this->efC, level);
                        curr_node = mutually_connect(query_id, top_candidates, level);
                        if (level == 0) { break; }
                    }
                }
                if (query_level > max_level) {
                    entrypoint_id = query_id;
                    max_level = query_level;
                }
            }
            this->max_level = max_level;
            this->init_node = entrypoint_id;
        }

        // Algorithm 2 of HNSW paper
        max_heap_t search_layer(const void *query, index_type init_node, index_type efS, index_type level) {
            max_heap_t topk_queue;
            max_heap_t cand_queue;
            set_of_visited_nodes.reset();

            dist_t topk_ub_dist = dist_fn(query, graph_l0.get_node_feat(init_node), dist_feat_dim);
            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(-topk_ub_dist, init_node);
            set_of_visited_nodes.mark_visited(init_node);

            GraphBase *G;
            if (level == 0) {
                G = &graph_l0;
            } else {
                G = &graph_l1;
            }

            // Best First Search loop
            while (!cand_queue.empty()) {
                pair_t cand_pair = cand_queue.top();
                if (-cand_pair.first > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.second;
                auto degree_ptr = G->get_node_degree_ptr(cand_node, level);
                auto neighbors = degree_ptr + 1;
                auto num_edges = *degree_ptr;
                // visiting neighbors of candidate node
                for (index_type j = 0; j < num_edges; j++) {
                    index_type next_node = neighbors[j];
                    if (!set_of_visited_nodes.is_visited(next_node)) {
                        set_of_visited_nodes.mark_visited(next_node);
                        dist_t next_lb_dist;
                        next_lb_dist = dist_fn(query, graph_l0.get_node_feat(next_node), dist_feat_dim);
                        if (next_lb_dist < topk_ub_dist || topk_queue.size() < efS) {
                            topk_queue.emplace(next_lb_dist, next_node);
                            cand_queue.emplace(-next_lb_dist, next_node);
                            if (topk_queue.size() > efS) {
                                topk_queue.pop();
                            }
                            if (!topk_queue.empty()) {
                                topk_ub_dist = topk_queue.top().first;
                            }
                        }
                    }
                }
            }
            return topk_queue;
        }

        // Algorithm 5 of HNSW paper
        // current implementation is not thread safe
        max_heap_t predict_single(const void *query, index_type efS, index_type topk=10) {
            index_type curr_node = this->init_node;
            auto &G1 = graph_l1;
            auto &G0 = graph_l0;
            // specialized search_layer for layer l=1,...,L because its faster for efS=1
            dist_t curr_dist = dist_fn(query, G0.get_node_feat(init_node), dist_feat_dim);
            for (index_type curr_level = this->max_level; curr_level >= 1; curr_level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    auto degree_ptr = G1.get_node_degree_ptr(curr_node, curr_level);
                    auto neighbors = degree_ptr + 1;
                    auto num_edges = *degree_ptr;
                    for (index_type j = 0; j < num_edges; j++) {
                        index_type next_node = neighbors[j];
                        dist_t next_dist = dist_fn(query, G0.get_node_feat(next_node), dist_feat_dim);
                        if (next_dist < curr_dist) {
                            curr_dist = next_dist;
                            curr_node = next_node;
                            changed = true;
                        }
                    }
                }
            }
            // generalized search_layer for layer=0 for efS >= 1
            auto topk_queue = search_layer(query, curr_node, std::max(efS, topk), 0);
            // remove extra when efS > topk
            while (topk_queue.size() > topk) {
                topk_queue.pop();
            }
            // reverse order from smallest distance indices first
            max_heap_t results;
            while (topk_queue.size() > 0) {
                auto rez = topk_queue.top();
                results.push(pair_t(rez.first, rez.second));
                topk_queue.pop();
            }
            return results;
        }

    };


} // end of namespace ann
} // end of namespace pecos

