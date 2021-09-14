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

#include <cmath>
#include <cstdlib>
#include <queue>
#include <random>
#include <vector>
#include <mutex>

#include <type_traits>

#include "utils/matrix.hpp"
#include "ann/feat_vectors.hpp"

namespace pecos {

namespace ann {

    typedef uint32_t index_type;
    typedef uint64_t mem_index_type;

    struct NeighborHood {
        index_type* degree_ptr;
        index_type* neighbor_ptr;

        NeighborHood(void *memory_ptr) {
            char* curr_ptr = reinterpret_cast<char*>(memory_ptr);
            degree_ptr = reinterpret_cast<index_type*>(curr_ptr);
            curr_ptr += sizeof(index_type);
            neighbor_ptr = reinterpret_cast<index_type*>(curr_ptr);
        }

        void set_degree(index_type degree) {
            *degree_ptr = degree;
        }

        const index_type& degree() const {
            return *degree_ptr;
        }

        index_type* begin() { return neighbor_ptr; }
        const index_type* begin() const { return neighbor_ptr; }

        index_type* end() { return neighbor_ptr + degree(); }
        const index_type* end() const { return neighbor_ptr + degree(); }

        index_type& operator[](size_t i) { return neighbor_ptr[i]; }
        const index_type& operator[](size_t i) const { return neighbor_ptr[i]; }

        void push_back(index_type dst) {
            neighbor_ptr[*degree_ptr] = dst;
            *degree_ptr += 1;
        }
        void clear() {
            *degree_ptr = 0;
        }
    };

    struct GraphBase {
        virtual const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const = 0;

        NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) {
            return const_cast<const GraphBase&>(*this).get_neighborhood(node_id, dummy_level_id);
        }
    };

    template<class FeatVec_T>
    struct GraphL0 : GraphBase {
        typedef FeatVec_T feat_vec_t;
        index_type num_node;
        index_type feat_dim;
        index_type max_degree;
        index_type node_mem_size;
        std::vector<uint64_t> mem_start_of_node;
        std::vector<char> buffer;

        size_t neighborhood_memory_size() const { return (1 + max_degree) * sizeof(index_type); }

        template<class MAT_T>
        void init(const MAT_T& feat_mat, index_type max_degree) {
            this->num_node = feat_mat.rows;
            this->feat_dim = feat_mat.cols;
            this->max_degree = max_degree;
            mem_start_of_node.resize(num_node + 1);
            mem_start_of_node[0] = 0;
            for(size_t i = 0; i < num_node; i++) {
                const feat_vec_t& xi(feat_mat.get_row(i));
                mem_start_of_node[i + 1] = mem_start_of_node[i] + neighborhood_memory_size() + xi.memory_size();
            }
            buffer.resize(mem_start_of_node[num_node], 0);
            if(feat_vec_t::is_fixed_size::value) {
                node_mem_size = buffer.size() / num_node;
            }

            // get_node_feat_ptr must appear after memory allocation (buffer.resize())
            for(size_t i = 0; i < num_node; i++) {
                const feat_vec_t& xi(feat_mat.get_row(i));
                xi.copy_to(get_node_feat_ptr(i));
            }
        }

        inline feat_vec_t get_node_feat(index_type node_id) const {
            return feat_vec_t(const_cast<void*>(get_node_feat_ptr(node_id)));
        }

        inline void prefetch_node_feat(index_type node_id) const {
#ifdef USE_SSE
             _mm_prefetch((char*)get_node_feat_ptr(node_id), _MM_HINT_T0);
#elif defined(__GNUC__)
             __builtin_prefetch((char*)get_node_feat_ptr(node_id), 0, 0);
#endif
        }

        inline const void* get_node_feat_ptr(index_type node_id) const {
            if(feat_vec_t::is_fixed_size::value) {
                return &buffer[node_id * (mem_index_type) node_mem_size + neighborhood_memory_size()];
            } else {
                return &buffer[mem_start_of_node[node_id] + neighborhood_memory_size()];
            }
        }

        inline void* get_node_feat_ptr(index_type node_id) {
            return const_cast<void*>(const_cast<const GraphL0<FeatVec_T>&>(*this).get_node_feat_ptr(node_id));
        }

        inline const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const {
            const index_type *neighborhood_ptr = nullptr;
            if(feat_vec_t::is_fixed_size::value) {
                neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer.data()[node_id * (mem_index_type) node_mem_size]);
            } else {
                neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer[mem_start_of_node[node_id]]);
            }
            return NeighborHood((void*)neighborhood_ptr);
        }
    };

    struct GraphL1 : GraphBase {
        index_type num_node;
        index_type max_level;
        index_type max_degree;
        index_type node_mem_size;
        index_type level_mem_size;
        std::vector<index_type> buffer;

        template<class MAT_T>
        void init(const MAT_T& feat_mat, index_type max_degree, index_type max_level) {
            this->num_node = feat_mat.rows;
            this->max_level = max_level;
            this->max_degree = max_degree;
            this->level_mem_size = 1 + max_degree;
            this->node_mem_size = max_level * this->level_mem_size;
            buffer.resize(num_node * (mem_index_type) this->node_mem_size, 0);
        }

        inline const NeighborHood get_neighborhood(index_type node_id, index_type level_id=0) const {
            const index_type *neighborhood_ptr = &buffer[node_id * (mem_index_type) this->node_mem_size + (level_id - 1) * (mem_index_type) this->level_mem_size];
            return NeighborHood((void*)neighborhood_ptr);
        }
    };

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

    template <typename T>
    struct CompareByFirst {
        constexpr bool operator()(T const &a, T const &b) const noexcept {
            return a.first < b.first;
        }
    };

    // PECOS-HNSW Interface
    template<typename dist_t, class FeatVec_T>
    struct HNSW {
        typedef FeatVec_T feat_vec_t;
        typedef SetOfVistedNodes<unsigned short int> set_of_visited_nodes_t;
        typedef typename std::pair<dist_t, index_type> pair_t;
        typedef typename std::priority_queue<pair_t, std::vector<pair_t>, CompareByFirst<pair_t>> max_heap_t;

        struct Searcher : SetOfVistedNodes<unsigned short int> {
            typedef HNSW<dist_t, FeatVec_T> hnsw_t;
            const hnsw_t* hnsw;

            Searcher(const hnsw_t* hnsw=nullptr):
                SetOfVistedNodes<unsigned short int>(hnsw? hnsw->num_node : 0),
                hnsw(hnsw) {}

            max_heap_t search_layer(const feat_vec_t& query, index_type init_node, index_type efS, index_type level) {
                return hnsw->search_layer(query, init_node, efS, level, *this);
            }

            std::vector<pair_t> predict_single(const feat_vec_t& query, index_type efS, index_type topk) {
                return hnsw->predict_single(query, efS, topk, *this);
            }
        };

        Searcher create_searcher() const {
            return Searcher(this);
        }

        // scalar variables
        index_type num_node;
        index_type maxM;  // number of max out degree for layer l=1,...,L
        index_type maxM0; // number of max out degree for layer l=0
        index_type efC;   // size of priority queue for construction time
        index_type max_level;
        index_type init_node;

        // data structures for multi-level graph
        GraphL0<feat_vec_t> graph_l0;   // neighborhood graph along with feature vectors at Level 0
        GraphL1 graph_l1;               // neighborhood graphs from Level 1 and above
        std::vector<index_type> node2level_vec;
        std::default_random_engine level_generator_;

        // destructor
        ~HNSW() {}
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

                for (auto& second_pair : return_list) {
                    dist_t curdist = feat_vec_t::distance(
                        graph_l0.get_node_feat(second_pair.second),
                        graph_l0.get_node_feat(curent_pair.second)
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

            for (auto& curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }

        // line 10-17, Algorithm 1 of HNSW paper
        template<bool lock_free=true>
        index_type mutually_connect(index_type query_id, max_heap_t &top_candidates, index_type level, std::vector<std::mutex>* mtx_nodes=nullptr) {
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

            GraphBase *G;
            if (level == 0) {
                G = &graph_l0;
            } else {
                G = &graph_l1;
            }

            auto add_link = [&](index_type src, index_type dst) {
                std::unique_lock<std::mutex>* lock_src = nullptr;
                if(!lock_free) {
                    lock_src = new std::unique_lock<std::mutex>(mtx_nodes->at(src));
                }

                auto neighbors = G->get_neighborhood(src, level);

                if (neighbors.degree() > Mcurmax)
                    throw std::runtime_error("Bad value of size of neighbors for this src node");
                if (src == dst)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > node2level_vec[src])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                if (neighbors.degree() < Mcurmax) {
                    neighbors.push_back(dst);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = feat_vec_t::distance(
                        graph_l0.get_node_feat(src),
                        graph_l0.get_node_feat(dst)
                    );
                    // Heuristic:
                    max_heap_t candidates;
                    candidates.emplace(d_max, dst);
                   for(auto& dst : neighbors) {
                        dist_t dist_j = feat_vec_t::distance(
                            graph_l0.get_node_feat(src),
                            graph_l0.get_node_feat(dst)
                        );
                        candidates.emplace(dist_j, dst);
                    }
                    get_neighbors_heuristic(candidates, Mcurmax);

                    neighbors.clear();
                    index_type indx = 0;
                    while (candidates.size() > 0) {
                        neighbors.push_back(candidates.top().second);
                        candidates.pop();
                        indx++;
                    }
                }

                if(!lock_free) {
                    delete lock_src;
                }
            };

            for (auto& dst : selected_neighbors) {
                add_link(query_id, dst);
                add_link(dst, query_id);
            }

            index_type next_closest_entry_point = selected_neighbors.back();
            return next_closest_entry_point;
        }

        // train, Algorithm 1 of HNSW paper (i.e., construct HNSW graph)
        template<class MAT_T>
        void train(const MAT_T &X_trn, index_type M, index_type efC, index_type max_level_upper_bound, int threads=1) {

            this->num_node = X_trn.rows;
            this->maxM = M;
            this->maxM0 = 2 * M;
            this->efC = efC;

            graph_l0.init(X_trn, this->maxM0);
            graph_l1.init(X_trn, this->maxM, max_level_upper_bound);
            node2level_vec.resize(num_node);

            this->max_level = 0;
            this->init_node = 0;

            // workspace to store thread-local variables
            struct workspace_t {
                HNSW<dist_t, FeatVec_T>& hnsw;
                std::mutex mtx_global;
                std::vector<std::mutex> mtx_nodes;
                std::vector<Searcher> searchers;
                workspace_t(HNSW<dist_t, FeatVec_T>& hnsw, int threads=1): hnsw(hnsw), mtx_nodes(hnsw.num_node) {
                    for(int i = 0; i < threads; i++) {
                        searchers.emplace_back(Searcher(&hnsw));
                    }
                }
            };

            // a thread-safe functor to add point
            auto add_point = [&](index_type query_id, workspace_t& ws, int thread_id, bool lock_free) {
                auto& hnsw = ws.hnsw;
                auto& graph_l0 = hnsw.graph_l0;
                auto& graph_l1 = hnsw.graph_l1;
                auto& searcher = ws.searchers[thread_id];
                // this is m_l defined in Sec 4.1 of HNSW paper
                float mult_l = 1.0 / log(1.0 * hnsw.maxM);

                // sample the query node's level
                index_type query_level = std::min<index_type>(hnsw.get_random_level(mult_l), graph_l1.max_level);
                hnsw.node2level_vec[query_id] = query_level;

                // obtain the global lock as we might need to change max_level and init_node
                std::unique_lock<std::mutex>* lock_global = nullptr;
                if(query_level > hnsw.max_level) {
                    lock_global = new std::unique_lock<std::mutex>(ws.mtx_global);
                }

                // make a copy about the current max_level and enterpoint_id
                auto max_level = hnsw.max_level;
                auto curr_node = hnsw.init_node;

                const feat_vec_t& query_feat_ptr = graph_l0.get_node_feat(query_id);

                bool is_first_node = (query_id == 0);
                if(is_first_node) {
                    hnsw.init_node = query_id;
                    hnsw.max_level = query_level;
                } else {
                    // find entrypoint with efS = 1 from level = local max_level to 1.
                    if(query_level < max_level) {
                        dist_t curr_dist = feat_vec_t::distance(
                            query_feat_ptr,
                            graph_l0.get_node_feat(curr_node)
                        );

                        for(auto level = max_level; level > query_level; level--) {
                            bool changed = true;
                            while (changed) {
                                changed = false;
                                std::unique_lock<std::mutex> lock_node(ws.mtx_nodes[curr_node]);
                                auto neighbors = graph_l1.get_neighborhood(curr_node, level);
                                for(auto& next_node : neighbors) {
                                    dist_t next_dist = feat_vec_t::distance(
                                        query_feat_ptr,
                                        graph_l0.get_node_feat(next_node)
                                    );
                                    if (next_dist < curr_dist) {
                                        curr_dist = next_dist;
                                        curr_node = next_node;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    if(lock_free) {
                        for (auto level = std::min(query_level, max_level); ; level--) {
                            auto top_candidates = search_layer<true>(query_feat_ptr, curr_node, this->efC, level, searcher, &ws.mtx_nodes);
                            curr_node = mutually_connect<true>(query_id, top_candidates, level, &ws.mtx_nodes);
                            if (level == 0) { break; }
                        }
                    } else {
                        for (auto level = std::min(query_level, max_level); ; level--) {
                            auto top_candidates = search_layer<false>(query_feat_ptr, curr_node, this->efC, level, searcher, &ws.mtx_nodes);
                            curr_node = mutually_connect<false>(query_id, top_candidates, level, &ws.mtx_nodes);
                            if (level == 0) { break; }
                        }
                    }

                    // if(query_level > hnsw.node2level_vec[hnsw.enterpoint_id])  // used is nmslib.
                    if(query_level > hnsw.max_level) {  // used in hnswlib.
                        hnsw.max_level = query_level;
                        hnsw.init_node = query_id;
                    }
                }

                if(lock_global != nullptr) {
                    delete lock_global;
                }
            }; // end of add_point

            threads = (threads <= 0) ? omp_get_num_procs() : threads;
            omp_set_num_threads(threads);
            workspace_t ws(*this, threads);
            bool lock_free = (threads == 1);
#pragma omp parallel for schedule(dynamic, 1)
            for (index_type query_id = 0; query_id < num_node; query_id++) {
                int thread_id = omp_get_thread_num();
                add_point(query_id, ws, thread_id, lock_free);
            }
        }

        // Algorithm 2 of HNSW paper
        template<bool lock_free=true>
        max_heap_t search_layer(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            max_heap_t topk_queue;
            max_heap_t cand_queue;
            searcher.reset();

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                graph_l0.get_node_feat(init_node)
            );
            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(-topk_ub_dist, init_node);
            searcher.mark_visited(init_node);

            const GraphBase *G;
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
                std::unique_lock<std::mutex>* lock_node = nullptr;
                if(!lock_free){
                    lock_node = new std::unique_lock<std::mutex>(mtx_nodes->at(cand_node));
                }
                // visiting neighbors of candidate node
                const auto neighbors = G->get_neighborhood(cand_node, level);
                graph_l0.prefetch_node_feat(neighbors[0]);
                for (index_type j = 0; j < neighbors.degree(); j++) {
                    graph_l0.prefetch_node_feat(neighbors[j + 1]);
                    auto next_node = neighbors[j];
                    if (!searcher.is_visited(next_node)) {
                        searcher.mark_visited(next_node);
                        dist_t next_lb_dist;
                        next_lb_dist = feat_vec_t::distance(
                            query,
                            graph_l0.get_node_feat(next_node)
                        );
                        if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                            cand_queue.emplace(-next_lb_dist, next_node);
                            graph_l0.prefetch_node_feat(cand_queue.top().second);
                            topk_queue.emplace(next_lb_dist, next_node);
                            if (topk_queue.size() > efS) {
                                topk_queue.pop();
                            }
                            if (!topk_queue.empty()) {
                                topk_ub_dist = topk_queue.top().first;
                            }
                        }
                    }
                }
                if(!lock_free){
                    delete lock_node;
                }
            }
            return topk_queue;
        }

        // Algorithm 5 of HNSW paper, thread-safe inference
        std::vector<pair_t> predict_single(const feat_vec_t& query, index_type efS, index_type topk, Searcher& searcher) const {
            index_type curr_node = this->init_node;
            auto &G1 = graph_l1;
            auto &G0 = graph_l0;
            // specialized search_layer for layer l=1,...,L because its faster for efS=1
            dist_t curr_dist = feat_vec_t::distance(
                query,
                G0.get_node_feat(init_node)
            );
            for (index_type curr_level = this->max_level; curr_level >= 1; curr_level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto neighbors = G1.get_neighborhood(curr_node, curr_level);
                    graph_l0.prefetch_node_feat(neighbors[0]);
                    for (index_type j = 0; j < neighbors.degree(); j++) {
                        graph_l0.prefetch_node_feat(neighbors[j + 1]);
                        auto next_node = neighbors[j];
                        dist_t next_dist = feat_vec_t::distance(
                            query,
                            G0.get_node_feat(next_node)
                        );
                        if (next_dist < curr_dist) {
                            curr_dist = next_dist;
                            curr_node = next_node;
                            changed = true;
                        }
                    }
                }
            }
            // generalized search_layer for layer=0 for efS >= 1
            auto topk_queue = search_layer(query, curr_node, std::max(efS, topk), 0, searcher);
            // remove extra when efS > topk
            while (topk_queue.size() > topk) {
                topk_queue.pop();
            }
            // return with smallest (distance,index) pair first
            // if topk < number of indexed items
            if (topk_queue.size() != topk) {
                throw std::runtime_error("efS can not be smaller than topk. try to use larger efS or smaller topk!");
            }
            std::vector<pair_t> results(topk);
            size_t sz = topk_queue.size();
            while (topk_queue.size() > 0) {
                results[--sz] = topk_queue.top();
                topk_queue.pop();
            }
            return results;
        }
    };


} // end of namespace ann

} // end of namespace pecos

