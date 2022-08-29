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

#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>



#include "ann/feat_vectors.hpp"
#include "ann/quantizer.hpp"
#include "third_party/nlohmann_json/json.hpp"
#include "utils/file_util.hpp"
#include "utils/matrix.hpp"
#include "utils/random.hpp"
#include "utils/type_util.hpp"

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

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&feat_dim, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&node_mem_size, 1, fp);
            size_t sz = mem_start_of_node.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<uint64_t>(&mem_start_of_node[0], sz, fp);
            }
            sz = buffer.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<char>(&buffer[0], sz, fp);
            }
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&feat_dim, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&node_mem_size, 1, fp);
            size_t sz = 0;
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            mem_start_of_node.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<uint64_t>(&mem_start_of_node[0], sz, fp);
            }
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            buffer.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<char>(&buffer[0], sz, fp);
            }
        }

        template<class MAT_T>
        void init(const MAT_T& feat_mat, index_type max_degree) {
            this->num_node = feat_mat.rows;
            this->feat_dim = feat_mat.cols;
            this->max_degree = max_degree;
            mem_start_of_node.resize(num_node + 1);
            mem_start_of_node[0] = 0;
            for (size_t i = 0; i < num_node; i++) {
                const feat_vec_t& xi(feat_mat.get_row(i));
                mem_start_of_node[i + 1] = mem_start_of_node[i] + neighborhood_memory_size() + xi.memory_size();
            }
            buffer.resize(mem_start_of_node[num_node], 0);
            if (feat_vec_t::is_fixed_size::value) {
                node_mem_size = buffer.size() / num_node;
            }

            // get_node_feat_ptr must appear after memory allocation (buffer.resize())
            for (size_t i = 0; i < num_node; i++) {
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
            if (feat_vec_t::is_fixed_size::value) {
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
            if (feat_vec_t::is_fixed_size::value) {
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

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&node_mem_size, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&level_mem_size, 1, fp);
            size_t sz = buffer.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<index_type>(&buffer[0], sz, fp);
            }
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_degree, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&node_mem_size, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&level_mem_size, 1, fp);
            size_t sz = 0;
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            buffer.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<index_type>(&buffer[0], sz, fp);
            }
        }

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

    template<class FeatVec_T>
    struct GraphProductQuantizer4Bits : GraphBase {
        typedef FeatVec_T feat_vec_t;
        ProductQuantizer4Bits quantizer;
        index_type num_node;
        // code_dimension is number of 4 bits code used to encode a data point in GraphPQ4Bits
        // code_dimension can be different from parameter num_local_codebooks in quantizer
        // as we might adjust code_dimension to make it divisble by 4. More details can be
        // found in pad_parameters function of ann/quantizer_impl/x86.hpp
        size_t code_dimension;
        // code_offset helps to locate memory position containing neighboring codes
        size_t code_offset;  
        size_t node_mem_size;
        index_type max_degree;
        std::vector<uint64_t> mem_start_of_node;
        std::vector<char> buffer;

        void save(FILE *fp) const {
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<size_t>(&code_dimension, 1, fp);
            pecos::file_util::fput_multiple<size_t>(&code_offset, 1, fp);
            pecos::file_util::fput_multiple<size_t>(&node_mem_size, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_degree, 1, fp);
            size_t sz = mem_start_of_node.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<uint64_t>(&mem_start_of_node[0], sz, fp);
            }
            sz = buffer.size();
            pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
            if (sz) {
                pecos::file_util::fput_multiple<char>(&buffer[0], sz, fp);
            }
            quantizer.save(fp);
            fclose(fp);
        }

        void load(FILE *fp) {
            pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fget_multiple<size_t>(&code_dimension, 1, fp);
            pecos::file_util::fget_multiple<size_t>(&code_offset, 1, fp);
            pecos::file_util::fget_multiple<size_t>(&node_mem_size, 1, fp);
            pecos::file_util::fget_multiple<index_type>(&max_degree, 1, fp);
            size_t sz = 0;
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            mem_start_of_node.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<uint64_t>(&mem_start_of_node[0], sz, fp);
            }
            pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
            buffer.resize(sz);
            if (sz) {
                pecos::file_util::fget_multiple<char>(&buffer[0], sz, fp);
            }

            quantizer.load(fp);

            fclose(fp);
        }


        void build_quantizer(const pecos::drm_t& X_trn, index_type subspace_dimension, index_type sub_sample_points) {
            size_t code_dimension = X_trn.cols;
            if (subspace_dimension == 0) {
                if (code_dimension >= 400) {
                    code_dimension =  code_dimension % 2 == 0 ? code_dimension / 2 : code_dimension / 2 + 1;
                }
            } else {
                code_dimension = code_dimension / subspace_dimension;
            }
            // currently, we don't support padding 0 on X_trn, so the cols of X_trn must be divisible by subspace_dimension.
            // otherwise, we will throw error in quantizer.train().
            quantizer.train(X_trn, code_dimension, sub_sample_points);
            quantizer.pack_codebook_for_inference();
            this->code_dimension = code_dimension;
        }

        void build_graph(GraphL0<feat_vec_t>& G) {
            max_degree = G.max_degree;
            quantizer.pad_parameters(max_degree, code_dimension);
            num_node = G.num_node;
            size_t num_of_local_centroids = quantizer.num_of_local_centroids;
            size_t neighbor_size = (1 + max_degree) * sizeof(index_type);
            code_offset = neighbor_size;

            std::vector<std::vector<uint8_t>> X_trn_codes(num_node, std::vector<uint8_t> (code_dimension, 0));
            for (size_t i = 0 ; i < num_node ; i++) {
                quantizer.encode(G.get_node_feat(i).val, X_trn_codes[i].data());
            }

            node_mem_size = neighbor_size + max_degree * code_dimension / 2;
            mem_start_of_node.resize(num_node + 1);
            mem_start_of_node[0] = 0;

            for (size_t i = 0; i < num_node; i++) {
                mem_start_of_node[i + 1] = mem_start_of_node[i] + node_mem_size;
            }

            buffer.resize(mem_start_of_node[num_node], 0);

            for (size_t i = 0; i < num_node; i++) {
                std::vector<std::vector<uint8_t>> neighbor_codes(max_degree, std::vector<uint8_t> (code_dimension, 0));

                memcpy(&buffer[mem_start_of_node[i]], &G.buffer[G.mem_start_of_node[i]], (1 + G.max_degree) * sizeof(index_type));

                index_type size = *reinterpret_cast<index_type*>(&G.buffer[G.mem_start_of_node[i]]);
                for (index_type j = 0; j < size; j++) {
                    index_type member = *reinterpret_cast<index_type *>(&G.buffer[G.mem_start_of_node[i] + sizeof(index_type) + j * sizeof(index_type)]);
                    memcpy(neighbor_codes[j].data(), X_trn_codes[member].data(), code_dimension);
                }

                index_type processed_num_of_neighbors = 0;
                std::vector<char> group_transposed_graph_codes(max_degree / 2 * code_dimension, 0);

                while (processed_num_of_neighbors < size) {
                    std::vector<char> group_code(num_of_local_centroids / 2 * code_dimension, 0);

                    for (index_type k = 0; k < code_dimension; k++) {
                        for (index_type j = 0; j < num_of_local_centroids; j += 2) {
                            uint8_t obj = neighbor_codes[processed_num_of_neighbors + j][k];
                            obj += (neighbor_codes[processed_num_of_neighbors + j + 1][k] << 4);
                            group_code[k * num_of_local_centroids / 2 + j / 2] = obj;
                        }
                    }
                    memcpy(&group_transposed_graph_codes[processed_num_of_neighbors * code_dimension / 2], &group_code[0], num_of_local_centroids * code_dimension / 2);
                    processed_num_of_neighbors += num_of_local_centroids;
                }
                memcpy(&buffer[mem_start_of_node[i] + (1 + max_degree) * sizeof(index_type)], group_transposed_graph_codes.data(), max_degree * code_dimension / 2);
            }
        }

        inline const char* get_neighbor_codes(index_type node_id) const {
            return &buffer[mem_start_of_node[node_id] + code_offset];
        }
        inline const NeighborHood get_neighborhood(index_type node_id, index_type dummy_level_id=0) const {
            const index_type *neighborhood_ptr = nullptr;
            neighborhood_ptr = reinterpret_cast<const index_type*>(&buffer.data()[node_id * (mem_index_type) node_mem_size]);
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
            if (curr_token == init_token) {
                std::fill(buffer.begin(), buffer.end(), init_token);
                curr_token = init_token + 1;
            }
        }
    };

    template <typename T1, typename T2>
    struct Pair {
        T1 dist;
        T2 node_id;
        Pair(const T1& dist=T1(), const T2& node_id=T2()): dist(dist), node_id(node_id) {}
        bool operator<(const Pair<T1, T2>& other) const { return dist < other.dist; }
        bool operator>(const Pair<T1, T2>& other) const { return dist > other.dist; }
    };

    template<typename T, typename _Compare = std::less<T>>
    struct heap_t : public std::vector<T> {
        typedef typename std::vector<T> container_type;
        typedef typename container_type::value_type value_type;
        typedef typename container_type::reference reference;
        typedef typename container_type::const_reference const_reference;
        typedef typename container_type::size_type size_type;
        typedef _Compare value_compare;

        _Compare comp;

        const_reference top() const { return this->front(); }

        void push(const value_type& __x) {
            this->push_back(__x);
            std::push_heap(this->begin(), this->end(), comp);
        }

#if __cplusplus >= 201103L
        void push(value_type&& __x) {
            this->push_back(std::move(__x));
            std::push_heap(this->begin(), this->end(), comp);
        }

        template<typename... _Args>
        void emplace(_Args&&... __args) {
            this->emplace_back(std::forward<_Args>(__args)...);
            std::push_heap(this->begin(), this->end(), comp);
        }
#endif
        void pop() {
            std::pop_heap(this->begin(), this->end(), comp);
            this->pop_back();
        }
    };

    // PECOS-HNSW Interface
    template<typename dist_t, class FeatVec_T>
    struct HNSW {
        typedef FeatVec_T feat_vec_t;
        typedef Pair<dist_t, index_type> pair_t;
        typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
        typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;

        struct Searcher : SetOfVistedNodes<unsigned short int> {
            typedef SetOfVistedNodes<unsigned short int> set_of_visited_nodes_t;
            typedef HNSW<dist_t, FeatVec_T> hnsw_t;
            typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
            typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;

            const hnsw_t* hnsw;
            max_heap_t topk_queue;
            min_heap_t cand_queue;

            Searcher(const hnsw_t* _hnsw=nullptr):
                SetOfVistedNodes<unsigned short int>(_hnsw? _hnsw->num_node : 0),
                hnsw(_hnsw)
            {}

            void reset() {
                set_of_visited_nodes_t::reset();
                topk_queue.clear();
                cand_queue.clear();
            }

            max_heap_t& search_level(const feat_vec_t& query, index_type init_node, index_type efS, index_type level) {
                return hnsw->search_level(query, init_node, efS, level, *this);
            }

            max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk) {
                return hnsw->predict_single(query, efS, topk, *this);
            }
        };

        Searcher create_searcher() const {
            return Searcher(this);
        }

        // scalar variables
        index_type num_node;
        index_type maxM;   // max number of out-degree for level l=1,...,L
        index_type maxM0;  // max number of out-degree for level l=0
        index_type efC;    // size of priority queue for construction time
        index_type max_level;
        index_type init_node;

        // data structures for multi-level graph
        GraphL0<feat_vec_t> graph_l0;   // neighborhood graph along with feature vectors at level 0
        GraphL1 graph_l1;               // neighborhood graphs from level 1 and above

        // destructor
        ~HNSW() {}

        static nlohmann::json load_config(const std::string& filepath) {
            std::ifstream loadfile(filepath);
            std::string json_str;
            if (loadfile.is_open()) {
                json_str.assign(
                    std::istreambuf_iterator<char>(loadfile),
                    std::istreambuf_iterator<char>()
                );
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            auto j_param = nlohmann::json::parse(json_str);
            std::string hnsw_t_cur = pecos::type_util::full_name<HNSW>();
            std::string hnsw_t_inp = j_param["hnsw_t"];
            if (hnsw_t_cur != hnsw_t_inp) {
                throw std::invalid_argument("Inconsistent HNSW_T: hnsw_t_cur = " + hnsw_t_cur  + " hnsw_t_cur = " + hnsw_t_inp);
            }
            return j_param;
        }

        void save_config(const std::string& filepath) const {
            nlohmann::json j_params = {
                {"hnsw_t", pecos::type_util::full_name<HNSW>()},
                {"version", "v1.0"},
                {"train_params", {
                    {"num_node", this->num_node},
                    {"maxM", this->maxM},
                    {"maxM0", this->maxM0},
                    {"efC", this->efC},
                    {"max_level", this->max_level},
                    {"init_node", this->init_node}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if (savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void save(const std::string& model_dir) const {
            if (mkdir(model_dir.c_str(), 0777) == -1) {
                if (errno != EEXIST) {
                    throw std::runtime_error("Unable to create save folder at " + model_dir);
                }
            }
            save_config(model_dir + "/config.json");
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "wb");
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM0, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&efC, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&init_node, 1, fp);
            graph_l0.save(fp);
            graph_l1.save(fp);
            fclose(fp);
        }

        void load(const std::string& model_dir) {
            auto config = load_config(model_dir + "/config.json");
            std::string version = config.find("version") != config.end() ? config["version"] : "not found";
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "rb");
            if (version == "v1.0") {
                pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM0, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&efC, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&max_level, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&init_node, 1, fp);
                graph_l0.load(fp);
                graph_l1.load(fp);
            } else {
                throw std::runtime_error("Unable to load this binary with version = " + version);
            }
            fclose(fp);
        }

        // Algorithm 4 of HNSW paper
        void get_neighbors_heuristic(max_heap_t &top_candidates, const index_type M) {
            if (top_candidates.size() < M) { return; }

            min_heap_t queue_closest;
            std::vector<pair_t> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(top_candidates.top());
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M) {
                    break;
                }
                auto curent_pair = queue_closest.top();
                dist_t dist_to_query = curent_pair.dist;
                queue_closest.pop();
                bool good = true;

                for (auto& second_pair : return_list) {
                    dist_t curdist = feat_vec_t::distance(
                        graph_l0.get_node_feat(second_pair.node_id),
                        graph_l0.get_node_feat(curent_pair.node_id)
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
                top_candidates.emplace(curent_pair);
            }
        }

        // line 10-17, Algorithm 1 of HNSW paper
        // it is the caller's responsibility to make sure top_candidates are available in the graph of this level.
        template<bool lock_free=true>
        index_type mutually_connect(index_type src_node_id, max_heap_t &top_candidates, index_type level, std::vector<std::mutex>* mtx_nodes=nullptr) {
            index_type Mcurmax = level ? this->maxM : this->maxM0;
            get_neighbors_heuristic(top_candidates, this->maxM);
            if (top_candidates.size() > this->maxM) {
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");
            }

            std::vector<index_type> selected_neighbors;
            selected_neighbors.reserve(this->maxM);
            while (top_candidates.size() > 0) {
                selected_neighbors.push_back(top_candidates.top().node_id);
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
                if (!lock_free) {
                    lock_src = new std::unique_lock<std::mutex>(mtx_nodes->at(src));
                }

                auto neighbors = G->get_neighborhood(src, level);

                if (neighbors.degree() > Mcurmax)
                    throw std::runtime_error("Bad value of size of neighbors for this src node");
                if (src == dst)
                    throw std::runtime_error("Trying to connect an element to itself");

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
                    for (auto& dst : neighbors) {
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
                        neighbors.push_back(candidates.top().node_id);
                        candidates.pop();
                        indx++;
                    }
                }

                if (!lock_free) {
                    delete lock_src;
                }
            };

            for (auto& dst : selected_neighbors) {
                add_link(src_node_id, dst);
                add_link(dst, src_node_id);
            }

            index_type next_closest_entry_point = selected_neighbors.back();
            return next_closest_entry_point;
        }

        // train, Algorithm 1 of HNSW paper (i.e., construct HNSW graph)
        // if max_level_upper_bound >= 0, the number of lavels in the hierarchical part is upper bounded by the give number
        template<class MAT_T>
        void train(const MAT_T &X_trn, index_type M, index_type efC, int threads=1, int max_level_upper_bound=-1) {
            // workspace to store thread-local variables
            struct workspace_t {
                HNSW<dist_t, FeatVec_T>& hnsw;
                std::mutex mtx_global;
                std::vector<std::mutex> mtx_nodes;
                std::vector<index_type> node2level;
                std::vector<Searcher> searchers;
                workspace_t(HNSW<dist_t, FeatVec_T>& hnsw, int threads=1):
                    hnsw(hnsw), mtx_nodes(hnsw.num_node), node2level(hnsw.num_node) {
                    for (int i = 0; i < threads; i++) {
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

                // sample the query node's level
                auto query_level = ws.node2level[query_id];

                // obtain the global lock as we might need to change max_level and init_node
                std::unique_lock<std::mutex>* lock_global = nullptr;
                if (query_level > hnsw.max_level) {
                    lock_global = new std::unique_lock<std::mutex>(ws.mtx_global);
                }

                // make a copy about the current max_level and enterpoint_id
                auto max_level = hnsw.max_level;
                auto curr_node = hnsw.init_node;

                const feat_vec_t& query_feat = graph_l0.get_node_feat(query_id);

                bool is_first_node = (query_id == 0);
                if (is_first_node) {
                    hnsw.init_node = query_id;
                    hnsw.max_level = query_level;
                } else {
                    // find entrypoint with efS = 1 from level = local max_level to 1.
                    if (query_level < max_level) {
                        dist_t curr_dist = feat_vec_t::distance(
                            query_feat,
                            graph_l0.get_node_feat(curr_node)
                        );

                        for (auto level = max_level; level > query_level; level--) {
                            bool changed = true;
                            while (changed) {
                                changed = false;
                                std::unique_lock<std::mutex> lock_node(ws.mtx_nodes[curr_node]);
                                auto neighbors = graph_l1.get_neighborhood(curr_node, level);
                                for (auto& next_node : neighbors) {
                                    dist_t next_dist = feat_vec_t::distance(
                                        query_feat,
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
                    if (lock_free) {
                        for (auto level = std::min(query_level, max_level); ; level--) {
                            auto& top_candidates = search_level<true>(query_feat, curr_node, this->efC, level, searcher, &ws.mtx_nodes);
                            curr_node = mutually_connect<true>(query_id, top_candidates, level, &ws.mtx_nodes);
                            if (level == 0) { break; }
                        }
                    } else {
                        for (auto level = std::min(query_level, max_level); ; level--) {
                            auto& top_candidates = search_level<false>(query_feat, curr_node, this->efC, level, searcher, &ws.mtx_nodes);
                            curr_node = mutually_connect<false>(query_id, top_candidates, level, &ws.mtx_nodes);
                            if (level == 0) { break; }
                        }
                    }

                    // if(query_level > ws.node2level[hnsw.enterpoint_id])  // used in nmslib.
                    if (query_level > hnsw.max_level) {  // used in hnswlib.
                        hnsw.max_level = query_level;
                        hnsw.init_node = query_id;
                    }
                }

                if (lock_global != nullptr) {
                    delete lock_global;
                }
            };  // end of add_point

            this->num_node = X_trn.rows;
            this->maxM = M;
            this->maxM0 = 2 * M;
            this->efC = efC;

            threads = (threads <= 0) ? omp_get_num_procs() : threads;
            omp_set_num_threads(threads);
            workspace_t ws(*this, threads);

            // pre-compute level for each node
            auto& node2level = ws.node2level;
            node2level.resize(num_node);
            const float mult_l = 1.0 / log(1.0 * this->maxM);  // m_l in Sec 4.1 of the HNSW paper
            random_number_generator<> rng;
            for (index_type node_id = 0; node_id < num_node; node_id++) {
                // line 4 of Algorithm 1 in HNSW paper
                node2level[node_id] = (index_type)(-log(rng.uniform(0.0, 1.0)) * mult_l);
                // if max_level_upper_bound is given, we cap the the level
                if (max_level_upper_bound >= 0) {
                    node2level[node_id] = std::min<index_type>(node2level[node_id], (index_type)max_level_upper_bound);
                }
            }

            max_level_upper_bound = *std::max_element(node2level.begin(), node2level.end());

            graph_l0.init(X_trn, this->maxM0);
            graph_l1.init(X_trn, this->maxM, max_level_upper_bound);

            this->max_level = 0;
            this->init_node = 0;

            bool lock_free = (threads == 1);
#pragma omp parallel for schedule(dynamic, 1)
            for (index_type node_id = 0; node_id < num_node; node_id++) {
                int thread_id = omp_get_thread_num();
                add_point(node_id, ws, thread_id, lock_free);
            }

            auto sort_neighbors_for_node = [&](index_type node_id, workspace_t& ws, int thread_id) {
                auto& hnsw = ws.hnsw;
                auto& graph_l0 = hnsw.graph_l0;
                auto& graph_l1 = hnsw.graph_l1;
                auto& queue = ws.searchers[thread_id].cand_queue;

                const auto &src = graph_l0.get_node_feat(node_id);
                for (index_type level = 0; level <= ws.node2level[node_id]; level++) {
                    GraphBase *G;
                    if (level == 0) {
                        G = &graph_l0;
                    } else {
                        G = &graph_l1;
                    }
                    auto neighbors = G->get_neighborhood(node_id, level);
                    if (neighbors.degree() == 0) {
                        return;
                    }
                    queue.clear();
                    for (index_type j = 0; j < neighbors.degree(); j++) {
                        const auto& dst = graph_l0.get_node_feat(neighbors[j]);
                        queue.emplace_back(feat_vec_t::distance(src, dst), neighbors[j]);
                    }
                    std::sort(queue.begin(), queue.end());
                    for (index_type j = 0; j < neighbors.degree(); j++) {
                        neighbors[j] = queue[j].node_id;
                    }
                }
            };

#pragma omp parallel for schedule(dynamic, 1)
            for (index_type node_id = 0; node_id < num_node; node_id++) {
                int thread_id = omp_get_thread_num();
                sort_neighbors_for_node(node_id, ws, thread_id);
            }
        }

        // Algorithm 2 of HNSW paper
        template<bool lock_free=true>
        max_heap_t& search_level(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            searcher.reset();
            max_heap_t& topk_queue = searcher.topk_queue;
            min_heap_t& cand_queue = searcher.cand_queue;

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                graph_l0.get_node_feat(init_node));
            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(topk_ub_dist, init_node);
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
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.node_id;
                std::unique_lock<std::mutex>* lock_node = nullptr;
                if (!lock_free) {
                    lock_node = new std::unique_lock<std::mutex>(mtx_nodes->at(cand_node));
                }
                // visiting neighbors of candidate node
                const auto neighbors = G->get_neighborhood(cand_node, level);
                if (neighbors.degree() != 0) {
                    graph_l0.prefetch_node_feat(neighbors[0]);
                    index_type max_j = neighbors.degree() - 1;
                    for (index_type j = 0; j <= max_j; j++) {
                        graph_l0.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
                        auto next_node = neighbors[j];
                        if (!searcher.is_visited(next_node)) {
                            searcher.mark_visited(next_node);
                            dist_t next_lb_dist;
                            next_lb_dist = feat_vec_t::distance(
                                query,
                                graph_l0.get_node_feat(next_node)
                            );
                            if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                                cand_queue.emplace(next_lb_dist, next_node);
                                graph_l0.prefetch_node_feat(cand_queue.top().node_id);
                                topk_queue.emplace(next_lb_dist, next_node);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }
                }

                if (!lock_free) {
                    delete lock_node;
                }
            }
            return topk_queue;
        }

        // Algorithm 5 of HNSW paper, thread-safe inference
        max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, Searcher& searcher) const {
            index_type curr_node = this->init_node;
            auto &G1 = graph_l1;
            auto &G0 = graph_l0;
            // specialized search_level for level l=1,...,L because its faster for efS=1
            dist_t curr_dist = feat_vec_t::distance(
                query,
                G0.get_node_feat(init_node)
            );
            for (index_type curr_level = this->max_level; curr_level >= 1; curr_level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto neighbors = G1.get_neighborhood(curr_node, curr_level);
                    if (neighbors.degree() != 0) {
                        graph_l0.prefetch_node_feat(neighbors[0]);
                        index_type max_j = neighbors.degree() - 1;
                        for (index_type j = 0; j <= max_j; j++) {
                            graph_l0.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
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
            }
            // generalized search_level for level=0 for efS >= 1
            searcher.search_level(query, curr_node, std::max(efS, topk), 0);
            auto& topk_queue = searcher.topk_queue;
            if (topk < efS) {
                // remove extra when efS > topk
                while (topk_queue.size() > topk) {
                    topk_queue.pop();
                }
            }
            std::sort_heap(topk_queue.begin(), topk_queue.end());
            return topk_queue;
        }
    };


    template<typename dist_t, class FeatVec_T>
    struct HNSWProductQuantizer4Bits {
        typedef FeatVec_T feat_vec_t;
        typedef Pair<dist_t, index_type> pair_t;
        typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
        typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;


        // scalar variables
        index_type num_node;
        index_type maxM;   // max number of out-degree for level l=1,...,L
        index_type maxM0;  // max number of out-degree for level l=0
        index_type efC;    // size of priority queue for construction time
        index_type max_level;
        index_type init_node;
        index_type subspace_dimension;  // dimension of each subspace in Product Quantization
        index_type sub_sample_points;   // number of sub-sampled points used to build quantizer subspace centors. 

        GraphL0<feat_vec_t> feature_vec;           // feature vectors only
        GraphL1 graph_l1;                       // neighborhood graphs from level 1 and above
        GraphProductQuantizer4Bits<feat_vec_t> graph_l0_pq4;   // Productquantized4Bits neighborhood graph built from graph_l0
        HNSWProductQuantizer4Bits() {
            std::string space_type = pecos::type_util::full_name<feat_vec_t>();
            if (space_type != "pecos::ann::FeatVecDenseL2Simd<float>") {
                throw std::runtime_error("Currently, we only support L2 distance with float type.");
            } 
        }
        ~HNSWProductQuantizer4Bits() {}
        struct Searcher : SetOfVistedNodes<unsigned short int> {
            typedef SetOfVistedNodes<unsigned short int> set_of_visited_nodes_t;
            typedef HNSWProductQuantizer4Bits<dist_t, FeatVec_T> hnswpq4_t;
            typedef heap_t<pair_t, std::less<pair_t>> max_heap_t;
            typedef heap_t<pair_t, std::greater<pair_t>> min_heap_t;

            const hnswpq4_t* hnsw;
            max_heap_t topk_queue;
            min_heap_t cand_queue;
            alignas(64) std::vector<uint8_t> lut;
            alignas(64) std::vector<float> appx_dist;
            float scale;
            float bias;

            Searcher(const hnswpq4_t* _hnsw=nullptr):
                SetOfVistedNodes<unsigned short int>(_hnsw? _hnsw->num_node : 0),
                hnsw(_hnsw)
            {}

            void reset() {
                set_of_visited_nodes_t::reset();
                topk_queue.clear();
                cand_queue.clear();
            }

            void prepare_inference() {
                auto num_of_local_centroids = hnsw->graph_l0_pq4.quantizer.num_of_local_centroids;
                auto max_degree = hnsw->graph_l0_pq4.max_degree;
                auto num_local_codebooks = hnsw->graph_l0_pq4.quantizer.num_local_codebooks;

                //  When using AVX512f, we have 16 centroids per local codebook, and each of it uses 8 bits to represent quantized
                //  distance value. Thus,m we will have 128 bits to load 1 set of local codebooks. Thus, a loadu_si512 will load
                //  512 / 128 == 4 local codebooks at a time. Thus, the lookup table size needs to be adjusted (padding 0) if
                //  if num_local_codebooks is not divisible by 4.
                size_t adjusted_num_local_codebooks = num_local_codebooks % 4 == 0 ? num_local_codebooks : (num_local_codebooks / 4 + 1) * 4;

                // Similarly, we have to parse every 16 neighbors at a time to maximally leverage avx512f.
                // Thus, we have to prepare result array which is multiple of 16 to make sure the SIMD
                // will not touch unavailable memory
                size_t adjusted_max_degree = max_degree % 16 == 0 ? max_degree : ((max_degree / 16) + 1) * 16;

                lut.resize(num_of_local_centroids * adjusted_num_local_codebooks, 0);
                appx_dist.resize(adjusted_max_degree, 0);
            }
            void setup_lut(float* query) {
                hnsw->graph_l0_pq4.quantizer.setup_lut(query, lut.data(), scale, bias);
            }
            void approximate_distance(size_t neighbor_size, const char* neighbor_codes) {
                // pass searcher to group_distance
                hnsw->graph_l0_pq4.quantizer.approximate_neighbor_group_distance(neighbor_size, appx_dist.data(), neighbor_codes, lut.data(), scale, bias);
            }

            max_heap_t& search_level(const feat_vec_t& query, index_type init_node, index_type efS, index_type level) {
                return hnsw->search_level(query, init_node, efS, level, *this);
            }

            max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, index_type num_rerank) {
                return hnsw->predict_single(query, efS, topk, *this, num_rerank);
            }
        };

        Searcher create_searcher() const {
            return Searcher(this);
        }


        static nlohmann::json load_config(const std::string& filepath) {
            std::ifstream loadfile(filepath);
            std::string json_str;
            if (loadfile.is_open()) {
                json_str.assign(
                    std::istreambuf_iterator<char>(loadfile),
                    std::istreambuf_iterator<char>());
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            auto j_param = nlohmann::json::parse(json_str);
            std::string hnsw_t_cur = pecos::type_util::full_name<HNSWProductQuantizer4Bits>();
            std::string hnsw_t_inp = j_param["hnsw_t"];
            if (hnsw_t_cur != hnsw_t_inp) {
                throw std::invalid_argument("Inconsistent HNSW_T: hnsw_t_cur = " + hnsw_t_cur  + " hnsw_t_cur = " + hnsw_t_inp);
            }
            return j_param;
        }

        void save_config(const std::string& filepath) const {
            nlohmann::json j_params = {
                {"hnsw_t", pecos::type_util::full_name<HNSWProductQuantizer4Bits>()},
                {"version", "v1.0"},
                {"train_params", {
                    {"num_node", this->num_node},
                    {"subspace_dimension", this->subspace_dimension},
                    {"sub_sample_points", this->sub_sample_points},
                    {"maxM", this->maxM},
                    {"maxM0", this->maxM0},
                    {"efC", this->efC},
                    {"max_level", this->max_level},
                    {"init_node", this->init_node}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if (savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void save(const std::string& model_dir) const {
            if (mkdir(model_dir.c_str(), 0777) == -1) {
                if (errno != EEXIST) {
                    throw std::runtime_error("Unable to create save folder at " + model_dir);
                }
            }
            save_config(model_dir + "/config.json");
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "wb");
            pecos::file_util::fput_multiple<index_type>(&num_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&maxM0, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&efC, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&max_level, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&init_node, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&subspace_dimension, 1, fp);
            pecos::file_util::fput_multiple<index_type>(&sub_sample_points, 1, fp);
            feature_vec.save(fp);
            graph_l1.save(fp);
            graph_l0_pq4.save(fp);
            fclose(fp);
        }

        void load(const std::string& model_dir) {
            auto config = load_config(model_dir + "/config.json");
            std::string version = config.find("version") != config.end() ? config["version"] : "not found";
            std::string index_path = model_dir + "/index.bin";
            FILE *fp = fopen(index_path.c_str(), "rb");
            if (version == "v1.0") {
                pecos::file_util::fget_multiple<index_type>(&num_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&maxM0, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&efC, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&max_level, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&init_node, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&subspace_dimension, 1, fp);
                pecos::file_util::fget_multiple<index_type>(&sub_sample_points, 1, fp);
                feature_vec.load(fp);
                graph_l1.load(fp);
                graph_l0_pq4.load(fp);
            } else {
                throw std::runtime_error("Unable to load this binary with version = " + version);
            }
            fclose(fp);
        }

        template<class MAT_T>
        void train(
            const MAT_T &X_trn,
            index_type M,
            index_type efC,
            index_type subspace_dimension=0,
            index_type sub_sample_points=0,
            int threads=1,
            int max_level_upper_bound=-1
        ) {
            HNSW<dist_t, feat_vec_t>* hnsw = new HNSW<dist_t, feat_vec_t>();
            hnsw->train(X_trn, M, efC, threads, max_level_upper_bound);
            this->num_node = hnsw->num_node;
            this->maxM = hnsw->maxM;
            this->maxM0 = hnsw->maxM0;
            this->efC = hnsw->efC;
            this->max_level = hnsw->max_level;
            this->init_node = hnsw->init_node;
            this->subspace_dimension = subspace_dimension;
            this->sub_sample_points = sub_sample_points;

            graph_l1.num_node = hnsw->graph_l1.num_node;
            graph_l1.max_level = hnsw->graph_l1.max_level;
            graph_l1.max_degree = hnsw->graph_l1.max_degree;
            graph_l1.node_mem_size = hnsw->graph_l1.node_mem_size;
            graph_l1.level_mem_size = hnsw->graph_l1.level_mem_size;
            graph_l1.buffer.resize(hnsw->graph_l1.buffer.size());
            memcpy(graph_l1.buffer.data(), hnsw->graph_l1.buffer.data(), hnsw->graph_l1.buffer.size() * sizeof(index_type));

            graph_l0_pq4.build_quantizer(X_trn, subspace_dimension, sub_sample_points);
            graph_l0_pq4.build_graph(hnsw->graph_l0);
            delete hnsw;
            feature_vec.init(X_trn, -1);
        }


        max_heap_t& predict_single(const feat_vec_t& query, index_type efS, index_type topk, Searcher& searcher, index_type num_rerank) const {
            index_type curr_node = this->init_node;
            auto &G1 = graph_l1;
            auto &G0 = feature_vec;
            // specialized search_level for level l=1,...,L because its faster for efS=1
            dist_t curr_dist = feat_vec_t::distance(
                query,
                G0.get_node_feat(init_node)
            );
            for (index_type curr_level = this->max_level; curr_level >= 1; curr_level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    const auto neighbors = G1.get_neighborhood(curr_node, curr_level);
                    if (neighbors.degree() != 0) {
                        feature_vec.prefetch_node_feat(neighbors[0]);
                        index_type max_j = neighbors.degree() - 1;
                        for (index_type j = 0; j <= max_j; j++) {
                            feature_vec.prefetch_node_feat(neighbors[std::min(j + 1, max_j)]);
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
            }
            // generalized search_level for level=0 for efS >= 1
            searcher.search_level(query, curr_node, std::max(efS, topk), 0);
            auto& topk_queue = searcher.topk_queue;


            if (num_rerank > 0) {
                index_type t_size = topk_queue.size() > num_rerank ? topk_queue.size() - num_rerank : 0;
                for (index_type i = 0; i < t_size; i++) {
                    topk_queue.pop();
                }
                for (auto i = topk_queue.begin(); i != topk_queue.end(); ++i) {
                    feature_vec.prefetch_node_feat((*(i + 1)).node_id);
                    pair_t cand_pair = (*i);
                    dist_t next_dist = feat_vec_t::distance(
                        query,
                        G0.get_node_feat(cand_pair.node_id)
                    );
                    (*i).dist = next_dist;
                }
                std::sort(topk_queue.begin(), topk_queue.end());
                if (topk_queue.size() > topk) {
                    topk_queue.resize(topk);
                }
                return searcher.topk_queue;
            }



            if (topk < efS) {
                // remove extra when efS > topk
                while (topk_queue.size() > topk) {
                    topk_queue.pop();
                }
            }
            std::sort_heap(topk_queue.begin(), topk_queue.end());
            return topk_queue;
        }

        max_heap_t& search_level(
            const feat_vec_t& query,
            index_type init_node,
            index_type efS,
            index_type level,
            Searcher& searcher,
            std::vector<std::mutex>* mtx_nodes=nullptr
        ) const {
            const auto *G0Q = &graph_l0_pq4;
            searcher.reset();
            searcher.setup_lut(query.val);
            max_heap_t& topk_queue = searcher.topk_queue;
            min_heap_t& cand_queue = searcher.cand_queue;

            dist_t topk_ub_dist = feat_vec_t::distance(
                query,
                feature_vec.get_node_feat(init_node)
            );
            topk_queue.emplace(topk_ub_dist, init_node);
            cand_queue.emplace(topk_ub_dist, init_node);
            searcher.mark_visited(init_node);

            while (!cand_queue.empty()) {
                pair_t cand_pair = cand_queue.top();
                if (cand_pair.dist > topk_ub_dist) {
                    break;
                }
                cand_queue.pop();

                index_type cand_node = cand_pair.node_id;

                // visiting neighbors of candidate node
                const auto neighbors = G0Q->get_neighborhood(cand_node, level);
                if (neighbors.degree() != 0) {
                    index_type max_j = neighbors.degree() - 1;

                    searcher.approximate_distance(max_j + 1, G0Q->get_neighbor_codes(cand_node));
                    for (index_type j = 0; j <= max_j; j++) {
                        auto next_node = neighbors[j];
                        dist_t next_lb_dist = searcher.appx_dist[j];

                        if (topk_queue.size() < efS || next_lb_dist < topk_ub_dist) {
                            if (!searcher.is_visited(next_node)) {
                                searcher.mark_visited(next_node);
                                cand_queue.emplace(next_lb_dist, next_node);
                                topk_queue.emplace(next_lb_dist, next_node);
                                if (topk_queue.size() > efS) {
                                    topk_queue.pop();
                                }
                                if (!topk_queue.empty()) {
                                    topk_ub_dist = topk_queue.top().dist;
                                }
                            }
                        }
                    }
                }

            }



            return topk_queue;
        }
    };
}  // end of namespace ann
}  // end of namespace pecos
