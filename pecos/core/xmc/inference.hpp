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

/*
* File: inference.hpp
*
* Description: Provides functionality for performing PECOS prediction.
*
* Note about memory management: Any function that returns a matrix type has allocated memory
* and it is incumbent upon the user to deallocate that memory by calling the free_underlying_memory
* method of the matrix in question.
*/

#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>
#include <vector>
#include "utils/matrix.hpp"
#include "utils/mmap_util.hpp"
#include "third_party/nlohmann_json/json.hpp"
#include "third_party/robin_hood_hashing/robin_hood.h"

#define DEFAULT_LAYER_TYPE LAYER_TYPE_BINARY_SEARCH_CHUNKED


namespace pecos {

    using robin_hood::unordered_set;
    using robin_hood::unordered_map;

    typedef ScipySparseNpz<true, float> ScipyCsrF32Npz;
    typedef ScipySparseNpz<false, float> ScipyCscF32Npz;

    enum layer_type_t {
        LAYER_TYPE_CSC,
        LAYER_TYPE_HASH_CHUNKED,
        LAYER_TYPE_BINARY_SEARCH_CHUNKED
    };

    struct HierarchicalMLModelMetadata {
        int depth;
        bool is_mmap=false;

        HierarchicalMLModelMetadata(const int d, const bool is_mmap=false) : depth(d), is_mmap(is_mmap) {}

        HierarchicalMLModelMetadata(const std::string& params_filepath) {
            std::ifstream ifs(params_filepath);
            if (!ifs.is_open()) {
                throw std::runtime_error("could not open " + params_filepath);
            }
            nlohmann::json j;
            ifs >> j;
            ifs.close();

            std::string model_type = j.value("model", "None");
            if (!(model_type == "HierarchicalMLModel")) {
                throw std::runtime_error(model_type + " loading is not implemented");
            }

            depth = j.value("depth", -1);
            if (depth <= 0) {
                throw std::runtime_error("model corrupted, depth is 0 or negative");
            }
            is_mmap = j.value("is_mmap", false);
        }

        void dump_json(const std::string& params_filepath) const {
            std::ofstream ofs(params_filepath);
            if (!ofs.is_open()) {
                throw std::runtime_error("could not open " + params_filepath);
            }

            ofs << "{\n";
            ofs << "\"model\": \"HierarchicalMLModel\",\n";
            ofs << "\"depth\": " << depth << ",\n";
            ofs << "\"is_mmap\": " << (is_mmap?"true":"false") << "\n";
            ofs << "}\n";

            ofs.close();
        }
    };

    struct MLModelMetadata {
        float bias;
        int only_topk;
        std::string post_processor;

        MLModelMetadata(
            float bias=1.0,
            int only_topk=10,
            std::string post_processor="l3-hinge"
        ) {
            this->bias = bias;
            this->only_topk = only_topk;
            this->post_processor = post_processor;
        }

        MLModelMetadata(const std::string& params_filepath) {
            std::ifstream ifs(params_filepath);
            if (!ifs.is_open()) {
                throw std::runtime_error("could not open " + params_filepath);
            }
            nlohmann::json j;
            ifs >> j;
            ifs.close();

            std::string model_type = j.value("model", "None");
            if (!(model_type == "MLModel")) {
                throw std::runtime_error(model_type + " loading is not implemented");
            }

            if (!j.contains("bias")) {
                 throw std::runtime_error("model corrupted, does not contain bias");
            }

            bias = j["bias"];

            if (!j.contains("pred_kwargs")) {
                 throw std::runtime_error("model corrupted, does not contain pred_kwargs");
            }

            auto& pred_kwargs = j["pred_kwargs"];

             if (!pred_kwargs.contains("only_topk")) {
                 throw std::runtime_error("model corrupted, does not contain only_topk in pred_kwargs");
            }

            if (!pred_kwargs.contains("post_processor")) {
                 throw std::runtime_error("model corrupted, does not contain post_processor in pred_kwargs");
            }


            only_topk = pred_kwargs["only_topk"];
            post_processor = pred_kwargs["post_processor"];
        }

        void dump_json(const std::string& params_filepath) const {
            std::ofstream ofs(params_filepath);
            if (!ofs.is_open()) {
                throw std::runtime_error("could not open " + params_filepath);
            }

            ofs << "{\n";
            ofs << "\"model\": \"MLModel\",\n";
            ofs << "\"bias\": " << bias << ",\n";
            ofs << "\"pred_kwargs\": {\n";
            ofs << "\t\"only_topk\": " << only_topk << ",\n";
            ofs << "\t\"post_processor\": \"" << post_processor << "\"\n";
            ofs << "\t}\n";
            ofs << "}\n";

            ofs.close();
        }
    };

    template <class T>
    struct PostProcessor {
        typedef std::function<T(const T&)> Transform;
        typedef std::function<T(const T&, const T&)> Combiner;

        Transform transform;
        Combiner combiner;

        PostProcessor(
            const Transform& transform = [](const T& v) { return v; },
            const Combiner& combiner = [](const T& x, const T& y) { return x; }
        ) : transform(transform), combiner(combiner) {}

        static PostProcessor get(const std::string& name) {
            auto startswith = [](const std::string& full, const std::string& pattern) -> bool {
                return full.size() >= pattern.size() && full.compare(0, pattern.size(), pattern) == 0;
            };

            auto endswith = [](const std::string& full, const std::string& pattern) -> bool {
                return full.size() >= pattern.size() &&
                       full.compare(full.size() - pattern.size(), pattern.size(), pattern) == 0;
            };
            const std::string log_prefix("log-");
            static unordered_map<std::string, PostProcessor<T>> post_processors;

            if (post_processors.find(name) != post_processors.end()) {
                return post_processors[name];
            }

            if (name == "noop") {
                post_processors[name] = PostProcessor<T>([](const T& v) -> T { return v; },
                                                         [](const T& x, const T& y) -> T { return x; });
                return post_processors[name];
            } else if (name == "sigmoid") {
                post_processors[name] = PostProcessor<T>(
                    [](const T& v) -> T { return 1.0 / (1.0 + std::exp(-v)); }, std::multiplies<T>());
            } else if (name == "log-sigmoid") {
                post_processors[name] = PostProcessor<T>(
                    [](const T& v) -> T { return -std::log(1.0 + std::exp(-v)); }, std::plus<T>());
            } else if (startswith(name, "log-l") && endswith(name, "-hinge")) {
                const std::string prefix("log-l");
                const std::string suffix("-hinge");
                const size_t p = std::atoi(
                    name.substr(prefix.size(), name.size() - prefix.size() - suffix.size()).c_str());
                post_processors[name] = PostProcessor<T>{[=](const T& v) -> T {
                                                             T z = std::max(0.0, 1.0 - v);
                                                             return -std::pow(z, p);
                                                         },
                                                         std::plus<T>()};
            } else if (startswith(name, "l") && endswith(name, "-hinge")) {
                const std::string prefix("l");
                const std::string suffix("-hinge");
                const size_t p = std::atoi(
                    name.substr(prefix.size(), name.size() - prefix.size() - suffix.size()).c_str());
                post_processors[name] = PostProcessor<T>{[=](const T& v) -> T {
                                                             T z = std::max(0.0, 1.0 - v);
                                                             return std::exp(-std::pow(z, p));
                                                         },
                                                         std::multiplies<T>()};
            }
            return post_processors[name];
        }
    };

    // A structure that holds a single nonzero entry in a chunked matrix
    struct chunk_entry_t {
        typedef typename csc_t::index_type index_type;
        typedef typename csc_t::value_type value_type;

        // The column offset of this entry in the chunk. (i.e., the column of
        // this entry is given by the col_offset plus the starting column of
        // the chunk this entry resides in.)
        index_type col_offset;
        // The value of this nonzero entry
        value_type val;
    };

    // View of a hash chunk, does not hold any memory
    struct hash_chunk_view_t {
        typedef typename csc_t::index_type index_type;
        typedef typename csc_t::mem_index_type mem_index_type;
        typedef typename chunk_entry_t::value_type value_type;

        index_type col_begin; // The column this chunk starts at (inclusive)
        index_type col_end; // The column this chunk ends at (exclusive)
        index_type nnz_rows; // The number of non-zero rows in this chunk
        // Using index_type instead of bool for struct padding, the value is still boolean
        index_type b_has_explicit_bias; // Whether or not this chunk has an explicit bias term

        // This struct does not hold memory, all pointers are only views
        unordered_map<index_type, index_type> row_hash; // Maps a matrix row index into an index of the row_ptr array below
        mem_index_type* row_ptr; // An array of where rows begin in hash_chunked_matrix_t::entries


        hash_chunk_view_t() :
            col_begin(0),
            col_end(0),
            nnz_rows(0),
            b_has_explicit_bias(false),
            row_ptr(nullptr) {
        }

        void set_row(index_type row, index_type i_row, mem_index_type i_entry) {
            row_ptr[i_row] = i_entry;
            row_hash[row] = i_row;
        }

        index_type row_ptr_size() {
            return nnz_rows == 0 ? (0) : (nnz_rows + 1);
        }
    };

    // View of a binary search chunk, does not hold any memory
    struct bin_search_chunk_view_t {
        typedef typename csc_t::index_type index_type;
        typedef typename csc_t::mem_index_type mem_index_type;
        typedef typename chunk_entry_t::value_type value_type;

        index_type col_begin; // The column this chunk starts at (inclusive)
        index_type col_end; // The column this chunk ends at (exclusive)
        index_type nnz_rows; // The number of non-zero rows in this chunk
        // Using index_type instead of bool for struct padding, the value is still boolean
        index_type b_has_explicit_bias; // Whether or not this chunk has an explicit bias term, 0=false

        // This struct does not hold memory, all pointers are only views
        index_type* row_idx; // Stores the row id of each nnz row, size nnz_rows
        mem_index_type* row_ptr; // Stores the pointer to data of each nnz row, size nnz_rows+1


        bin_search_chunk_view_t() :
            col_begin(0),
            col_end(0),
            nnz_rows(0),
            b_has_explicit_bias(false),
            row_idx(nullptr),
            row_ptr(nullptr) {
        }

        void set_row(index_type row, index_type i_row, mem_index_type i_entry) {
            row_ptr[i_row] = i_entry;
            row_idx[i_row] = row;
        }

        index_type row_idx_size() {
            return nnz_rows;
        }

        index_type row_ptr_size() {
            return nnz_rows == 0 ? (0) : (nnz_rows + 1);
        }
    };

    struct hash_chunked_matrix_t {
        typedef hash_chunk_view_t chunk_t;
        typedef typename chunk_t::index_type index_type;
        typedef typename chunk_t::mem_index_type mem_index_type;
        typedef typename chunk_t::value_type value_type;
        typedef uint32_t chunk_index_type;

        static const layer_type_t layer_type = LAYER_TYPE_HASH_CHUNKED;

        chunk_t* chunks; // The chunks of this matrix
        chunk_entry_t* entries; // The nz entries of this matrix
        index_type chunk_count;
        index_type cols;
        index_type rows;

        // actual memory storage
        std::vector<chunk_t> chunks_;
        std::vector<mem_index_type> chunks_row_ptr_;
        std::vector<chunk_entry_t> entries_;

        // NOTE: Only use this function when metadata is assigned and chunks allocated
        void allocate_chunks_row_ptrs_(const std::vector<index_type> &chunk_nnz_rows) {
            mem_index_type chunks_row_ptr_size = 0;
            for (chunk_index_type i=0; i < chunk_count; ++i) {
                auto& chunk = chunks[i];
                if (chunk.nnz_rows != 0) {
                    chunks_row_ptr_size += chunk.row_ptr_size();
                }
            }
            chunks_row_ptr_.resize(chunks_row_ptr_size);
            mem_index_type* tmp_row_ptr_ptr = chunks_row_ptr_.data();
            for (chunk_index_type i=0; i < chunk_count; ++i) {
                auto& chunk = chunks[i];
                if (chunk.nnz_rows != 0) {
                    chunk.row_ptr = tmp_row_ptr_ptr;
                    tmp_row_ptr_ptr += chunk.row_ptr_size();
                }
            }
        }

        mem_index_type get_nnz() const {
            auto& lastChunk = chunks[chunk_count - 1];
            return lastChunk.row_ptr[lastChunk.row_hash.size()];
        }

        bool check_bias_explicit(const chunk_t& chunk) const {
            return chunk.row_hash.find(rows - 1) != chunk.row_hash.end();
        }

        void save_mmap(const std::string& file_name) const {
            throw std::runtime_error("Not implemented yet.");
        }

        void load_mmap(const std::string& file_name, const bool lazy_load) {
            throw std::runtime_error("Not implemented yet.");
        }
    };

    struct bin_search_chunked_matrix_t {
        typedef bin_search_chunk_view_t chunk_t;
        typedef typename chunk_t::index_type index_type;
        typedef typename chunk_t::mem_index_type mem_index_type;
        typedef typename chunk_t::value_type value_type;
        typedef uint32_t chunk_index_type;

        static const layer_type_t layer_type = LAYER_TYPE_BINARY_SEARCH_CHUNKED;

        chunk_t* chunks; // The chunks of this matrix
        chunk_entry_t* entries; // The nz entries of this matrix
        index_type chunk_count;
        index_type cols;
        index_type rows;

        // actual memory storage
        mmap_util::MmapableVector<chunk_t> chunks_;
        mmap_util::MmapableVector<index_type> chunks_row_idx_;
        mmap_util::MmapableVector<mem_index_type> chunks_row_ptr_;
        mmap_util::MmapableVector<chunk_entry_t> entries_;

        // mmap
        mmap_util::MmapStore mmap_store;

        void save_to_mmap_store(mmap_util::MmapStore& mmap_s) const {
            // scalars
            mmap_s.fput_one<index_type>(chunk_count);
            mmap_s.fput_one<index_type>(rows);
            mmap_s.fput_one<index_type>(cols);
            // arrays
            chunks_.save_to_mmap_store(mmap_s);
            chunks_row_idx_.save_to_mmap_store(mmap_s);
            chunks_row_ptr_.save_to_mmap_store(mmap_s);
            entries_.save_to_mmap_store(mmap_s);
        }

        void load_from_mmap_store(mmap_util::MmapStore& mmap_s) {
           // scalars
            chunk_count = mmap_s.fget_one<index_type>();
            rows = mmap_s.fget_one<index_type>();
            cols = mmap_s.fget_one<index_type>();
            // arrays
            chunks_.load_from_mmap_store(mmap_s);
            chunks_row_idx_.load_from_mmap_store(mmap_s);
            chunks_row_ptr_.load_from_mmap_store(mmap_s);
            entries_.load_from_mmap_store(mmap_s);

            // post-processing
            // Re-assign chunks view pointers
            chunks_.to_self_alloc_vec(); // Convert to memory vector for assigning
            auto chunks_row_idx_data = chunks_row_idx_.data();
            auto chunks_row_ptr_data = chunks_row_ptr_.data();
            for (index_type i=0; i < chunk_count; ++i) {
                auto& chunk = chunks_[i];
                if (chunk.nnz_rows != 0) {
                    chunk.row_idx = chunks_row_idx_data;
                    chunks_row_idx_data += chunk.row_idx_size();
                    chunk.row_ptr = chunks_row_ptr_data;
                    chunks_row_ptr_data += chunk.row_ptr_size();
                } else {
                    chunk.row_idx = nullptr;
                    chunk.row_ptr = nullptr;
                }
            }
            chunks = chunks_.data();
            entries = entries_.data();
        }

        void save_mmap(const std::string& file_name) const {
            mmap_util::MmapStore mmap_s = mmap_util::MmapStore();
            mmap_s.open(file_name, "w");
            save_to_mmap_store(mmap_s);
            mmap_s.close();
        }

        void load_mmap(const std::string& file_name, const bool lazy_load) {
            mmap_store.open(file_name, lazy_load?"r_lazy":"r");
            load_from_mmap_store(mmap_store);
        }

        // NOTE: Only use this function when metadata is assigned and chunks allocated
        void allocate_chunks_row_ptrs_(const std::vector<index_type> &chunk_nnz_rows) {
            mem_index_type chunks_row_idx_size = 0;
            mem_index_type chunks_row_ptr_size = 0;
            for (chunk_index_type i=0; i < chunk_count; ++i) {
                auto& chunk = chunks[i];
                if (chunk.nnz_rows != 0) {
                    chunks_row_idx_size += chunk.row_idx_size();
                    chunks_row_ptr_size += chunk.row_ptr_size();
                }
            }
            chunks_row_idx_.resize(chunks_row_idx_size);
            chunks_row_ptr_.resize(chunks_row_ptr_size);
            index_type* tmp_row_idx_ptr = chunks_row_idx_.data();
            mem_index_type* tmp_row_ptr_ptr = chunks_row_ptr_.data();
            for (chunk_index_type i=0; i < chunk_count; ++i) {
                auto& chunk = chunks[i];
                if (chunk.nnz_rows != 0) {
                    chunk.row_idx = tmp_row_idx_ptr;
                    tmp_row_idx_ptr += chunk.row_idx_size();
                    chunk.row_ptr = tmp_row_ptr_ptr;
                    tmp_row_ptr_ptr += chunk.row_ptr_size();
                }
            }
        }

        uint64_t get_nnz() const {
            auto& lastChunk = chunks[chunk_count - 1];
            return lastChunk.row_ptr[lastChunk.nnz_rows];
        }

        bool check_bias_explicit(const chunk_t& chunk) const {
            return chunk.nnz_rows > 0 && chunk.row_idx[chunk.nnz_rows - 1] == rows - 1;
        }
    };

    // Adds a scalar multiple of a sparse row of a chunk to a dense output matrix block
    template <typename chunked_matrix_t>
    inline void add_scaled_chunk_row_to_output_block(const chunked_matrix_t& matrix,
        const typename chunked_matrix_t::chunk_t& chunk,
        const typename chunked_matrix_t::index_type nz_row_idx,
        const typename chunked_matrix_t::value_type scalar,
        typename chunked_matrix_t::value_type* output_block) {
        uint64_t row_start = chunk.row_ptr[nz_row_idx];
        uint64_t row_end = chunk.row_ptr[nz_row_idx + 1];
        for (uint64_t j = row_start; j < row_end; ++j) {
            auto& entry = matrix.entries[j];
            output_block[entry.col_offset] += scalar * entry.val;
        }
    }

    // Create a chunked matrix from a csc matrix. chunk_col_idx specifies the
    // column starts of each chunk.
    template <typename matrix_type_t>
    void allocate_chunked_matrix_(
        const uint32_t chunk_count,
        const csc_t::index_type cols,
        const csc_t::index_type rows,
        const csc_t::mem_index_type nnz,
        const csc_t::mem_index_type chunk_col_idx[],
        const std::vector<typename matrix_type_t::index_type>& chunk_nnz_rows,
        matrix_type_t& chunked_mat
    ) {
        // Assign metadata
        chunked_mat.chunk_count = chunk_count;
        chunked_mat.cols = cols;
        chunked_mat.rows = rows;

        // Allocate entries
        chunked_mat.entries_.resize(nnz);
        chunked_mat.entries = chunked_mat.entries_.data();

        // Allocate chunks
        chunked_mat.chunks_.resize(chunk_count);
        chunked_mat.chunks = chunked_mat.chunks_.data();

        // Assign chunks col idx
        for (uint32_t i=0; i < chunk_count; ++i) {
            auto& chunk = chunked_mat.chunks[i];
            chunk.col_begin = chunk_col_idx[i];
            chunk.col_end = chunk_col_idx[i + 1];
            chunk.nnz_rows = chunk_nnz_rows[i];
        }

        // Allocate chunks row ptrs
        chunked_mat.allocate_chunks_row_ptrs_(chunk_nnz_rows);
    }

    template <typename matrix_type_t>
    void make_chunked_from_csc(const csc_t& mat,
        const csc_t::mem_index_type chunk_col_idx[],
        const uint32_t chunk_count,
        matrix_type_t& chunked) {

        typedef typename matrix_type_t::index_type index_type;
        typedef typename matrix_type_t::mem_index_type mem_index_type;
        typedef typename matrix_type_t::chunk_index_type chunk_index_type;

        struct chunk_nz_entry_t {
            chunk_entry_t::index_type row;
            chunk_entry_t::index_type col;
            chunk_entry_t::value_type val;

            bool operator<(const chunk_nz_entry_t& other) const {
                return row < other.row;
            }
        };

        // Collect chunks cols ptr
        std::vector<mem_index_type> chunk_ptr(chunk_count + 1);
        for (chunk_index_type i = 0; i < chunk_count; ++i) {
            chunk_ptr[i] = mat.col_ptr[chunk_col_idx[i]];
        }
        chunk_ptr[chunk_count] = mat.col_ptr[mat.cols];

        // Collect chunks nnz_rows for allocating
        std::vector<index_type> chunk_nnz_rows(chunk_count);
        for (chunk_index_type i_chunk = 0; i_chunk < chunk_count; ++i_chunk) {
            auto chunk_nnz = chunk_ptr[i_chunk + 1] - chunk_ptr[i_chunk];
            // Empty chunk
            if (chunk_nnz == 0) {
                chunk_nnz_rows[i_chunk] = 0;
                continue;
            }
            // Count chunk's unique nonzero rows with set
            unordered_set<chunk_entry_t::index_type> nnz_rows;
            for (auto col = chunk_col_idx[i_chunk]; col < chunk_col_idx[i_chunk + 1]; ++col) {
                for (auto m_col = mat.col_ptr[col]; m_col < mat.col_ptr[col + 1]; ++m_col) {
                    nnz_rows.insert(mat.row_idx[m_col]);
                }
            }
            chunk_nnz_rows[i_chunk] = nnz_rows.size();
        }

        // Allocate
        allocate_chunked_matrix_<matrix_type_t>(
            chunk_count, mat.cols, mat.rows, mat.col_ptr[mat.cols],
            chunk_col_idx, chunk_nnz_rows, chunked);

        // Assign chunks row idx & ptrs value
        std::vector<chunk_nz_entry_t> nonzeros;
        for (chunk_index_type i_chunk = 0; i_chunk < chunk_count; ++i_chunk) {
            auto& chunk = chunked.chunks[i_chunk];
            mem_index_type chunk_nnz = chunk_ptr[i_chunk + 1] - chunk_ptr[i_chunk];

            // Ignore empty chunks
            if (chunk.nnz_rows == 0) {
                continue;
            }

            // Collect nonzeros
            nonzeros.resize(chunk_nnz);
            mem_index_type i_nz = 0;
            for (auto col = chunk.col_begin; col < chunk.col_end; ++col) {
                for (auto m_col = mat.col_ptr[col]; m_col < mat.col_ptr[col + 1]; ++m_col) {
                    auto& nz_entry = nonzeros[i_nz++];
                    nz_entry.row = mat.row_idx[m_col];
                    nz_entry.col = col;
                    nz_entry.val = mat.val[m_col];
                }
            }
            // Sort by row, remains sorted by columns
            std::stable_sort(nonzeros.begin(), nonzeros.end());

            // Assign all nonzero entries
            chunk.row_ptr[chunk.nnz_rows] = mat.col_ptr[chunk.col_end];
            index_type last_row = static_cast<index_type>(-1);
            index_type i_nz_row = 0;
            for (mem_index_type i_nz = 0, i_entry = chunk_ptr[i_chunk]; i_nz < nonzeros.size(); ++i_nz, ++i_entry) {
                auto& nz_entry = nonzeros[i_nz];
                // Assign the chunk's row
                // The row has changed, save the row's metadata into the chunk
                if (nz_entry.row != last_row) {
                    chunk.set_row(nz_entry.row, i_nz_row, i_entry);
                    ++i_nz_row;
                    last_row = nz_entry.row;
                }
                chunked.entries[i_entry].col_offset = nz_entry.col - chunk.col_begin;
                chunked.entries[i_entry].val = nz_entry.val;
            }
        }
    }

    // Checks if the rows of C (i.e., the children nodes) are contiguously ordered.
    // That is, all of the children of a node are contiguous in row space and these
    // contiguous groups of children are ordered by their respective parents.
    // Each row of C has at most one non-zero value.
    // In the case where the tree is pruned, i.e., some rows of C are empty, the function returns false.
    // This ordering is necessary to use chunked matrices.
    bool check_if_contiguously_ordered(const csc_t& C) {
        // When tree is pruned, return false
        if (C.get_nnz() < C.rows) {
            return false;
        }
        bool b_ordered = true;
        for (csc_t::mem_index_type i = 0; i < C.col_ptr[C.cols]; ++i) {
            b_ordered = b_ordered && C.row_idx[i] == i;
        }
        return b_ordered;
    }

    template <typename matrix_t>
    void make_chunked_W_from_layer_matrices(const csc_t& W, const csc_t& C,
        const bool b_use_bias, matrix_t& chunked_W) {

        typedef typename matrix_t::chunk_index_type chunk_index_t;

        // Make sure that the rows of C are contiguous in order of parent node
        make_chunked_from_csc<matrix_t>(W, C.col_ptr, C.cols, chunked_W);

        // Precompute whether each chunk actually has a bias term.
        if (b_use_bias) {
            for (chunk_index_t i_chunk = 0; i_chunk < chunked_W.chunk_count; ++i_chunk) {
                auto& chunk = chunked_W.chunks[i_chunk];
                chunk.b_has_explicit_bias = chunked_W.check_bias_explicit(chunk);
            }
        } else {
            for (chunk_index_t i_chunk = 0; i_chunk < chunked_W.chunk_count; ++i_chunk) {
                auto& chunk = chunked_W.chunks[i_chunk];
                chunk.b_has_explicit_bias = false;
            }
        }
    }

    // Defines templated operations for vector x chunk products
    template <typename query_vector_t, typename chunked_matrix_t>
    struct chunk_ops {
        // Sparse vector x chunk
        static void compute_chunk_inner_product_write_to_zeroed_block(
            const query_vector_t& v, const typename chunked_matrix_t::chunk_t& chunk,
            const chunked_matrix_t& chunk_matrix,
            typename chunked_matrix_t::value_type* output_block,
            typename chunked_matrix_t::value_type bias, bool b_use_bias);
    };

    template <>
    struct chunk_ops<typename csr_t::row_vec_t, hash_chunked_matrix_t> {
        // Please make sure that the memory in result_dest has already been zeroed!
        // Compute the inner product of a vector and chunk in hash format.
        // Inner product is computed via hash table lookup.
        static void compute_chunk_inner_product_write_to_zeroed_block(
            const csr_t::row_vec_t& v, const hash_chunk_view_t& chunk,
            const hash_chunked_matrix_t& chunk_matrix,
            typename hash_chunked_matrix_t::value_type* output_block,
            typename hash_chunked_matrix_t::value_type bias, bool b_use_bias) {

            // The chunk has a bias
            if (b_use_bias) {
                // Add bias to result
                add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                    chunk.row_hash.find(chunk_matrix.rows - 1)->second,
                    bias, output_block);
            }

            // Add everything else
            for (csr_t::row_vec_t::index_type i = 0; i < v.nnz; ++i) {
                auto v_val = v.val[i];
                auto it = chunk.row_hash.find(v.idx[i]);
                if (it != chunk.row_hash.end()) {
                    add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                        it->second, v_val, output_block);
                }
            }
        }
    };

    template <>
    struct chunk_ops<typename drm_t::row_vec_t, hash_chunked_matrix_t> {
        static void compute_chunk_inner_product_write_to_zeroed_block(
            const drm_t::row_vec_t& v, const hash_chunk_view_t& chunk,
            const hash_chunked_matrix_t& chunk_matrix,
            typename hash_chunked_matrix_t::value_type* output_block,
            typename hash_chunked_matrix_t::value_type bias, bool b_use_bias) {

            if (b_use_bias) {
                // Because the hash map is unordered, we need to check if every
                // entry is the bias term.
                // This is very slow, don't use if you can avoid it.
                for (auto it = chunk.row_hash.begin(); it != chunk.row_hash.end(); ++it) {
                    if (it->first == v.len) {
                        // The bias term
                        add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                            it->second, bias, output_block);
                    } else {
                        // Not the bias term
                        auto v_val = v.val[it->first];
                        add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                            it->second, v_val, output_block);
                    }
                }
            } else {
                for (auto it = chunk.row_hash.begin(); it != chunk.row_hash.end(); ++it) {
                    auto v_val = v.val[it->first];
                    add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                        it->second, v_val, output_block);
                }
            }
        }
    };

    template <>
    struct chunk_ops<typename csr_t::row_vec_t, bin_search_chunked_matrix_t> {
        // Please make sure that the memory in result_dest has already been zeroed!
        // Compute the inner product of a vector and chunk in binary search format.
        // Inner product is computed via binary search.
        static void compute_chunk_inner_product_write_to_zeroed_block(
            const csr_t::row_vec_t& v, const bin_search_chunk_view_t& chunk,
            const bin_search_chunked_matrix_t& chunk_matrix,
            typename bin_search_chunked_matrix_t::value_type* output_block,
            typename bin_search_chunked_matrix_t::value_type bias, bool b_use_bias) {

            typedef typename bin_search_chunked_matrix_t::index_type chunk_index_t;
            typedef typename csr_t::row_vec_t::index_type vec_index_t;

            chunk_index_t* chunk_idx_end = &chunk.row_idx[chunk.nnz_rows];
            vec_index_t* v_idx_end = &v.idx[v.nnz];
            chunk_index_t s = 0;
            vec_index_t t = 0;

            while (s < chunk.nnz_rows && t < v.nnz) {
                if (chunk.row_idx[s] == v.idx[t]) {
                    auto v_val = v.val[t];
                    add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                        s, v_val, output_block);
                    ++s;
                    ++t;
                } else if (chunk.row_idx[s] < v.idx[t]) {
                    // Perform a binary search on chunk.row_idx
                    s = std::lower_bound(&chunk.row_idx[s], chunk_idx_end, v.idx[t]) - chunk.row_idx;
                }
                else if (chunk.row_idx[s] > v.idx[t]) {
                    // Perform a binary search on v.idx
                    t = std::lower_bound(&v.idx[t], v_idx_end, chunk.row_idx[s]) - v.idx;
                }
            }

            // There is a bias
            if (b_use_bias) {
                // Add bias term if necessary
                auto last = chunk.nnz_rows - 1;
                add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                    last, bias, output_block);
            }
        }
    };

    template <>
    struct chunk_ops<typename drm_t::row_vec_t, bin_search_chunked_matrix_t> {
        static void compute_chunk_inner_product_write_to_zeroed_block(
            const typename drm_t::row_vec_t& v, const bin_search_chunk_view_t& chunk,
            const bin_search_chunked_matrix_t& chunk_matrix,
            typename bin_search_chunked_matrix_t::value_type* output_block,
            typename bin_search_chunked_matrix_t::value_type bias, bool b_use_bias) {

            uint32_t it_end = chunk.nnz_rows;
            if (b_use_bias) {
                // Add bias term
                add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                    it_end - 1, bias, output_block);
                // Exclude bias term from below
                --it_end;
            }

            // Iterate through all non-bias terms
            for (uint32_t it = 0; it != it_end; ++it) {
                auto v_val = v.val[chunk.row_idx[it]];
                add_scaled_chunk_row_to_output_block(chunk_matrix, chunk,
                    it, v_val, output_block);
            }
        }
    };

    template <layer_type_t type>
    struct LAYER_TYPE_METADATA_;

    template <>
    struct LAYER_TYPE_METADATA_<LAYER_TYPE_CSC> {
        typedef csc_t matrix_t;
    };

    template <>
    struct LAYER_TYPE_METADATA_<LAYER_TYPE_HASH_CHUNKED> {
        typedef hash_chunked_matrix_t matrix_t;
    };

    template <>
    struct LAYER_TYPE_METADATA_<LAYER_TYPE_BINARY_SEARCH_CHUNKED> {
        typedef bin_search_chunked_matrix_t matrix_t;
    };

    template <typename matrix_t>
    struct WEIGHT_MATRIX_METADATA_;

    template <typename matrix_t>
    struct QUERY_MATRIX_METADATA_;

    template <>
    struct QUERY_MATRIX_METADATA_<csr_t> {
        static constexpr const char* TYPE_NAME = "csr_t";
    };

    template <>
    struct QUERY_MATRIX_METADATA_<drm_t> {
        static constexpr const char* TYPE_NAME = "drm_t";
    };

    template <>
    struct WEIGHT_MATRIX_METADATA_<csc_t> {
        const static bool IS_CHUNKED = false;
        const static layer_type_t LAYER_TYPE = LAYER_TYPE_CSC;
        static constexpr const char* TYPE_NAME = "csc_t";
    };

    template <>
    struct WEIGHT_MATRIX_METADATA_<hash_chunked_matrix_t> {
        const static bool IS_CHUNKED = true;
        const static layer_type_t LAYER_TYPE = LAYER_TYPE_HASH_CHUNKED;
        static constexpr const char* TYPE_NAME = "hash_chunked_matrix_t";
    };

    template <>
    struct WEIGHT_MATRIX_METADATA_<bin_search_chunked_matrix_t> {
        const static bool IS_CHUNKED = true;
        const static layer_type_t LAYER_TYPE = LAYER_TYPE_BINARY_SEARCH_CHUNKED;
        static constexpr const char* TYPE_NAME = "bin_search_chunked_matrix_t";
    };

    template<typename matrix_t,
        bool chunked = WEIGHT_MATRIX_METADATA_<matrix_t>::IS_CHUNKED>
    struct w_ops;

    template<typename chunked_matrix_t>
    struct w_ops<chunked_matrix_t, true> {
        template <typename query_matrix_t, typename prediction_matrix_t>
        static void compute_sparse_predictions(const query_matrix_t& X, const chunked_matrix_t& W,
            typename csr_t::mem_index_type* row_ptr, // Sparsity pattern of prediction at current layer
            typename csr_t::index_type* col_idx,
            bool b_sort_by_chunk,
            float bias,
            const prediction_matrix_t& prev_layer_pred,
            prediction_matrix_t& curr_layer_pred);
    };

    template <>
    struct w_ops<csc_t, false> {
        template <typename query_matrix_t, typename prediction_matrix_t>
        static void compute_sparse_predictions(const query_matrix_t& X, const csc_t& W,
            typename csr_t::mem_index_type* row_ptr,
            typename csr_t::index_type* col_idx, // Sparsity pattern of prediction at current layer
            bool b_sort_by_chunk,
            float bias,
            const prediction_matrix_t& prev_layer_pred,
            prediction_matrix_t& curr_layer_pred);
    };

    // Compute the predictions of a layer (before post process) on the specified sparsity pattern
    template <typename chunked_matrix_t>
    template <typename query_matrix_t, typename prediction_matrix_t>
    void w_ops<chunked_matrix_t, true>::compute_sparse_predictions(const query_matrix_t& X,
        const chunked_matrix_t& W,
        typename csr_t::mem_index_type* row_ptr,
        typename csr_t::index_type* col_idx, // Sparsity pattern of prediction at current layer
        bool b_sort_by_chunk,
        float bias,
        const prediction_matrix_t& prev_layer_pred,
        prediction_matrix_t& curr_layer_pred) {

        struct compute_query_t {
            typename query_matrix_t::index_type row;
            typename chunked_matrix_t::chunk_index_type chunk;
            typename csr_t::mem_index_type write_addr;

            bool operator<(const compute_query_t& other) const {
                return chunk < other.chunk;
            }
        };

        typename prediction_matrix_t::mem_index_type* parent_row_ptr = prev_layer_pred.row_ptr;
        typename prediction_matrix_t::index_type* parent_col_idx = prev_layer_pred.col_idx;

        auto rows = X.rows;
        auto cols = W.cols;
        auto nnz = row_ptr[rows];
        auto parent_nnz = parent_row_ptr[rows];

        typedef typename query_matrix_t::row_vec_t query_row_t;
        typedef typename csr_t::index_type index_type;
        typedef typename csr_t::value_type value_type;
        typedef typename csr_t::mem_index_type mem_index_type;

        std::vector<compute_query_t> compute_queries(parent_nnz);

        curr_layer_pred.rows = rows;
        curr_layer_pred.cols = cols;
        curr_layer_pred.row_ptr = new mem_index_type[rows + 1];
        std::memcpy(curr_layer_pred.row_ptr, row_ptr, sizeof(mem_index_type) * (rows + 1));
        curr_layer_pred.col_idx = new index_type[nnz];
        std::memcpy(curr_layer_pred.col_idx, col_idx, sizeof(index_type) * nnz);
        curr_layer_pred.val = new value_type[nnz];

#pragma omp parallel for schedule(dynamic,4)
        for (index_type row = 0; row < rows; ++row) {
            mem_index_type row_start = row_ptr[row];
            mem_index_type row_end = row_ptr[row + 1];

            mem_index_type write_addr = row_start;
            mem_index_type parent_read_begin = parent_row_ptr[row];
            mem_index_type parent_read_end = parent_row_ptr[row + 1];
            for (mem_index_type parent_read_addr = parent_read_begin; parent_read_addr < parent_read_end;
                ++parent_read_addr) {
                compute_queries[parent_read_addr].row = row;
                compute_queries[parent_read_addr].chunk = parent_col_idx[parent_read_addr];
                compute_queries[parent_read_addr].write_addr = write_addr;
                auto& chunk = W.chunks[parent_col_idx[parent_read_addr]];
                write_addr += (chunk.col_end - chunk.col_begin);
            }

            // Zero out row
            std::fill(&curr_layer_pred.val[row_start], &curr_layer_pred.val[row_end], 0.0);
        }

        // Sort vector x chunk queries by chunk for better cache coherence if requested
        if (b_sort_by_chunk) {
            std::stable_sort(compute_queries.begin(), compute_queries.end());
        }

#pragma omp parallel for schedule(dynamic,64)
        for (mem_index_type i_query = 0; i_query < parent_nnz; ++i_query) {
            compute_query_t* query = &compute_queries[i_query];
            auto xi = X.get_row(query->row);
            auto& chunk = W.chunks[query->chunk];
            auto write_ptr = &curr_layer_pred.val[query->write_addr];
            auto b_use_bias = chunk.b_has_explicit_bias;
            chunk_ops<query_row_t, chunked_matrix_t>::
                compute_chunk_inner_product_write_to_zeroed_block(
                    xi, chunk, W, write_ptr, bias, b_use_bias);
        }

    }


    template <typename query_vec_t, typename weight_vec_t>
    struct vector_ops {
        static float inner_product(const query_vec_t& query,
            const weight_vec_t& weight,
            const typename weight_vec_t::index_type weight_dim,
            const typename weight_vec_t::value_type bias, bool b_use_bias);
    };

    template <>
    struct vector_ops<typename csr_t::row_vec_t, typename csc_t::col_vec_t> {
        static float inner_product(const typename csr_t::row_vec_t& query,
            const typename csc_t::col_vec_t& weight,
            typename csc_t::col_vec_t::index_type weight_dim,
            typename csc_t::col_vec_t::value_type bias, bool b_use_bias) {

            typedef typename csc_t::col_vec_t::value_type value_type;

            value_type res = 0.0;
            if (b_use_bias) {
                 // Is the bias term in the weight vector nonzero?
                if (weight.nnz > 0 && weight.idx[weight.nnz - 1] == weight_dim - 1) {
                    // Add bias to result
                    res += bias * weight.val[weight.nnz - 1];
                }
            }

            res += do_dot_product(query, weight);
            return res;
        }
    };

    template <>
    struct vector_ops<typename drm_t::row_vec_t, typename csc_t::col_vec_t> {
        static float inner_product(const typename drm_t::row_vec_t& query,
            const typename csc_t::col_vec_t& weight,
            typename csc_t::col_vec_t::index_type weight_dim,
            typename csc_t::col_vec_t::value_type bias, bool b_use_bias) {

            typedef typename csc_t::col_vec_t::value_type value_type;
            typedef typename csc_t::col_vec_t::index_type index_type;

            if (b_use_bias) {
                value_type res = 0.0;
                index_type loop_range;

                // Is the bias term in the weight vector nonzero?
                bool nz_bias = weight.nnz > 0 && weight.idx[weight.nnz - 1] == weight_dim - 1;

                // If bias is nonzero in weight vector
                if (nz_bias) {
                    // Skip over bias when performing inner product
                    loop_range = weight.nnz - 1;
                    // Add bias to result
                    res += bias * weight.val[loop_range];
                } else {
                    loop_range = weight.nnz;
                }

                // Do remaining inner product
                for (index_type s = 0; s < loop_range; s++) {
                    res += query[weight.idx[s]] * weight.val[s];
                }

                return res;
            } else {
                return do_dot_product(query, weight);
            }
        }
    };

    // Unchunked version of compute_sparse_predictions.
    template <typename query_matrix_t, typename prediction_matrix_t>
    void w_ops<csc_t, false>::compute_sparse_predictions(const query_matrix_t& X,
        const csc_t& W,
        typename csr_t::mem_index_type* row_ptr,
        typename csr_t::index_type* col_idx, // Sparsity pattern for this layer
        bool b_sort_by_chunk,
        float bias,
        const prediction_matrix_t& prev_layer_pred,
        prediction_matrix_t& curr_layer_pred) {

        typedef typename query_matrix_t::row_vec_t query_row_t;
        typedef typename csc_t::col_vec_t weight_col_t;

        struct compute_query_t {
            typename query_row_t::index_type row;
            typename weight_col_t::index_type col;
            typename csr_t::mem_index_type write_addr;

            bool operator<(const compute_query_t& other) const {
                return col < other.col;
            }
        };

        auto rows = X.rows;
        auto cols = W.cols;
        auto nnz = row_ptr[rows];

        bool b_use_bias = bias > 0.0;

        typedef typename csr_t::index_type index_type;
        typedef typename csr_t::value_type value_type;
        typedef typename csr_t::mem_index_type mem_index_type;

        std::vector<compute_query_t> queries(nnz);

        curr_layer_pred.rows = rows;
        curr_layer_pred.cols = cols;
        curr_layer_pred.row_ptr = new mem_index_type[rows + 1];
        std::memcpy(curr_layer_pred.row_ptr, row_ptr, sizeof(mem_index_type) * (rows + 1));
        curr_layer_pred.col_idx = new index_type[nnz];
        std::memcpy(curr_layer_pred.col_idx, col_idx, sizeof(index_type) * nnz);
        curr_layer_pred.val = new value_type[nnz];

#pragma omp parallel for schedule(dynamic,4)
        for (index_type row = 0; row < rows; ++row) {
            for (mem_index_type i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
                queries[i].row = row;
                queries[i].col = col_idx[i];
                queries[i].write_addr = i;
            }
        }

        // Sort by columns
        if (b_sort_by_chunk) {
            std::sort(queries.begin(), queries.end());
        }

#pragma omp parallel for schedule(dynamic,64)
        for (mem_index_type i_query = 0; i_query < nnz; ++i_query) {
            compute_query_t* q = &queries[i_query];
            auto Xi = X.get_row(q->row);
            auto Wj = W.get_col(q->col);

            // Do dot product
            curr_layer_pred.val[q->write_addr] = vector_ops<query_row_t, weight_col_t>::inner_product(
                    Xi, Wj, W.rows, bias, b_use_bias);
        }

    }

    // Prolongates the predictions of the previous layer to all of the children of nodes in
    // the active beam. The result is returned as a csr_t matrix. This also has the dual purpose
    // of computing the sparsity pattern for predictions of the current layer, as the prolongated
    // labels and the predictions will have the same sparsity pattern.
    csr_t prolongate_predictions(const csr_t& csr_pred, const csc_t& C) {
        typedef typename csr_t::mem_index_type mem_index_type;
        typedef typename csr_t::index_type index_type;
        typedef typename csr_t::value_type value_type;

        auto rows = csr_pred.rows;
        auto cols = C.rows;

        // Compute the nnz's of each row
        // We convert this to row_idx later, so start indexing at 1 instead of 0
        mem_index_type* row_ptr = new mem_index_type[rows + 1];
        row_ptr[0] = 0;
        for (index_type row = 0; row < rows; ++row) {
            mem_index_type row_nnz = 0;

            mem_index_type row_start = csr_pred.row_ptr[row];
            mem_index_type row_end = csr_pred.row_ptr[row + 1];

            // Number of elements in a column of C is the number of children of that cluster
            for (mem_index_type i = row_start; i < row_end; ++i) {
                row_nnz += C.nnz_of_col(csr_pred.col_idx[i]);
            }

            row_ptr[row + 1] = row_nnz;
        }

        // Perform summation
        for (index_type i = 0; i < rows; ++i) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Allocate the col_idx entries
        auto nnz = row_ptr[rows];
        index_type* col_idx = new index_type[nnz];
        value_type* val = new value_type[nnz];

        // Actually compute the resulting labels
#pragma omp parallel for schedule(dynamic,4)
        for (index_type row = 0; row < rows; ++row) {
            mem_index_type csr_pred_row_start = csr_pred.row_ptr[row];
            mem_index_type csr_pred_row_end = csr_pred.row_ptr[row + 1];

            mem_index_type output_row_start = row_ptr[row];
            mem_index_type i = output_row_start;

            for (mem_index_type j = csr_pred_row_start; j < csr_pred_row_end; ++j) {
                mem_index_type C_col_start = C.col_ptr[csr_pred.col_idx[j]];
                mem_index_type C_col_end = C.col_ptr[csr_pred.col_idx[j] + 1];

                for (mem_index_type k = C_col_start; k < C_col_end; ++k) {
                    col_idx[i] = C.row_idx[k];
                    val[i] = csr_pred.val[j];
                    ++i;
                }
            }
        }

        csr_t result;
        result.col_idx = col_idx;
        result.row_ptr = row_ptr;
        result.rows = rows;
        result.cols = cols;
        result.val = val;
        return result;
    }

    // Obtain the top k values of each row of X sorted in decreasing order by their value.
    // This is meant to be the C++ equivalent of the Python function sorted_csr in smat_util.py
    template <typename prediction_matrix_t>
    void sorted_csr(prediction_matrix_t& X, const uint32_t k) {
        typedef typename csr_t::mem_index_type mem_index_type;
        typedef typename csr_t::index_type index_type;
        typedef typename csr_t::value_type value_type;

        auto rows = X.rows;
        auto cols = X.cols;

        mem_index_type* new_row_ptr = new mem_index_type[rows + 1];
        new_row_ptr[0] = 0;

        // Determine sizes of each row (i.e., a row may have less than k elements)
        for (index_type row = 0; row < rows; ++row) {
            new_row_ptr[row + 1] = new_row_ptr[row] + std::min<index_type>(X.nnz_of_row(row), k);
        }
        mem_index_type nnz = new_row_ptr[rows];

        auto new_col_idx = new index_type[nnz];
        auto new_val = new value_type[nnz];

        // X_permutation is used to rearrange the elements so that the top k in
        // value are at the beginning of every row.
        std::vector<mem_index_type> X_permutation(X.get_nnz());

#pragma omp parallel for schedule(dynamic,2)
        for (index_type row = 0; row < rows; ++row) {
            mem_index_type source_row_start = X.row_ptr[row];
            mem_index_type source_row_end = X.row_ptr[row + 1];
            mem_index_type source_row_size = source_row_end - source_row_start;
            mem_index_type target_write_head = new_row_ptr[row];
            mem_index_type copy_size = new_row_ptr[row + 1] - new_row_ptr[row];
            mem_index_type source_read_head = X.row_ptr[row];

            auto vals = X.val;

            // Initially set X_permutation as the identity
            for (mem_index_type i = X.row_ptr[row]; i < X.row_ptr[row + 1]; ++i) {
                X_permutation[i] = i;
            }

            // A compare function to sort elements in this row by value
            auto comp = [vals](const mem_index_type i, const mem_index_type j) {
                if (vals[i] == vals[j]) {
                    // Break ties by column index. Technically this is arbitrary,
                    // but we need this to pass the tests.
                    return j > i;
                } else {
                    return vals[i] > vals[j];
                }
            };

            if (source_row_size > k) {
                // Select top elements in each row using quick select
                // Afterwards, they will appear at the beginning of X_permutation
                std::nth_element(&X_permutation[source_row_start], &X_permutation[source_row_start + k],
                    &X_permutation[source_row_end], comp);
            }

            // Sort all entries to copy over in decreasing order by value
            std::sort(&X_permutation[source_row_start], &X_permutation[source_row_start + copy_size], comp);

            // Copy all selected entries
            for (mem_index_type i = 0; i < copy_size; ++i, ++target_write_head, ++source_read_head) {
                new_val[target_write_head] = X.val[X_permutation[source_read_head]];
                new_col_idx[target_write_head] = X.col_idx[X_permutation[source_read_head]];
            }
        }
        X.free_underlying_memory();
        X.rows = rows;
        X.cols = cols;
        X.val = new_val;
        X.col_idx = new_col_idx;
        X.row_ptr = new_row_ptr;

    }

    // Prolongates the predictions of the previous layer to the select children of nodes.
    // The result is returned as a csr_t matrix.
    csr_t prolongate_sparse_predictions(const csr_t& prev_layer_pred, const csc_t& C, const csr_t& selected_outputs_csr) {
        typedef typename csr_t::mem_index_type mem_index_type;
        typedef typename csr_t::index_type index_type;
        typedef typename csr_t::value_type value_type;

        auto rows = selected_outputs_csr.rows;
        auto cols = selected_outputs_csr.cols;

        // Compute the nnz's of each row
        mem_index_type* row_ptr = new mem_index_type[rows + 1];
        std::memcpy(row_ptr, selected_outputs_csr.row_ptr, sizeof(mem_index_type) * (rows + 1));

        // Allocate the col_idx entries
        auto nnz = row_ptr[rows];
        index_type* col_idx = new index_type[nnz];
        value_type* val = new value_type[nnz];

        // Actually compute the resulting labels
#pragma omp parallel for schedule(dynamic,4)
        for (index_type row = 0; row < rows; ++row) {
            mem_index_type prev_layer_pred_row_start = selected_outputs_csr.row_ptr[row];
            mem_index_type prev_layer_pred_row_end = selected_outputs_csr.row_ptr[row + 1];

            unordered_set<index_type> valid_cols;
            valid_cols.reserve(prev_layer_pred_row_end - prev_layer_pred_row_start);
            for (mem_index_type i = prev_layer_pred_row_start; i < prev_layer_pred_row_end; ++i) {
                valid_cols.insert(selected_outputs_csr.col_idx[i]);
            }

            mem_index_type csr_pred_row_start = prev_layer_pred.row_ptr[row];
            mem_index_type csr_pred_row_end = prev_layer_pred.row_ptr[row + 1];

            mem_index_type output_row_start = row_ptr[row];
            mem_index_type k = output_row_start;

            for (mem_index_type i = csr_pred_row_start; i < csr_pred_row_end; ++i) {
                mem_index_type C_col_start = C.col_ptr[prev_layer_pred.col_idx[i]];
                mem_index_type C_col_end = C.col_ptr[prev_layer_pred.col_idx[i] + 1];

                for (mem_index_type j = C_col_start; j < C_col_end; ++j) {
                    if (valid_cols.count(C.row_idx[j])) {
                        col_idx[k] = C.row_idx[j];
                        val[k] = prev_layer_pred.val[i];
                        ++k;
                    }
                }
            }
        }

        csr_t result;
        result.col_idx = col_idx;
        result.row_ptr = row_ptr;
        result.rows = rows;
        result.cols = cols;
        result.val = val;
        return result;
    }

    void transform_matrix_csr(const PostProcessor<typename csr_t::value_type>& post_processor,
        csr_t& mat) {
        typedef typename csr_t::value_type value_type;
        typedef typename csr_t::mem_index_type mem_index_type;

        mem_index_type nnz = mat.get_nnz();

#pragma omp parallel for schedule(dynamic,64)
        for (mem_index_type i = 0; i < nnz; ++i) {
            mat.val[i] = (value_type) post_processor.transform(mat.val[i]);
        }
    }

    void combine_matrices_csr(const PostProcessor<typename csr_t::value_type>& post_processor,
        csr_t& mat1, csr_t& mat2) {
        typedef typename csr_t::value_type value_type;
        typedef typename csr_t::mem_index_type mem_index_type;

        mem_index_type nnz = mat1.get_nnz();

#pragma omp parallel for schedule(dynamic,64)
        for (mem_index_type i = 0; i < nnz; ++i) {
            mat1.val[i] = (value_type) post_processor.combiner(mat1.val[i], mat2.val[i]);
        }
    }

    template <typename T>
    struct statistics_t {
        T q0;
        T q1;
        T q2;
        T q3;
        T q4;
        T mean;
    };

    template <typename T>
    statistics_t<T> compute_statistics(std::vector<T>& data) {
        statistics_t<T> result;
        std::sort(data.begin(), data.end());
        result.q0 = data[0];
        result.q1 = data[data.size() / 4];
        result.q2 = data[data.size() / 2];
        result.q3 = data[3 * data.size() / 4];
        result.q4 = data[data.size() - 1];

        T sum = static_cast<T>(0);
        for (auto dat : data) {
            sum += dat;
        }

        result.mean = sum / data.size();
        return result;
    }

    struct layer_statistics_t {
        typedef typename csc_t::index_type index_type;
        typedef typename csc_t::mem_index_type mem_index_type;

        statistics_t<index_type> nnz_per_col;
        mem_index_type nnz;
        index_type num_children;
        index_type num_parents;

        static layer_statistics_t compute(const csc_t& W, const csc_t& C) {
            std::vector<index_type> nnz_per_col_data;
            nnz_per_col_data.reserve(W.cols);
            for (index_type i = 0; i < W.cols; ++i) {
                nnz_per_col_data.emplace_back(W.col_ptr[i + 1] - W.col_ptr[i]);
            }

            layer_statistics_t result;
            result.nnz_per_col = compute_statistics(nnz_per_col_data);
            result.nnz = W.get_nnz();
            result.num_children = C.rows;
            result.num_parents = C.cols;
            return result;
        }
    };

    struct query_statistics_t {
        typedef typename csr_t::index_type index_type;
        typedef typename csr_t::mem_index_type mem_index_type;

        statistics_t<index_type> nnz_per_row;
        index_type rows;
        index_type cols;
        mem_index_type nnz;

        static query_statistics_t compute(const csr_t& X) {
            std::vector<index_type> nnz_per_row_data;
            nnz_per_row_data.reserve(X.rows);
            for (index_type i = 0; i < X.rows; ++i) {
                nnz_per_row_data.emplace_back(X.row_ptr[i + 1] - X.row_ptr[i]);
            }

            query_statistics_t result;
            result.nnz_per_row = compute_statistics(nnz_per_row_data);
            result.nnz = X.get_nnz();
            result.rows = X.rows;
            result.cols = X.cols;
            return result;
        }
    };

    csr_t csr_npz_to_csr_t_deep_copy(ScipyCsrF32Npz& mat) {
        csr_t result;
        result.rows = mat.rows();
        result.cols = mat.cols();
        result.col_idx = mat.indices.data();
        result.row_ptr = mat.indptr.data();
        result.val = mat.data.data();
        return result.deep_copy();
    }

    csc_t csc_npz_to_csc_t_deep_copy(ScipyCscF32Npz& mat) {
        csc_t result;
        result.rows = mat.rows();
        result.cols = mat.cols();
        result.row_idx = mat.indices.data();
        result.col_ptr = mat.indptr.data();
        result.val = mat.data.data();
        return result.deep_copy();
    }

    // An abstract interface for a layer of the model
    template <typename index_type, typename value_type>
    class IModelLayer {
    protected:
        virtual void init(
            csc_t& W,
            csc_t& C,
            uint32_t depth,
            bool b_assumes_ownership,
            MLModelMetadata& metadata
        ) = 0;
        virtual void init_mmap(
            const std::string foldername,
            uint32_t depth,
            MLModelMetadata& metadata,
            const bool lazy_load
        ) = 0;
        static IModelLayer<index_type, value_type>* instantiate(const layer_type_t layer_type);
        static void load(const std::string& folderpath, const uint32_t cur_depth,
            IModelLayer<index_type, value_type>* model);
        static void load_mmap(const std::string& folderpath, const uint32_t cur_depth, const bool lazy_load,
            IModelLayer<index_type, value_type>* model);

    public:
        virtual void save_mmap(
            const std::string& folderpath
        ) const = 0;
        virtual void predict(
            const csr_t& X,
            const csr_t& prev_layer_pred,
            bool no_prev_pred,
            const uint32_t overridden_only_topk,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) = 0;
        virtual void predict(
            const drm_t& X,
            csr_t& prev_layer_pred,
            bool no_prev_pred,
            const uint32_t overridden_only_topk,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) = 0;

        virtual void predict_on_selected_outputs(
            const csr_t& X,
            const csr_t& selected_outputs_csr,
            const csr_t& prev_layer_pred,
            bool no_prev_pred,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) = 0;
        virtual void predict_on_selected_outputs(
            const drm_t& X,
            const csr_t& selected_outputs_csr,
            csr_t& prev_layer_pred,
            bool no_prev_pred,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) = 0;

        virtual ~IModelLayer() = 0;

        virtual csc_t get_C() const = 0;

        // Layer statistics
        virtual layer_statistics_t get_statistics() const = 0;
        virtual layer_type_t get_type() const = 0;
        virtual index_type label_count() const = 0;
        virtual index_type feature_count() const = 0;
        virtual index_type code_count() const = 0;
        virtual value_type bias() const = 0;

        static IModelLayer<index_type, value_type>* instantiate(const std::string& folderpath,
            const layer_type_t layer_type, const uint32_t cur_depth);
        static IModelLayer<index_type, value_type>* instantiate_mmap(const std::string& folderpath,
            const layer_type_t layer_type, const uint32_t cur_depth, const bool lazy_load);
    };

    template <typename index_type, typename value_type>
    void IModelLayer<index_type, value_type>::load(const std::string& folderpath,
        const uint32_t cur_depth,
        IModelLayer<index_type, value_type>* model) {
        MLModelMetadata metadata(folderpath + "/param.json");

        // Load npz matrices. These are large data struct contains multiple vectors
        std::string w_npz_path = folderpath + "/W.npz";
        std::string c_npz_path = folderpath + "/C.npz";
        ScipyCscF32Npz* npz_W = new ScipyCscF32Npz;
        ScipyCscF32Npz* npz_C = new ScipyCscF32Npz;
        npz_W->load(w_npz_path);
        if ((cur_depth == 0) && (access(c_npz_path.c_str(), F_OK) != 0)) {
            // this is to handle the case where the root layer does not have code saved.
            npz_C->fill_ones(npz_W->cols(), 1);
        } else {
            npz_C->load(c_npz_path);
        }

        // We perform a deep copy because MLModel assumes ownership of the memory.
        csc_t csc_C = csc_npz_to_csc_t_deep_copy(* npz_C);
        csc_t csc_W = csc_npz_to_csc_t_deep_copy(* npz_W);

        // Free npz matrices to reduce memory footprint
        delete npz_W;
        delete npz_C;

        // Init model
        model->init(csc_W, csc_C, cur_depth, true, metadata);
    }

    template <typename index_type, typename value_type>
    void IModelLayer<index_type, value_type>::load_mmap(
        const std::string& folderpath,
        const uint32_t cur_depth,
        const bool lazy_load,
        IModelLayer<index_type, value_type>* model) {
        MLModelMetadata metadata(folderpath + "/param.json");

        model->init_mmap(folderpath, cur_depth, metadata, lazy_load);
    }

    template <typename index_type, typename value_type>
    IModelLayer<index_type, value_type>* IModelLayer<index_type, value_type>::instantiate(
        const std::string& folderpath,
        const layer_type_t layer_type, const uint32_t cur_depth) {
        IModelLayer* result = IModelLayer::instantiate(layer_type);
        IModelLayer::load(folderpath, cur_depth, result);

        return result;
    }

    template <typename index_type, typename value_type>
    IModelLayer<index_type, value_type>* IModelLayer<index_type, value_type>::instantiate_mmap(
        const std::string& folderpath,
        const layer_type_t layer_type, const uint32_t cur_depth,
        const bool lazy_load) {
        IModelLayer* result = IModelLayer::instantiate(layer_type);
        IModelLayer::load_mmap(folderpath, cur_depth, lazy_load, result);

        return result;
    }

    template <typename index_type, typename value_type>
    IModelLayer<index_type, value_type>::~IModelLayer() {
    }

    template <typename matrix_t,
        bool chunked = WEIGHT_MATRIX_METADATA_<matrix_t>::IS_CHUNKED>
    class LayerData;

    // Unchunked layer data
    template <typename matrix_t>
    class LayerData<matrix_t, false> {
    public:
        typedef typename matrix_t::index_type index_type;
        typedef typename matrix_t::value_type value_type;

        // Classifier weights of each layer
        // Feature dimension x Cluster dimension
        matrix_t W;

        // Parent to child indicator matrix
        // Child cluster dimension x Parent cluster dimension
        csc_t C;

        // Whether or not this structure has ownership of W and C matrices
        bool b_assumes_ownership;

        // The bias for this layer if the model uses a bias
        value_type bias;

        // Initializes this layer data
        void init(csc_t& W, csc_t& C, bool b_assumes_ownership, value_type bias) {
            this->bias = bias;
            this->b_assumes_ownership = b_assumes_ownership;
            this->W = W;
            this->C = C;
        }

        // Initialize mmap data
        void init_mmap(const std::string& foldername, bool lazy_load, value_type bias) {
            throw std::runtime_error("Not implemented yet.");
        }

        // Save layer data to mmap format
        void save_mmap(const std::string& foldername) const {
            throw std::runtime_error("Not implemented yet.");
        }

        // Not necessary for unchuncked layer data
        void reorder_prediction(csr_t& prediction) {
        }

        // Frees all memory that is owned by this class
        ~LayerData() {
            if (b_assumes_ownership) {
                W.free_underlying_memory();
                C.free_underlying_memory();
            }
        }
    };

    // Chunked layer data
    template <typename chunked_matrix_t>
    class LayerData<chunked_matrix_t, true> {
    public:
        typedef typename chunked_matrix_t::index_type index_type;
        typedef typename chunked_matrix_t::value_type value_type;

        template <typename index_type=uint32_t>
        struct rearrangement_t {
            mmap_util::MmapableVector<index_type> perm; // The rearrangement, stored as a vector
            mmap_util::MmapableVector<index_type> perm_inv; // The inverse of the rearrangement

            // mmap store
            mmap_util::MmapStore mmap_store;

            ~rearrangement_t() {
                perm.clear();
                perm_inv.clear();
            }

            // mmap save/load
            void save_to_mmap_store(mmap_util::MmapStore& mmap_s) const {
                perm.save_to_mmap_store(mmap_s);
                perm_inv.save_to_mmap_store(mmap_s);
            }

            void load_from_mmap_store(mmap_util::MmapStore& mmap_s) {
                perm.load_from_mmap_store(mmap_s);
                perm_inv.load_from_mmap_store(mmap_s);
            }

            void save_mmap(const std::string& file_name) const {
                mmap_util::MmapStore mmap_s = mmap_util::MmapStore();
                mmap_s.open(file_name, "w");
                save_to_mmap_store(mmap_s);
                mmap_s.close();
            }

            void load_mmap(const std::string& file_name, const bool lazy_load) {
                mmap_store.open(file_name, lazy_load?"r_lazy":"r");
                load_from_mmap_store(mmap_store);
            }

            // Creates rearrangement to reorder the rows of C so that they are in correct contiguous order
            // The index type of the rearrangement must be the same as the index type of the input matrix
            // C has at most one non-zero value in each row.
            void initialize_from_codes(const csc_t& C) {

                typedef typename csc_t::mem_index_type mem_index_type;

                perm.resize(C.rows);
                mem_index_type new_size = C.get_nnz();
                for (mem_index_type i=0; i < C.rows; ++i){
                    perm[i] = new_size;
                }

                perm_inv.resize(new_size);
                for (mem_index_type i = 0; i < new_size; ++i) {
                    perm[C.row_idx[i]] = i;
                    perm_inv[i] = C.row_idx[i];
                }

            }

            // Creates a rearranged C (contiguously arranged)from existing C.
            // The input C matrix here must be the same as the matrix that passes into initialize_from_codes
            csc_t get_rearranged_codes(const csc_t& C) {

                typedef typename csc_t::mem_index_type mem_index_type;
                csc_t C_rearranged = C.deep_copy();
                C_rearranged.rows = perm_inv.size();
                for (mem_index_type i = 0; i < C_rearranged.rows; ++i) {
                    C_rearranged.row_idx[i] = perm[C_rearranged.row_idx[i]];
                }
                return C_rearranged;
            }

            void rearrange_prediction_result_back(csr_t& mat){

                typedef typename csr_t::mem_index_type mem_index_type;

                mem_index_type nnz = mat.get_nnz();
                for (mem_index_type i = 0; i < nnz; ++i)
                    mat.col_idx[i] = perm_inv[mat.col_idx[i]];
                mat.cols = perm.size();
            }


            csc_t get_rearranged_weight_matrix(const csc_t& mat) {

                typedef typename csc_t::mem_index_type mem_index_type;
                typedef typename csc_t::value_type value_type;

                csc_t result;
                result.rows = mat.rows;
                result.cols = perm_inv.size();
                mem_index_type new_nnz = 0;
                for (index_type col = 0; col < mat.cols; ++col){
                    if (perm[col] < perm_inv.size()) {
                        new_nnz += mat.nnz_of_col(col);
                    }
                }

                result.row_idx = new index_type[new_nnz];
                result.val = new value_type[new_nnz];
                result.col_ptr = new mem_index_type[result.cols + 1];
                result.col_ptr[0] = 0;

                // Copy memory from source
                for (index_type col = 0; col < result.cols; ++col) {
                    index_type original_col = perm_inv[col];
                    mem_index_type column_size = mat.nnz_of_col(original_col);
                    result.col_ptr[col + 1] = result.col_ptr[col] + column_size;

                    mem_index_type read_addr = mat.col_ptr[original_col];
                    mem_index_type write_addr = result.col_ptr[col];

                    std::memcpy(&result.row_idx[write_addr], &mat.row_idx[read_addr],
                        sizeof(index_type) * column_size);
                    std::memcpy(&result.val[write_addr], &mat.val[read_addr],
                        sizeof(value_type) * column_size);
                }
                return result;
            }

        };

        // Classifier weights of each layer
        // Feature dimension x Cluster dimension
        chunked_matrix_t W;

        // Parent to child indicator matrix
        // Child cluster dimension x Parent cluster dimension
        csc_t C;

        // Algorithm requires children to be contiguous by parent.
        // If the children are not contiguous by parent, they are reordered,
        // and the resulting reordering is stored in children_rearrangement.
        bool b_children_reordered;
        rearrangement_t<typename csc_t::index_type> children_rearrangement;

        // Whether or not this layer assumes ownership of the W and C matrices it is passed.
        // If set to true, the user should not access the W and C matrices after they have
        // been given to the model, as it is not guaranteed that they will still be in memory.
        bool b_assumes_ownership;

        // The bias for this layer if the model uses a bias
        value_type bias;

        // Initializes this layer data
        void init(csc_t& _W, csc_t& _C, bool b_assumes_ownership, value_type bias) {
            bool b_has_bias = bias > 0.0;
            this->bias = bias;
            this->b_assumes_ownership = b_assumes_ownership;

            if (!check_if_contiguously_ordered(_C)) {
                // For chunked matrices to work, the children of a layer must be contiguously ordered
                // If this is not the case, then we must compute a rearrangement to reorder them
                // This rearrangement must then be undone during inference time.
                children_rearrangement.initialize_from_codes(_C);

                csc_t C_rearranged = children_rearrangement.get_rearranged_codes(_C);

                auto W_rearranged = children_rearrangement.get_rearranged_weight_matrix(_W);

                if (b_assumes_ownership) {
                    _C.free_underlying_memory();
                    _W.free_underlying_memory();
                }

                this->b_children_reordered = true;
                make_chunked_W_from_layer_matrices<chunked_matrix_t>(W_rearranged, C_rearranged, b_has_bias, this->W);
                this->C = C_rearranged;
                W_rearranged.free_underlying_memory();
            }
            else {
                this->b_children_reordered = false;
                make_chunked_W_from_layer_matrices<chunked_matrix_t>(_W, _C, b_has_bias, this->W);
                this->C = _C;

                if (b_assumes_ownership) {
                    _W.free_underlying_memory();
                }
            }
        }

        // Initializes mmap for layer data
        void init_mmap(const std::string& foldername, const bool lazy_load, value_type bias) {
            this->bias = bias;
            this->b_assumes_ownership = true; // Always true for mmap

            // load W
            // W is already chunktized
            this->W.load_mmap(mmap_W_fn_(foldername), lazy_load);

            // load C
            // C is already permuted
            this->C.load_mmap(mmap_C_fn_(foldername), lazy_load);

            // load rearrangement if exists
            std::string perm_mmap_fn = mmap_perm_fn_(foldername);
            if (access(perm_mmap_fn.c_str(), F_OK) == 0) { // Rearrangement mmap file exist
                this->b_children_reordered = true;
                this->children_rearrangement.load_mmap(perm_mmap_fn, lazy_load);
            }
            else {
                this->b_children_reordered = false;
            }
        }

        // Save layer data to mmap format
        void save_mmap(const std::string& foldername) const {
            W.save_mmap(mmap_W_fn_(foldername));
            C.save_mmap(mmap_C_fn_(foldername));
            if (b_children_reordered) {
                children_rearrangement.save_mmap(mmap_perm_fn_(foldername));
            }
        }

        // Not necessary for unchuncked layer data
        void reorder_prediction(csr_t& prediction) {
            if (b_children_reordered) {
                children_rearrangement.rearrange_prediction_result_back(prediction);
            }
        }

        // Frees all memory that is owned by this class
        ~LayerData() {
            C.free_underlying_memory();
        }

    private:
        // mmap file names
        inline std::string mmap_W_fn_(const std::string& foldername) const {return foldername + "/W.mmap_store";}
        inline std::string mmap_C_fn_(const std::string& foldername) const {return foldername + "/C.mmap_store";}
        inline std::string mmap_perm_fn_(const std::string& foldername) const {return foldername + "/perm.mmap_store";}
    };

    template <typename w_matrix_t>
    class MLModel :
        public IModelLayer<
            typename w_matrix_t::index_type,
            typename w_matrix_t::value_type> {
    public:
        typedef typename w_matrix_t::index_type index_type;
        typedef typename w_matrix_t::value_type value_type;
        typedef IModelLayer<index_type, value_type> ISpecializedModelLayer;

    private:
        // Metadata
        MLModelMetadata metadata;

        // The matrix data for this layer
        LayerData<w_matrix_t> layer_data;

        // Layer statistics for benchmarking
        layer_statistics_t statistics;

        // The depth of this layer
        uint32_t cur_depth;

        // Prediction kwargs
        PostProcessor<value_type> post_processor;
        uint32_t only_topk;

    protected:
        void init(
            csc_t& W,
            csc_t& C,
            uint32_t depth,
            bool b_assumes_ownership,
            MLModelMetadata& metadata
        ) override {
            this->metadata = metadata;

            statistics = layer_statistics_t::compute(W, C);
            layer_data.init(W, C, b_assumes_ownership, metadata.bias);
            cur_depth = depth;

            post_processor = PostProcessor<value_type>::get(metadata.post_processor);
            only_topk = metadata.only_topk;
        }

        void init_mmap(
            const std::string foldername,
            const uint32_t depth,
            MLModelMetadata& metadata,
            const bool lazy_load
        ) override {
            this->metadata = metadata;

            // No statistics to init for mmap since original W and C do not exist
            layer_data.init_mmap(foldername, lazy_load, metadata.bias);
            cur_depth = depth;

            post_processor = PostProcessor<value_type>::get(metadata.post_processor);
            only_topk = metadata.only_topk;
        }

    public:
        MLModel() {
        }

        MLModel(
            csc_t& W,
            csc_t& C,
            uint32_t cur_depth,
            bool b_assumes_ownership,
            MLModelMetadata& metadata
        ) {
            init(W, C, cur_depth, b_assumes_ownership, metadata);
        }

        MLModel(
            ScipyCscF32& W,
            ScipyCscF32& C,
            uint32_t cur_depth,
            bool b_assumes_ownership,
            MLModelMetadata& metadata
        ) {
            init(csc_t(&W),  csc_t(&C), cur_depth, b_assumes_ownership, metadata);
        }

        // Save mmap
        void save_mmap(const std::string& folderpath) const {
            const std::string metadata_path = folderpath + "/param.json";
            metadata.dump_json(metadata_path);

            layer_data.save_mmap(folderpath);
        }

        // The internal prediction function for a layer, this method is templated to take any
        // supported query matrix type. It is called by both versions of the ModelLayer::predict method
        // X should have the same number of rows as prev_layer_pred
        // prev_layer_pred should have the same number of cols as layer_data.C
        // If layer_data.bias > 0, the row number of layer_data.W, which is the dimension of W, should be one more than the number of cols of X.
        // If layer_data.bias > 0, the row number of layer_data.W, which is the dimension of W, should be one more than the number of cols of X.
        // If layer_data.bias <= 0, the row number of layer_data.W, which is the dimension of W, should be same as the number of cols of X.
        template <typename query_mat_t, typename prediction_matrix_t>
        void predict_internal(
            const query_mat_t& X,
            const prediction_matrix_t& prev_layer_pred,
            bool no_prev_pred,
            const uint32_t overridden_only_topk,
            const char* overridden_post_processor,
            prediction_matrix_t& curr_layer_pred,
            const int threads=-1,
            const bool b_sort_by_chunk=true
        ) {

            // Check that the prev_layer_pred is of valid shape
            if (prev_layer_pred.rows != X.rows) {
                throw std::invalid_argument(
                    "Instance dimension of query and prev_layer_pred matrix do not match"
                );
            }
            if (prev_layer_pred.cols != layer_data.C.cols) {
                throw std::invalid_argument(
                    "Label dimension of prev_layer_pred and C matrix do not match"
                );
            }

            set_threads(threads);

            uint32_t only_topk_to_use = (overridden_only_topk > 0) ? overridden_only_topk : only_topk;
            const PostProcessor<value_type>& post_processor_to_use =
                (overridden_post_processor == nullptr) ? post_processor
                    : PostProcessor<value_type>::get(overridden_post_processor);

            // Prolongate predictions of previous layer to this layer
            csr_t labels = prolongate_predictions(prev_layer_pred, layer_data.C);

            // Compute predictions for this layer
            w_ops<w_matrix_t>::compute_sparse_predictions(X, layer_data.W,
                labels.row_ptr, labels.col_idx,
                b_sort_by_chunk, layer_data.bias, prev_layer_pred, curr_layer_pred);

            // Transform the predictions for this layer and combine with previous layer
            transform_matrix_csr(post_processor_to_use, curr_layer_pred);
            if (!no_prev_pred) {
                combine_matrices_csr(post_processor_to_use, curr_layer_pred, labels);
            }
            labels.free_underlying_memory();

            // Narrow the search to the top k results
            sorted_csr(curr_layer_pred, only_topk_to_use);

            // Reorder columns of prediction if necessary
            layer_data.reorder_prediction(curr_layer_pred);
        }

        void predict(
            const csr_t& X,
            const csr_t& prev_layer_pred,
            bool no_prev_pred,
            const uint32_t overridden_only_topk,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) override {
            bool b_sort_by_chunk = (X.rows > 1) ? true : false;
            predict_internal<csr_t, csr_t>(
                X,
                prev_layer_pred,
                no_prev_pred,
                overridden_only_topk,
                overridden_post_processor,
                curr_layer_pred,
                threads,
                b_sort_by_chunk
            );
        }

        void predict(
            const drm_t& X,
            csr_t& prev_layer_pred,
            bool no_prev_pred,
            const uint32_t overridden_only_topk,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) override {
            bool b_sort_by_chunk=false;
            predict_internal<drm_t, csr_t>(
                X,
                prev_layer_pred,
                no_prev_pred,
                overridden_only_topk,
                overridden_post_processor,
                curr_layer_pred,
                threads,
                b_sort_by_chunk
            );
        }

        // The internal prediction function for a sparse layer prediction, this method is templated to take any
        // supported query matrix type. It is called by both versions of the ModelLayer::predict_on_selected_outputs method
        // X should have the same number of rows as prev_layer_pred
        template <typename query_mat_t, typename prediction_matrix_t>
        void predict_on_selected_outputs_internal(
            const query_mat_t& X,
            const csr_t& selected_outputs_csr,
            const prediction_matrix_t& prev_layer_pred,
            bool no_prev_pred,
            const char* overridden_post_processor,
            prediction_matrix_t& curr_layer_pred,
            const int threads=-1,
            const bool b_sort_by_chunk=true
        ) {

            // Check for valid supported layer type
            if (this->get_type() != LAYER_TYPE_CSC) {
                throw std::invalid_argument(
                    "Predict on selected outputs only supported by layer_type_t = LAYER_TYPE_CSC"
                );
            }

            // Check that the prev_layer_pred is of valid shape
            if (prev_layer_pred.rows != X.rows) {
                throw std::invalid_argument(
                    "Instance dimension of query and prev_layer_pred matrix do not match"
                );
            }
            if (prev_layer_pred.cols != layer_data.C.cols) {
                throw std::invalid_argument(
                    "Label dimension of prev_layer_pred and C matrix do not match"
                );
            }

            set_threads(threads);

            csr_t labels = prolongate_sparse_predictions(prev_layer_pred, layer_data.C, selected_outputs_csr);

            const PostProcessor<value_type>& post_processor_to_use =
                (overridden_post_processor == nullptr) ? post_processor
                    : PostProcessor<value_type>::get(overridden_post_processor);

            // Compute predictions for this layer
            w_ops<w_matrix_t>::compute_sparse_predictions(X, layer_data.W,
                labels.row_ptr, labels.col_idx,
                b_sort_by_chunk, layer_data.bias, prev_layer_pred, curr_layer_pred);

            // Transform the predictions for this layer and combine with previous layer
            transform_matrix_csr(post_processor_to_use, curr_layer_pred);
            if (!no_prev_pred) {
                combine_matrices_csr(post_processor_to_use, curr_layer_pred, labels);
            }

            labels.free_underlying_memory();
        }

        void predict_on_selected_outputs(
            const csr_t& X,
            const csr_t& selected_outputs_csr,
            const csr_t& prev_layer_pred,
            bool no_prev_pred,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) override {
            bool b_sort_by_chunk = (X.rows > 1) ? true : false;
            predict_on_selected_outputs_internal<csr_t, csr_t>(
                X,
                selected_outputs_csr,
                prev_layer_pred,
                no_prev_pred,
                overridden_post_processor,
                curr_layer_pred,
                threads,
                b_sort_by_chunk
            );
        }

        void predict_on_selected_outputs(
            const drm_t& X,
            const csr_t& selected_outputs_csr,
            csr_t& prev_layer_pred,
            bool no_prev_pred,
            const char* overridden_post_processor,
            csr_t& curr_layer_pred,
            const int threads=-1
        ) override {
            bool b_sort_by_chunk=false;
            predict_on_selected_outputs_internal<drm_t, csr_t>(
                X,
                selected_outputs_csr,
                prev_layer_pred,
                no_prev_pred,
                overridden_post_processor,
                curr_layer_pred,
                threads,
                b_sort_by_chunk
            );
        }

        ~MLModel() override {
        }

        csc_t get_C() const override {
            return layer_data.C.deep_copy();
        }

        layer_statistics_t get_statistics() const override {
            return statistics;
        }

        layer_type_t get_type() const override {
            return WEIGHT_MATRIX_METADATA_<w_matrix_t>::LAYER_TYPE;
        }

        index_type label_count() const override {
            return layer_data.W.cols;
        }

        index_type feature_count() const override {
            if (layer_data.bias > 0.0) {
                return layer_data.W.rows - 1;
            } else {
                return layer_data.W.rows;
            }
        }

        index_type code_count() const override {
            return layer_data.C.cols;
        }

        value_type bias() const override {
            return layer_data.bias;
        }

        MLModel(const std::string& folderpath, const uint32_t cur_depth) {
            ISpecializedModelLayer::load(folderpath, cur_depth, this);
        }
    };

    template <typename index_type, typename value_type>
    IModelLayer<index_type, value_type>* IModelLayer<index_type, value_type>::instantiate(
        const layer_type_t layer_type) {
        switch (layer_type) {
            case LAYER_TYPE_BINARY_SEARCH_CHUNKED:
            {
                typedef typename LAYER_TYPE_METADATA_<LAYER_TYPE_BINARY_SEARCH_CHUNKED>::matrix_t w_matrix_t;
                return new MLModel<w_matrix_t>();
            }
            case LAYER_TYPE_HASH_CHUNKED:
            {
                typedef typename LAYER_TYPE_METADATA_<LAYER_TYPE_HASH_CHUNKED>::matrix_t w_matrix_t;
                return new MLModel<w_matrix_t>();
            }
            case LAYER_TYPE_CSC:
            {
                typedef typename LAYER_TYPE_METADATA_<LAYER_TYPE_CSC>::matrix_t w_matrix_t;
                return new MLModel<w_matrix_t>();
            }
            default:
            {
                typedef typename LAYER_TYPE_METADATA_<DEFAULT_LAYER_TYPE>::matrix_t w_matrix_t;
                return new MLModel<w_matrix_t>();
            }
        }
    }

    // A class defining a chain of layers that form a prediction model
    class HierarchicalMLModel {
    public:
        typedef typename csc_t::index_type index_type;
        typedef typename csc_t::value_type value_type;
        typedef IModelLayer<index_type, value_type> ISpecializedModelLayer;

    private:

        std::vector<ISpecializedModelLayer*> model_layers;

    public:
        ISpecializedModelLayer* operator[](const uint32_t i) {
            return model_layers[i];
        }

        inline uint32_t depth() const {
            return model_layers.size();
        }

        inline index_type label_count() const {
            return model_layers[model_layers.size() - 1]->label_count();
        }

        inline index_type feature_count() const {
            return model_layers[model_layers.size() - 1]->feature_count();
        }

        inline index_type code_count() const {
            return model_layers[model_layers.size() - 1]->code_count();
        }

        inline index_type get_int_attr(const char* attr) {
            if (std::strcmp(attr, "depth") == 0) {
                return this->depth();
            } else if (std::strcmp(attr, "nr_features") == 0) {
                return this->feature_count();
            } else if (std::strcmp(attr, "nr_labels") == 0) {
                return this->label_count();
            } else if (std::strcmp(attr, "nr_codes") == 0) {
                return this->code_count();
            } else {
                throw std::runtime_error(std::string(attr) + " is not implemented in get_int_attr.");
            }
        }

        std::vector<layer_statistics_t> get_layer_statistics() const {
            std::vector<layer_statistics_t> result;
            result.reserve(depth());

            for (auto layer : model_layers) {
                result.emplace_back(layer->get_statistics());
            }

            return result;
        }

        inline const std::vector<ISpecializedModelLayer*>& get_model_layers() const {
            return model_layers;
        }


    private:
        void destroy_layers() {
            // Free memory associated with model matrices
            for (auto layer : model_layers) {
                delete layer;
            }
            model_layers.clear();
        }

    public:
        // Initialize this model by creating all layers
        void init(std::vector<ISpecializedModelLayer*>& layers) {
            // Destroy all memory associated with this layer
            destroy_layers();
            model_layers = layers;
        }

        HierarchicalMLModel() {}

        HierarchicalMLModel(std::vector<ISpecializedModelLayer*>& layers) {
            init(layers);
        }

        ~HierarchicalMLModel() {
            // Free memory associated with model matrices
            destroy_layers();
        }

        /*
        * Perform a prediction using the specified parameters.
        * Parameters:
        *
        * X: The csr matrix of queries. Every row represents a query to the model.
        *
        * overridden_beam_size (optional): The beam size to use in prediction, set to 0 to use defaults.
        *
        * overridden_post_processor (optional): A string specifying which post-processor to use for
        * predictions on each layer of the model. Set to nullptr to use defaults.
        *
        * overridden_only_topk (optional): The number of final predictions to return, set to 0 to use defaults.
        *
        * threads (optional): The number of threads to use for prediction computations. Set to -1 to use maximum
        * of threads.
        *
        * depth (optional): Allows the user to return predictions for a layer of the model other than the last.
        * Set this to 0 to perform prediction for the leaves of the tree.
        *
        * prediction (prediction_matrix_t): prediction output matrix
        */
        template <typename query_matrix_t, typename prediction_matrix_t>
        void predict(
            const query_matrix_t& queries,
            prediction_matrix_t& prediction,
            const uint32_t overridden_beam_size=0,
            const char* overridden_post_processor=nullptr,
            const uint32_t overridden_only_topk=0,
            const int threads=-1,
            const uint32_t depth=0
        ) {

            uint32_t prediction_depth = (depth > 0) ?
                std::min<uint32_t>(depth, model_layers.size()) : model_layers.size();


            // Create first layer's pred;
            prediction_matrix_t prev_layer_pred;
            prev_layer_pred.fill_ones(queries.rows, 1);


            // Run the prediction loop, passing predictions down through layers of the model
            for (uint32_t i_layer = 0; i_layer < prediction_depth; ++i_layer) {
                ISpecializedModelLayer* layer = model_layers[i_layer];

                // Determine topk for this layer
                uint32_t local_only_topk = (i_layer == prediction_depth - 1) ? overridden_only_topk : overridden_beam_size;
                bool no_prev_pred = (i_layer == 0);
                // Run beam search for one layer
                prediction_matrix_t curr_layer_pred;
                layer->predict(
                    queries,
                    prev_layer_pred,
                    no_prev_pred,
                    local_only_topk,
                    overridden_post_processor,
                    curr_layer_pred,
                    threads
                );
                prev_layer_pred.free_underlying_memory();
                prev_layer_pred = curr_layer_pred;
            }
            prediction = prev_layer_pred;
        }

        /*
        * Perform a select prediction using the specified parameters.
        * Parameters:
        *
        * queries: The csr matrix of queries. Every row represents a query to the model.
        *
        * selected_outputs_csr: The csr matrix of selected outputs. Each non zero entry represents
        * a pair to predict
        *
        * overridden_post_processor (optional): A string specifying which post-processor to use for
        * predictions on each layer of the model. Set to nullptr to use defaults.
        *
        * threads (optional): The number of threads to use for prediction computations. Set to -1 to use maximum
        * of threads.
        *
        * prediction (prediction_matrix_t): prediction output matrix
        */
        template <typename query_matrix_t, typename prediction_matrix_t>
        void predict_on_selected_outputs(
            const query_matrix_t& queries,
            const csr_t& selected_outputs_csr,
            prediction_matrix_t& prediction,
            const char* overridden_post_processor=nullptr,
            const int threads=-1
        ) {
            uint32_t prediction_depth = model_layers.size();

            // Check for valid supported layer types
            for (uint32_t i_layer = 0; i_layer < prediction_depth; ++i_layer) {
                ISpecializedModelLayer* layer = model_layers[i_layer];

                if (layer->get_type() != LAYER_TYPE_CSC) {
                    throw std::invalid_argument(
                        "Predict on selected outputs only supported by layer_type_t = LAYER_TYPE_CSC"
                    );
                }
            }

            // Find the sparsity pattern of each layer
            std::vector<csr_t> selected_outputs_csrs(prediction_depth);
            selected_outputs_csrs[0] = selected_outputs_csr;

            for (uint32_t i_layer = 1; i_layer < prediction_depth; ++i_layer) {
                ISpecializedModelLayer* layer = model_layers[prediction_depth - i_layer];
                csc_t C = layer->get_C();
                csr_t csr_C = C.to_csr();
                csr_t output_csr;
                smat_x_smat(selected_outputs_csrs[i_layer - 1], csr_C, output_csr, false, true, threads);
                selected_outputs_csrs[i_layer] = output_csr;
                C.free_underlying_memory();
                csr_C.free_underlying_memory();
            }

            // Create first layer's prev pred;
            prediction_matrix_t prev_layer_pred;
            prev_layer_pred.fill_ones(queries.rows, 1);

            // Run the prediction loop, passing predictions down through layers of the model
            for (uint32_t i_layer = 0; i_layer < prediction_depth; ++i_layer) {
                ISpecializedModelLayer* layer = model_layers[i_layer];

                bool no_prev_pred = (i_layer == 0);
                // Find the prediction for one layer
                prediction_matrix_t curr_layer_pred;
                layer->predict_on_selected_outputs(
                    queries,
                    selected_outputs_csrs[prediction_depth - 1 - i_layer],
                    prev_layer_pred,
                    no_prev_pred,
                    overridden_post_processor,
                    curr_layer_pred,
                    threads
                );
                prev_layer_pred.free_underlying_memory();
                prev_layer_pred = curr_layer_pred;
            }
            prediction = prev_layer_pred;

            for (uint32_t i = 1; i < prediction_depth; ++i) {
                selected_outputs_csrs[i].free_underlying_memory();
            }
        }

        // Save mmap
        // Currently only bin_search
        void save_mmap(
            const std::string& folderpath
        ) const {
            // Create folder
            if (system(("mkdir -p " + folderpath).c_str()) == -1) {
                throw std::runtime_error("Cannot create folder: " + folderpath);
            }

            // Dump metadata
            auto depth = model_layers.size();
            std::string metadata_path = folderpath + "/param.json";
            HierarchicalMLModelMetadata metadata(depth, true);
            metadata.dump_json(metadata_path);

            // Save each layer
            for (std::size_t d = 0; d < depth; d++) {
                std::string layer_path = folderpath + "/" + std::to_string(d) + ".model/";
                // Create folder for layer
                if (system(("mkdir -p " + layer_path).c_str()) == -1) {
                    throw std::runtime_error("Cannot create layer folder: " + layer_path);
                }
                model_layers[d]->save_mmap(layer_path);
            }
        }

        static void load_mmap(
            const std::string& folderpath,
            HierarchicalMLModel* model,
            const int depth,
            const bool lazy_load
        ) {
            auto layer_type = LAYER_TYPE_BINARY_SEARCH_CHUNKED; // Only supported type for mmap
            std::vector<ISpecializedModelLayer*> layers(depth);

            // Abstractly instantiate every layer
            for (auto d = 0; d < depth; d++) {
                std::string layer_path = folderpath + "/" + std::to_string(d) + ".model/";
                layers[d] = ISpecializedModelLayer::instantiate_mmap(layer_path, layer_type, d, lazy_load);
            }

            // Model chain assumes ownership of the memory associated with the matrices above
            model->init(layers);
        }

        static void load(
            const std::string& folderpath,
            HierarchicalMLModel* model,
            const int depth,
            layer_type_t layer_type = DEFAULT_LAYER_TYPE
        ) {
            std::vector<ISpecializedModelLayer*> layers(depth);

            // Abstractly instantiate every layer
            for (auto d = 0; d < depth; d++) {
                std::string layer_path = folderpath + "/" + std::to_string(d) + ".model/";
                layers[d] = ISpecializedModelLayer::instantiate(layer_path, layer_type, d);
            }

            // Model chain assumes ownership of the memory associated with the matrices above
            model->init(layers);
        }

        // Constructor for mmap
        HierarchicalMLModel(
            const std::string& folderpath,
            const bool lazy_load
        ) {
            HierarchicalMLModelMetadata xlinear_metadata(folderpath + "/param.json");
            if (!xlinear_metadata.is_mmap) {
                throw std::runtime_error("This folder contains npz model. Cannot load in mmap format.");
            }
            HierarchicalMLModel::load_mmap(folderpath, this, xlinear_metadata.depth, lazy_load);
        }

        HierarchicalMLModel(
            const std::string& folderpath,
            layer_type_t layer_type = DEFAULT_LAYER_TYPE
        ) {
            HierarchicalMLModelMetadata xlinear_metadata(folderpath + "/param.json");
            if (xlinear_metadata.is_mmap) {
                throw std::runtime_error("This folder contains mmap model. Cannot load in npz format.");
            }
            HierarchicalMLModel::load(folderpath, this, xlinear_metadata.depth, layer_type);
        }
    };
} // end namespace pecos

#endif // end of __INFERENCE_H__
