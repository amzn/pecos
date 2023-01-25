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

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "mmap_util.hpp"
#include "parallel.hpp"
#include "scipy_loader.hpp"

typedef float float32_t;
typedef double float64_t;


typedef std::vector<uint32_t> u32_dvec_t;
typedef std::vector<uint64_t> u64_dvec_t;
typedef std::vector<float32_t> f32_dvec_t;
typedef std::vector<float64_t> f64_dvec_t;


// ===== C Interface for Structure/Types =====

extern "C" {

    // rows, cols, nnz, &row_ptr, &col_ptr, &val_ptr
    typedef void(*py_coo_allocator_t)(uint64_t, uint64_t, uint64_t, void*, void*, void*);
    typedef void (*py_sparse_allocator_t)(bool, uint64_t, uint64_t, uint64_t, void*, void*, void*);

    typedef struct {
        uint32_t rows, cols;
        uint64_t* col_ptr;
        uint32_t* row_idx;
        float* val;
    } ScipyCscF32;

    typedef struct {
        uint32_t rows, cols;
        uint64_t* row_ptr;
        uint32_t* col_idx;
        float* val;
    } ScipyCsrF32;

    typedef struct {
        uint32_t rows, cols;
        float* val;
    } ScipyDrmF32;

    typedef struct {
        uint32_t rows, cols;
        float* val;
    } ScipyDcmF32;

} // end of extern C

namespace pecos {
    // ===== Container for sparse-dense vectors =====
    // For sparse vectors computational acceleration
    template<class IDX_T=uint32_t, class VAL_T=float32_t>
    struct sdvec_t {
        typedef IDX_T index_type;
        typedef VAL_T value_type;

        struct entry_t {
            value_type val;
            bool touched;
            entry_t(value_type val=0, bool touched=0): val(val), touched(touched) {}
        };

        struct container_t {
            index_type len;
            index_type nr_touch;
            std::vector<entry_t> entries;
            std::vector<index_type> touched_indices;

            container_t(index_type len=0) : len(len), nr_touch(0) {
                entries.resize(len);
                touched_indices.resize(len);
            }

            void resize(index_type len_) {
                len = len_;
                // if len_ >= len, nr_touch remains unchanged, otherwise delete indices >= len
                if(len_ < len) {
                    index_type write_pos = 0;
                    for(size_t i = 0; i < nr_touch; i++) {
                        if(touched_indices[i] < len) {
                            touched_indices[write_pos] = touched_indices[i];
                            write_pos += 1;
                        } 
                    }
                    nr_touch = write_pos;
                }
                entries.resize(len);
                touched_indices.resize(len);
            }
        };

        index_type& len;
        index_type& nr_touch;
        std::vector<entry_t>& entries;
        std::vector<index_type>& touched_indices;

        sdvec_t(container_t& cont) : len(cont.len), nr_touch(cont.nr_touch), entries(cont.entries), touched_indices(cont.touched_indices) {}

        value_type& add_nonzero_at(index_type idx, value_type v) {
            entries[idx].val += static_cast<value_type>(v);
            if(!entries[idx].touched) {
                entries[idx].touched = 1;
                touched_indices[nr_touch++] = static_cast<index_type>(idx);
            }
            return entries[idx].val;
        }

        void sort_nz_indices() {
            std::sort(touched_indices.data(), touched_indices.data() + nr_touch);
        }

        value_type& operator[](size_t i) { return add_nonzero_at(i, 0.0); }
        const value_type& operator[](size_t i) const { return entries[i].val; }

        void fill_zeros() {
            if(nr_touch < (len >> 1)) {
                for(size_t t = 0; t < nr_touch; t++) {
                    entries[touched_indices[t]].val = 0;
                    entries[touched_indices[t]].touched = 0;
                }
            } else {
                memset(static_cast<void*>(entries.data()), 0, sizeof(entry_t) * len);
            }
            nr_touch = 0;
        }
    };

    // ===== Wrapper for sparse/dense vectors =====
    template<class IDX_T=uint32_t, class VAL_T=float32_t>
    struct sparse_vec_t {
        typedef IDX_T index_type;
        typedef VAL_T value_type;

        index_type nnz;
        index_type* idx;
        value_type* val;
        sparse_vec_t(index_type nnz=0, index_type* idx=NULL, value_type* val=NULL): nnz(nnz), idx(idx), val(val) {}

        index_type get_nnz() const { return nnz; }
    };

    template<class VAL_T=float32_t>
    struct dense_vec_t {
        typedef VAL_T value_type;
        typedef uint64_t index_type;

        uint64_t len;
        value_type* val;
        dense_vec_t(uint64_t len=0, value_type* val=NULL): len(len), val(val) {}
        dense_vec_t(std::vector<value_type>& x): len(x.size()), val(x.data()) {}

        value_type& operator[](size_t i) { return val[i]; }
        const value_type& operator[](size_t i) const { return val[i]; }

        value_type& at(size_t i) { return val[i]; }
        const value_type& at(size_t i) const { return val[i]; }

        uint64_t get_nnz() const { return len; }
        void fill_zeros() const{ std::fill(val, val + len, 0); }
    };

    // ===== Wrapper for sparse/dense matrices =====
    struct csr_t;
    struct csc_t;
    struct drm_t;
    struct dcm_t;

    struct csr_t { // Compressed Sparse Rows
        typedef float32_t value_type;
        typedef uint32_t index_type;
        typedef uint64_t mem_index_type;
        typedef sparse_vec_t<index_type, value_type> row_vec_t;

        const static bool IS_COLUMN_MAJORED = false;
        index_type rows, cols;

        // create alias names for indptr, indices, data,
        // which are later used in smat_x_smat function.
        union {
            mem_index_type *row_ptr;
            mem_index_type *indptr;
        };
        union {
            index_type *col_idx;
            index_type *indices;
        };
        union {
            value_type *val;
            value_type *data;
        };
        csr_t() :
            rows(0),
            cols(0),
            row_ptr(nullptr),
            col_idx(nullptr),
            val(nullptr) { }

        csr_t(const ScipyCsrF32* py) :
            rows(py->rows),
            cols(py->cols),
            row_ptr(py->row_ptr),
            col_idx(py->col_idx),
            val(py->val) { }

        // Save/load mmap
        // Signature for symmetry, not implemented
        void save_to_mmap_store(mmap_util::MmapStore& mmap_s) const {
            throw std::runtime_error("Not implemented yet.");
        }

        void load_from_mmap_store(mmap_util::MmapStore& mmap_s) {
            throw std::runtime_error("Not implemented yet.");
        }

        void save_mmap(const std::string& file_name) const {
            throw std::runtime_error("Not implemented yet.");
        }

        void load_mmap(const std::string& file_name, const bool lazy_load) {
            throw std::runtime_error("Not implemented yet.");
        }

        bool is_empty() const {
            return val == nullptr;
        }

        index_type nnz_of_row(index_type idx) const {
            return static_cast<index_type>(row_ptr[idx + 1] - row_ptr[idx]);
        }

        mem_index_type get_nnz() const {
            return row_ptr[rows];
        }

        row_vec_t get_row(index_type idx) const {
            auto offset = row_ptr[idx];
            return row_vec_t(nnz_of_row(idx), &col_idx[offset], &val[offset]);
        }

        // Frees the underlying memory of the matrix (i.e., col_ptr, row_idx, and val arrays)
        // Every function in the inference code that returns a matrix has allocated memory, and
        // therefore one should call this function to free that memory.
        void free_underlying_memory() {
            if (row_ptr) {
                delete[] row_ptr;
                row_ptr = nullptr;
            }
            if (col_idx) {
                delete[] col_idx;
                col_idx = nullptr;
            }
            if (val) {
                delete[] val;
                val = nullptr;
            }
        }

        csc_t transpose() const;
        csc_t to_csc() const;

        // Creates a deep copy of this matrix
        // This allocates memory, so one should call free_underlying_memory on the copy when
        // one is finished using it.
        csr_t deep_copy() const {
            mem_index_type nnz = row_ptr[rows];
            csr_t res;
            res.allocate(rows, cols, nnz);
            std::memcpy(res.col_idx, col_idx, sizeof(index_type) * nnz);
            std::memcpy(res.val, val, sizeof(value_type) * nnz);
            std::memcpy(res.row_ptr, row_ptr, sizeof(mem_index_type) * (rows + 1));
            return res;
        }

        void create_pycsr(const py_sparse_allocator_t& pred_alloc) const {
            uint64_t nnz = row_ptr[rows];
            uint32_t* indices = nullptr;
            uint64_t* indptr = nullptr;
            float* data = nullptr;

            // Copy everything into the python allocated memory
            pred_alloc(false, rows, cols, nnz, &indices, &indptr, &data);

            for (mem_index_type i = 0; i < nnz; ++i) {
                indices[i] = col_idx[i];
                data[i] = val[i];
            }
            for (index_type i = 0; i < rows + 1; ++i) {
                indptr[i] = row_ptr[i];
            }
        }

        void allocate(index_type rows, index_type cols, mem_index_type nnz) {
            this->rows = rows;
            this->cols = cols;
            row_ptr = new mem_index_type[rows + 1];
            col_idx = new index_type[nnz];
            val = new value_type[nnz];
        }

        // Construct a csr_t object with shape _rows x _cols filled by 1.
        void fill_ones(index_type _rows, index_type _cols) {
            mem_index_type nnz = (mem_index_type) _rows * _cols;
            this->allocate(_rows, _cols, nnz);

            row_ptr[0] = 0;
            mem_index_type ind = 0;
            for(index_type r = 0; r < rows; r++) {
                for(index_type c = 0; c < cols; c++) {
                    col_idx[ind] = c;
                    val[ind] = 1.0;
                    ind++;
                }
                row_ptr[r + 1] = row_ptr[r] + cols;
            }
        }
    };

    struct csc_t { // Compressed Sparse Columns
        typedef float32_t value_type;
        typedef uint32_t index_type;
        typedef uint64_t mem_index_type;
        typedef sparse_vec_t<index_type, value_type> col_vec_t;

        const static bool IS_COLUMN_MAJORED = true;
        index_type rows, cols;

        // create alias names for indptr, indices, data,
        // which are later used in smat_x_smat function.
        union {
            mem_index_type *col_ptr;
            mem_index_type *indptr;
        };
        union {
            index_type *row_idx;
            index_type *indices;
        };
        union {
            value_type *val;
            value_type *data;
        };

        // mmap
        std::shared_ptr<mmap_util::MmapStore> mmap_store_ptr = nullptr;

        csc_t() :
            rows(0),
            cols(0),
            col_ptr(nullptr),
            row_idx(nullptr),
            val(nullptr) { }

        csc_t(const ScipyCscF32* py) :
            rows(py->rows),
            cols(py->cols),
            col_ptr(py->col_ptr),
            row_idx(py->row_idx),
            val(py->val) { }

        // Save/load mmap
        void save_to_mmap_store(mmap_util::MmapStore& mmap_s) const {
            auto nnz = get_nnz();
            // scalars
            mmap_s.fput_one<index_type>(rows);
            mmap_s.fput_one<index_type>(cols);
            mmap_s.fput_one<mem_index_type>(nnz);
            // arrays
            mmap_s.fput_multiple<mem_index_type>(col_ptr, cols + 1);
            mmap_s.fput_multiple<index_type>(row_idx, nnz);
            mmap_s.fput_multiple<value_type>(val, nnz);
        }

        void load_from_mmap_store(mmap_util::MmapStore& mmap_s) {
            // scalars
            rows = mmap_s.fget_one<index_type>();
            cols = mmap_s.fget_one<index_type>();
            auto nnz = mmap_s.fget_one<mem_index_type>();
            // arrays
            col_ptr = mmap_s.fget_multiple<mem_index_type>(cols + 1);
            row_idx = mmap_s.fget_multiple<index_type>(nnz);
            val = mmap_s.fget_multiple<value_type>(nnz);
        }

        void save_mmap(const std::string& file_name) const {
            mmap_util::MmapStore mmap_s = mmap_util::MmapStore();
            mmap_s.open(file_name, "w");
            save_to_mmap_store(mmap_s);
            mmap_s.close();
        }

        void load_mmap(const std::string& file_name, const bool lazy_load) {
            free_underlying_memory(); // Clear any existing memory
            mmap_store_ptr = std::make_shared<mmap_util::MmapStore>(); // Create instance
            mmap_store_ptr->open(file_name, lazy_load?"r_lazy":"r");
            load_from_mmap_store(*mmap_store_ptr);
        }

        bool is_empty() const {
            return val == nullptr;
        }

        index_type nnz_of_col(index_type idx) const {
            return static_cast<index_type>(col_ptr[idx + 1] - col_ptr[idx]);
        }

        mem_index_type get_nnz() const {
            return col_ptr[cols];
        }

        col_vec_t get_col(index_type idx) const {
            auto offset = col_ptr[idx];
            return col_vec_t(nnz_of_col(idx), &row_idx[offset], &val[offset]);
        }

        // Frees the underlying memory of the matrix (i.e., col_ptr, row_idx, and val arrays)
        // Every function in the inference code that returns a matrix has allocated memory, and
        // therefore one should call this function to free that memory.
        void free_underlying_memory() {
            if (mmap_store_ptr) { // mmap case, no need to check and free other pointers
                mmap_store_ptr.reset(); // decrease reference count
            } else { // memory case
                if (col_ptr) {
                    delete[] col_ptr;
                }
                if (row_idx) {
                    delete[] row_idx;
                }
                if (val) {
                    delete[] val;
                }
            }
            mmap_store_ptr = nullptr;
            col_ptr = nullptr;
            row_idx = nullptr;
            val = nullptr;
            rows = 0;
            cols = 0;
        }

        csr_t transpose() const ;
        csr_t to_csr() const;

        // Creates a deep copy of this matrix
        // This allocates memory, so one should call free_underlying_memory on the copy when
        // one is finished using it.
        csc_t deep_copy() const {
            if (mmap_store_ptr) {
                throw std::runtime_error("Cannot deep copy for mmap instance.");
            }
            mem_index_type nnz = col_ptr[cols];
            csc_t res;
            res.allocate(rows, cols, nnz);
            std::memcpy(res.row_idx, row_idx, sizeof(index_type) * nnz);
            std::memcpy(res.val, val, sizeof(value_type) * nnz);
            std::memcpy(res.col_ptr, col_ptr, sizeof(mem_index_type) * (cols + 1));
            return res;
        }

        void allocate(index_type rows, index_type cols, mem_index_type nnz) {
            if (mmap_store_ptr) {
                throw std::runtime_error("Cannot allocate for mmap instance.");
            }
            this->rows = rows;
            this->cols = cols;
            col_ptr = new mem_index_type[cols + 1];
            row_idx = new index_type[nnz];
            val = new value_type[nnz];
        }

        // Construct a csc_t object with shape _rows x _cols filled by 1.
        void fill_ones(index_type _rows, index_type _cols) {
            if (mmap_store_ptr) {
                throw std::runtime_error("Cannot fill ones for mmap instance.");
            }
            mem_index_type nnz = (mem_index_type) _rows * _cols;
            this->free_underlying_memory();
            this->allocate(_rows, _cols, nnz);
            col_ptr[0] = 0;
            mem_index_type ind = 0;
            for(index_type c = 0; c < cols; c++) {
                for(index_type r = 0; r < rows; r++) {
                    row_idx[ind] = r;
                    val[ind] = 1.0;
                    ind++;
                }
                col_ptr[c + 1] = col_ptr[c] + rows;
            }
        }
    };

    struct drm_t { // Dense Row Majored Matrix
        typedef float32_t value_type;
        typedef uint32_t index_type;
        typedef uint64_t mem_index_type;
        typedef dense_vec_t<value_type> row_vec_t;

        index_type rows, cols;
        value_type *val;

        drm_t() {}

        drm_t(const ScipyDrmF32* py) {
            rows = py->rows;
            cols = py->cols;
            val = py->val;
        }

        row_vec_t get_row(index_type idx) const {
            return row_vec_t(cols,
                &val[static_cast<mem_index_type>(cols) * static_cast<mem_index_type>(idx)]);
        }

        dcm_t transpose() const ;

        mem_index_type get_nnz() const {
            return static_cast<mem_index_type>(rows) * static_cast<mem_index_type>(cols);
        }
    };

    struct dcm_t { // Dense Column Majored Matrix
        typedef float32_t value_type;
        typedef uint32_t index_type;
        typedef uint64_t mem_index_type;
        typedef dense_vec_t<value_type> col_vec_t;

        index_type rows, cols;
        value_type *val;

        dcm_t() {}

        dcm_t(const ScipyDcmF32* py) {
            rows = py->rows;
            cols = py->cols;
            val = py->val;
        }

        drm_t transpose() const ;

        col_vec_t get_col(index_type idx) const {
            return col_vec_t(cols,
                &val[static_cast<mem_index_type>(rows) * static_cast<mem_index_type>(idx)]);
        }

        mem_index_type get_nnz() const {
            return static_cast<mem_index_type>(rows) * static_cast<mem_index_type>(cols);
        }
    };

    // Transpose Methods
    csc_t csr_t::transpose() const {
        csc_t ret;
        ret.rows = cols;
        ret.cols = rows;
        ret.col_ptr = row_ptr;
        ret.row_idx = col_idx;
        ret.val = val;
        return ret;
    }

    csr_t csc_t::transpose() const {
        csr_t ret;
        ret.rows = cols;
        ret.cols = rows;
        ret.row_ptr = col_ptr;
        ret.col_idx = row_idx;
        ret.val = val;
        return ret;
    }

    dcm_t drm_t::transpose() const {
        dcm_t ret;
        ret.rows = cols;
        ret.cols = rows;
        ret.val = val;
        return ret;
    }

    drm_t dcm_t::transpose() const {
        drm_t ret;
        ret.rows = cols;
        ret.cols = rows;
        ret.val = val;
        return ret;
    }

    // CSC to CSR
    csc_t csr_t::to_csc() const {
        csc_t ret;
        auto nnz = this->get_nnz();
        ret.rows = rows;
        ret.cols = cols;
        ret.col_ptr = new mem_index_type[cols + 1];
        ret.row_idx = new index_type[nnz];
        ret.val = new value_type[nnz];
        memset(ret.col_ptr, 0, sizeof(mem_index_type) * (cols + 1));
        for(mem_index_type s = 0; s < nnz; ++s) {
            ++ret.col_ptr[col_idx[s] + 1];
        }
        std::partial_sum(ret.col_ptr, ret.col_ptr + cols + 1, ret.col_ptr);
        for(index_type r = 0; r < rows; r++) {
            for(mem_index_type s = row_ptr[r]; s < row_ptr[r + 1]; ++s) {
                index_type c = col_idx[s];
                ret.row_idx[ret.col_ptr[c]] = r;
                ret.val[ret.col_ptr[c]++] = val[s];
            }
        }
        std::move_backward(ret.col_ptr, ret.col_ptr + cols, ret.col_ptr + cols + 1);
        ret.col_ptr[0] = 0;
        return ret;
    }

    csr_t csc_t::to_csr() const {
        return this->transpose().to_csc().transpose();
    }


    // ===== Container for Sparse Coordinate Matrix
    struct coo_t { // Coordinate Sparse Matrix
        typedef float32_t value_type;

        uint32_t rows;
        uint32_t cols;
        std::vector<uint32_t> row_idx;
        std::vector<uint32_t> col_idx;
        std::vector<value_type> val;

        coo_t(uint32_t rows=0, uint32_t cols=0): rows(rows), cols(cols) {}

        size_t nnz() const { return val.size(); }

        void reshape(uint32_t rows_, uint32_t cols_) {
            rows = rows_;
            cols = cols_;
            clear();
        }

        void clear() {
            row_idx.clear();
            col_idx.clear();
            val.clear();
        }

        void reserve(size_t capacity) {
            row_idx.reserve(capacity);
            col_idx.reserve(capacity);
            val.reserve(capacity);
        }

        void swap(coo_t& other) {
            std::swap(rows, other.rows);
            std::swap(cols, other.cols);
            row_idx.swap(other.row_idx);
            col_idx.swap(other.col_idx);
            val.swap(other.val);
        }

        void extends(coo_t& other) {
            std::copy(other.row_idx.begin(), other.row_idx.end(), std::back_inserter(row_idx));
            std::copy(other.col_idx.begin(), other.col_idx.end(), std::back_inserter(col_idx));
            std::copy(other.val.begin(), other.val.end(), std::back_inserter(val));
        }

        template<typename I, typename V>
        void push_back(I i, I j, V x, double threshold=0) {
            if(std::fabs(x) >= threshold) {
                row_idx.push_back(i);
                col_idx.push_back(j);
                val.push_back(x);
            }
        }

        void create_pycoo(const py_coo_allocator_t& alloc) const {
            uint64_t* row_ptr=NULL;
            uint64_t* col_ptr=NULL;
            value_type* val_ptr=NULL;
            alloc(rows, cols, nnz(), &row_ptr, &col_ptr, &val_ptr);
            for(size_t i = 0; i < nnz(); i++) {
                row_ptr[i] = row_idx[i];
                col_ptr[i] = col_idx[i];
                val_ptr[i] = val[i];
            }
        }
    };

    // ===== Container for Sparse Matrix used by spmm.hpp
    template<bool is_col_major>
    struct spmm_mat_t {
        typedef uint64_t mem_index_type;
        typedef uint32_t index_type;
        typedef float value_type;

        const static bool IS_COLUMN_MAJORED = is_col_major;
        index_type rows, cols;
        mem_index_type *indptr;
        index_type *indices;
        value_type *data;
        py_sparse_allocator_t pred_alloc;

        spmm_mat_t(py_sparse_allocator_t pred_alloc) :
            rows(0),
            cols(0),
            indptr(nullptr),
            indices(nullptr),
            data(nullptr),
            pred_alloc(pred_alloc) {}

        void allocate(index_type rows, index_type cols, mem_index_type nnz) {
            this->rows = rows;
            this->cols = cols;
            pred_alloc(IS_COLUMN_MAJORED, rows, cols, nnz, &indices, &indptr, &data);
        }
    };

    // ===== BLAS C++ Wrapper =====

    extern "C" {
        double ddot_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
        float sdot_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);

        ptrdiff_t dscal_(ptrdiff_t *, double *, double *, ptrdiff_t *);
        ptrdiff_t sscal_(ptrdiff_t *, float *, float *, ptrdiff_t *);

        ptrdiff_t daxpy_(ptrdiff_t *, double *, double *, ptrdiff_t *, double *, ptrdiff_t *);
        ptrdiff_t saxpy_(ptrdiff_t *, float *, float *, ptrdiff_t *, float *, ptrdiff_t *);

        double dcopy_(ptrdiff_t *, double *, ptrdiff_t *, double *, ptrdiff_t *);
        float scopy_(ptrdiff_t *, float *, ptrdiff_t *, float *, ptrdiff_t *);
    }

    template<typename val_type> val_type dot(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
    template<> inline double dot(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return ddot_(len, x, xinc, y, yinc); }
    template<> inline float dot(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return sdot_(len, x, xinc, y, yinc); }

    template<typename val_type> val_type scal(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *);
    template<> inline double scal(ptrdiff_t *len, double *a, double *x, ptrdiff_t *xinc) { return dscal_(len, a, x, xinc); }
    template<> inline float scal(ptrdiff_t *len, float *a,  float *x, ptrdiff_t *xinc) { return sscal_(len, a, x, xinc); }

    template<typename val_type> ptrdiff_t axpy(ptrdiff_t *, val_type *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
    template<> inline ptrdiff_t axpy(ptrdiff_t *len, double *alpha, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return daxpy_(len, alpha, x, xinc, y, yinc); };
    template<> inline ptrdiff_t axpy(ptrdiff_t *len, float *alpha, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return saxpy_(len, alpha, x, xinc, y, yinc); };

    template<typename val_type> val_type copy(ptrdiff_t *, val_type *, ptrdiff_t *, val_type *, ptrdiff_t *);
    template<> inline double copy(ptrdiff_t *len, double *x, ptrdiff_t *xinc, double *y, ptrdiff_t *yinc) { return dcopy_(len,x,xinc,y,yinc); }
    template<> inline float copy(ptrdiff_t *len, float *x, ptrdiff_t *xinc, float *y, ptrdiff_t *yinc) { return scopy_(len,x,xinc,y,yinc); }

    // ===== do_dot_product =====
    template<class IX, class VX, class IY, class VY>
    float32_t do_dot_product(const sparse_vec_t<IX, VX>& x, const sdvec_t<IY, VY>& y) {
        float32_t ret = 0;
        for(size_t s = 0; s < x.nnz; s++) {
            auto &idx = x.idx[s];
            if(y.entries[idx].touched)
                ret += y.entries[idx].val * x.val[s];
        }
        return ret;
    }

    template<class IX, class VX, class IY, class VY>
    float32_t do_dot_product(const sdvec_t<IX, VX>& x, const sparse_vec_t<IY, VY>& y) {
        return do_dot_product(y, x);
    }


    template<typename val_type>
    val_type do_dot_product(const val_type *x, const val_type *y, size_t size) {
        // This uses a BLAS implementation
        val_type *xx = const_cast<val_type*>(x);
        val_type *yy = const_cast<val_type*>(y);
        ptrdiff_t inc = 1;
        ptrdiff_t len = (ptrdiff_t) size;
        return dot(&len, xx, &inc, yy, &inc);
    }

    template<class IX, class VX, class IY, class VY>
    float32_t do_dot_product(const sparse_vec_t<IX, VX>& x, const sparse_vec_t<IY, VY>& y) {
        // This function assume that nz entries in both x and y are stored in an
        // ascending order in terms of idx
        if(x.nnz > y.nnz) { return do_dot_product(y, x); }

        //float64_t ret = 0;
        float32_t ret = 0;
        size_t s = 0, t = 0;
        IX *xend = x.idx + x.nnz;
        IY *yend = y.idx + y.nnz;
        while(s < x.nnz && t < y.nnz) {
            if(x.idx[s] == y.idx[t]) {
                ret += x.val[s] * y.val[t];
                s++;
                t++;
            } else if(x.idx[s] < y.idx[t]) {
                s = std::lower_bound(x.idx + s, xend, y.idx[t]) - x.idx;
            } else {
                t = std::lower_bound(y.idx + t, yend, x.idx[s]) - y.idx;
            }
        }
        return static_cast<float32_t>(ret);
    }

    template<class VX, class VY>
    float32_t do_dot_product(const dense_vec_t<VX>& x, const dense_vec_t<VY>& y) {
        float32_t ret = 0;
        for(size_t i = 0; i < x.len; i++) {
            ret += x[i] * y[i];
        }
        return ret;
    }

    template<class VX, class IY, class VY>
    float32_t do_dot_product(const dense_vec_t<VX>& x, const sparse_vec_t<IY, VY>& y) {
        float32_t ret = 0;
        for(size_t s = 0; s < y.nnz; s++) {
            ret += x[y.idx[s]] * y.val[s];
        }
        return ret;
    }

    template<class IX, class VX, class VY>
    float32_t do_dot_product(const sparse_vec_t<IX, VX>& x, const dense_vec_t<VY>& y) {
        return do_dot_product(y, x);
    }


    template<class IX, class VX, class IY, class VY>
    float32_t do_dot_product(const sdvec_t<IX, VX>& x, const sdvec_t<IY, VY>& y) {
        if(x.nr_touch > y.nr_touch) {
            return do_dot_product(y, x);
        }
        float32_t ret = 0;
        for(size_t s = 0; s < x.nr_touch; s++) {
            auto &idx = x.touched_indices[s];
            if(y.entries[idx].touched)
                ret += y.entries[idx].val * x.entries[idx].val;
        }
        return static_cast<float32_t>(ret);
    }

    template<class IX, class VX, class VY>
    float32_t do_dot_product(const sdvec_t<IX, VX>& x, const dense_vec_t<VY>& y) {
        float32_t ret = 0;
        if(x.nr_touch > (x.len >> 1) ) {
            for(size_t i = 0; i < y.len; i++) {
                ret += x.entries[i].val * y[i];
            }
        }
        else {
            for(size_t s = 0; s < x.nr_touch; s++) {
                auto &idx = x.touched_indices[s];
                ret += x.entries[idx].val * y[idx];
            }
        }
        return static_cast<float32_t>(ret);
    }

    template<class VX, class IY, class VY>
    float32_t do_dot_product(const dense_vec_t<VX>& x, const sdvec_t<IY, VY>& y) {
        return do_dot_product(y, x);
    }

    // ===== do_ax2py =====
    template <class VX, class VY, typename T>
    void do_ax2py(T alpha, const dense_vec_t<VX> &x, dense_vec_t<VY> &y) {
        for (size_t i = 0; i < x.len; i++) {
            y[i] += x[i] * x[i] * alpha;
        }
    }

    template <class IX, class VX, class VY, typename T>
    void do_ax2py(T alpha, const sparse_vec_t<IX, VX> &x, dense_vec_t<VY> &y) {
        for (size_t s = 0; s < x.nnz; s++) {
            y[x.idx[s]] += x.val[s] * x.val[s] * alpha;
        }
    }

    // ===== do_xp2y =====
    template <class VX, class VY>
    void do_xp2y(const dense_vec_t<VX> &x, dense_vec_t<VY> &y) {
        for (size_t i = 0; i < x.len; i++) {
            y[i] = x[i] + 2 * y[i];
        }
    }

    template <class IX, class VX, class VY>
    void do_xp2y(const sparse_vec_t<IX, VX> &x, dense_vec_t<VY> &y) {
        for (size_t s = 0; s < x.nnz; s++) {
            y[x.idx[s]] = x.val[s] + 2 * y[x.idx[s]];
        }
    }

    // ===== do_axpy =====
    template<typename val_type, typename T>
    val_type* do_axpy(T alpha, const val_type *x, val_type *y, size_t size) {
        // This uses a BLAS implementation
        if(alpha == 0) { return y; }
        val_type alpha_ = (val_type)alpha;
        ptrdiff_t inc = 1;
        ptrdiff_t len = (ptrdiff_t) size;
        val_type *xx = const_cast<val_type*>(x);
        axpy(&len, &alpha_, xx, &inc, y, &inc);
        return y;
    }

    template<class VX, class VY, typename T>
    void do_axpy(T alpha, const dense_vec_t<VX>&x, dense_vec_t<VY>& y) {
        for(size_t i = 0; i < x.len; i++) {
            y[i] += alpha * x[i];
        }
    }

    template<class IX, class VX, class VY, typename T>
    void do_axpy(T alpha, const sparse_vec_t<IX, VX>&x, dense_vec_t<VY>& y) {
        for(size_t s = 0; s < x.nnz; s++) {
            y[x.idx[s]] += alpha * x.val[s];
        }
    }

    template<class VX, class VY, typename T>
    void do_axpy(T alpha, const VX& x, std::vector<VY>& y) {
        return do_axpy(alpha, x, dense_vec_t<VY>(y));
    }

    template<class IX, class VX, class IY, class VY, typename T>
    void do_axpy(T alpha, const sparse_vec_t<IX, VX>& x, sdvec_t<IY, VY>& y) {
        for(size_t s = 0; s < x.nnz; s++) {
            y.add_nonzero_at(x.idx[s], x.val[s] * alpha);
        }
    }


    template<class IX, class VX, class IY, class VY, typename T>
    void do_axpy(T alpha, const sdvec_t<IX, VX>& x, sdvec_t<IY, VY>& y) {
        for(size_t s = 0; s < x.nr_touch; s++) {
            auto &idx = x.touched_indices[s];
            y.add_nonzero_at(idx, x.entries[idx].val * alpha);
        }
    }


    template<class VX, class IY, class VY, typename T>
    void do_axpy(T alpha, const dense_vec_t<VX>& x, sdvec_t<IY, VY>& y) {
        if(y.nr_touch == x.len)
            for(size_t i = 0; i < x.len; i++) {
                y.entries[i].val += alpha * x[i];
            }
        else {
            for(size_t i = 0; i < x.len; i++) {
                y.add_nonzero_at(i, alpha * x[i]);
            }
        }
    }

    // ===== do_scale =====
    template<class val_type, class T>
    void do_scale(T alpha, val_type *x, size_t size) {
        // This uses a BLAS implementation
        if(alpha == 0.0) {
            memset(x, 0, sizeof(val_type) * size);
        } else if (alpha == 1.0) {
            return;
        } else {
            val_type alpha_minus_one = (val_type)(alpha - 1);
            do_axpy(alpha_minus_one, x, x, size);
        }
    }

    template<class VX, typename T>
    void do_scale(T alpha, dense_vec_t<VX>& x) {
        for(size_t i = 0; i < x.len; i++) {
            x[i] *= alpha;
        }
    }

    template<class VX, typename T>
    void do_scale(T alpha, sparse_vec_t<VX>& x) {
        for(size_t s = 0; s < x.nnz; s++) {
            x.val[s] *= alpha;
        }
    }

    template<class IX, class VX, typename T>
    void do_scale(T alpha, sdvec_t<IX, VX>& x) {
        for(size_t s = 0; s < x.nr_touch; s++) {
            auto &idx = x.touched_indices[s];
            x.entries[idx].val = x.entries[idx].val * alpha;
        }
    }

    template<class X_MAT, class M_MAT, class V>
    void compute_sparse_entries_from_rowmajored_X_and_colmajored_M(const X_MAT& X, const M_MAT& M, uint64_t len, uint32_t *X_row_idx, uint32_t *M_col_idx, V *val, int threads) {
        // This function assume that nz entries in both x and y are stored in an
        // ascending order in terms of idx
        set_threads(threads);
#pragma omp parallel for schedule(dynamic,64)
        for(size_t idx = 0; idx < len; idx++) {
            const auto& xi = X.get_row(X_row_idx[idx]);
            const auto& mj = M.get_col(M_col_idx[idx]);
            val[idx] = static_cast<V>(do_dot_product(xi, mj));
        }
    }

    /*
    spmm_mat_t (defined in xlinear.cpp) is a data structure type that supports
    (1) spmm_mat_t.allocate(), which allocates memory for self.indptr, self.indices, self.data
    (2) spmm_mat_t::IS_COLUMN_MAJORED, which specifies if it is column or row major

    smat_x_smat function:
    given two sparse column major matrices, X and W,
    multithreaded compute sparse-sparse matrix multiplication of X and W,
    and output the results into Z, the spmm_mat_t data type.
    (1) if Z::IS_COL_MAJORED is true, then Z = XW
    (2) if Z::IS_COL_MAJORED is false, then Z = (XW).transpose()
    This design choice is to be compatible with the python interface and handling csc_x_csc case.
    */
    template <typename spmm_mat_t>
    void smat_x_smat(
        const csc_t& X,
        const csc_t& W,
        spmm_mat_t& Z,
        const bool eliminate_zeros=false,
        const bool sorted_indices=true,
        int threads=1
    ) {
        // sanity check
        if(X.cols != W.rows) {
            std::runtime_error("X.cols != W.rows");
        }

        typedef typename csc_t::index_type index_type;
        typedef typename csc_t::mem_index_type mem_index_type;
        typedef typename csc_t::value_type value_type;

        struct sdvec_t {
            index_type len;
            index_type nr_touch;

            struct entry_t {
                value_type val;
                bool touched;
                entry_t(value_type val=0, bool touched=0): val(val), touched(touched) {}
            };

            std::vector<entry_t> entries;
            std::vector<index_type> touched_indices;

            sdvec_t(index_type len=0) : len(len), nr_touch(0) {
                entries.resize(len);
                touched_indices.resize(len);
            }

            void resize(index_type len_) {
                len = len_;
                nr_touch = 0;
                entries.resize(len);
                touched_indices.resize(len);
            }

            value_type& add_nonzero_at(index_type idx, value_type v) {
                entries[idx].val += static_cast<value_type>(v);
                if(!entries[idx].touched) {
                    entries[idx].touched = 1;
                    touched_indices[nr_touch++] = static_cast<index_type>(idx);
                }
                return entries[idx].val;
            }

            void sort_nz_indices() {
                std::sort(touched_indices.data(), touched_indices.data() + nr_touch);
            }

            void clear() {
                if(nr_touch < (len >> 1)) {
                    for(size_t t = 0; t < nr_touch; t++) {
                        entries[touched_indices[t]].val = 0;
                        entries[touched_indices[t]].touched = 0;
                    }
                } else {
                    memset(static_cast<void*>(entries.data()), 0, sizeof(entry_t) * len);
                }
                nr_touch = 0;
            }
        };

        struct worker_t {
            worker_t() {}
            sdvec_t temp;

            void set_rows(index_type rows) { temp.resize(rows); }
        };

        index_type rows = X.rows, cols = W.cols;
        threads = set_threads(threads);

        // compute workloads for each thread
        std::vector<index_type> workloads(threads + 1);
        workloads[0] = 0;
        workloads[threads] = cols;
        if(threads > 1) {
            std::vector<size_t> flops(cols);
#pragma omp parallel for schedule(dynamic,16)
            for(size_t c = 0; c < cols; c++) {
                flops[c] = 0;
                for(size_t s = W.col_ptr[c]; s != W.col_ptr[c + 1]; ++s) {
                    flops[c] += X.nnz_of_col(W.row_idx[s]);
                }
            }
            parallel_partial_sum(flops.begin(), flops.end(), flops.begin(), threads);
            size_t avg_flops = flops[cols - 1] / threads + (flops[cols - 1] % threads != 0);
#pragma omp parallel for schedule(static,1)
            for (int tid = 1; tid < threads; tid++) {
                auto low = std::lower_bound(flops.begin(), flops.end(), tid*avg_flops);
                index_type pos = static_cast<index_type>(low - flops.begin());
                workloads[tid] = (pos >= cols) ? cols - 1 : pos;
            }
        }

        // compute maxnnz of Z = XW, for each column, and use it as col_ptr
        std::vector<mem_index_type> col_ptr(cols + 1);
#pragma omp parallel for schedule(static,1) shared(col_ptr)
        for(int tid = 0; tid < threads; tid++) {
            // the mask vector is essentially a binary sparse accumulator,
            // see https://people.eecs.berkeley.edu/~aydin/GALLA-sparse.pdf.
            // the idx c is strictly smaller than std::numeric_limits<index_type>::max().
            std::vector<index_type> mask(rows, std::numeric_limits<index_type>::max());
            index_type c_start = workloads[tid];
            index_type c_end = workloads[tid + 1];
            for(index_type c = c_start; c < c_end; ++c) {
                auto Wc = W.get_col(c);
                if(Wc.nnz == 0) {
                    col_ptr[c + 1] = 0;
                    continue;
                }
                for(index_type s = 0; s < Wc.nnz; ++s) {
                    auto Xs = X.get_col(Wc.idx[s]);
                    for(index_type t = 0; t < Xs.nnz; ++t) {
                        auto ridx = Xs.idx[t];
                        if (mask[ridx] != c) {
                            mask[ridx] = c;
                            col_ptr[c + 1]++;
                        }
                    }
                }
            }
        }
        parallel_partial_sum(col_ptr.begin(), col_ptr.end(), col_ptr.begin(), threads);
        mem_index_type max_nnz = col_ptr[cols];

        // once getting the maxnnz, use Z to allocate memory
        if(spmm_mat_t::IS_COLUMN_MAJORED) {
            Z.allocate(rows, cols, max_nnz);
        } else {
            Z.allocate(cols, rows, max_nnz);
        }
#pragma omp parallel for schedule(static, 1)
        for (index_type idx = 0; idx < cols + 1; ++idx) {
            Z.indptr[idx] = col_ptr[idx];
        }

        // main matmul block
        std::vector<worker_t> worker_set(threads);
#pragma omp parallel for schedule(static,1)
        for(int tid = 0; tid < threads; tid++) {
            worker_t& worker = worker_set[tid];
            worker.set_rows(rows);
            auto& temp = worker.temp;

            index_type c_start = workloads[tid];
            index_type c_end = workloads[tid + 1];
            for(index_type c = c_start; c < c_end; ++c) {
                auto Wc = W.get_col(c);
                if(Wc.nnz > 0) {
                    temp.clear();
                    for(index_type s = 0; s < Wc.nnz; ++s) {
                        // temp += Wc[i] * Xi
                        auto Xs = X.get_col(Wc.idx[s]);
                        auto Wci = Wc.val[s];
                        for(index_type t = 0; t < Xs.nnz; ++t) {
                            temp.add_nonzero_at(Xs.idx[t], Wci * Xs.val[t]);
                        }
                    }
                    if(sorted_indices) {
                        temp.sort_nz_indices();
                    }
                    for(index_type s = 0; s < temp.nr_touch; s++) {
                        size_t r = temp.touched_indices[s];
                        size_t offset = col_ptr[c] + s;
                        Z.indices[offset] = r;
                        Z.data[offset] = temp.entries[r].val;
                    }
                }
            }
        }

        if(eliminate_zeros) {
            mem_index_type true_nnz = 0;
            mem_index_type col_start = 0;
            for(index_type c = 0; c < cols; c++) {
                for(auto k = col_start; k < Z.indptr[c + 1]; k++) {
                    auto idx = Z.indices[k];
                    auto val = Z.data[k];
                    if(val != 0) {
                        Z.indices[true_nnz] = idx;
                        Z.data[true_nnz] = val;
                        true_nnz++;
                    }
                }
                col_start = Z.indptr[c + 1];
                Z.indptr[c + 1] = true_nnz;
            }
        }
    }

    /*
    smat_x_smat function:
    given two sparse row major matrices, X and W,
    multithreaded compute sparse-sparse matrix multiplication of X and W,
    and output the results into Z, the spmm_mat_t data type.
    (1) if Z::IS_COL_MAJORED is true, then Z = (XW).transpose()
    (2) if Z::IS_COL_MAJORED is false, then Z = XW
    This design choice is to be compatible with the python interface and handling csr_x_csr case.
    */
    template <typename spmm_mat_t>
    void smat_x_smat(
        const csr_t& X,
        const csr_t& W,
        spmm_mat_t& Z,
        const bool eliminate_zeros=false,
        const bool sorted_indices=true,
        int threads=1
    ) {
        smat_x_smat(W.transpose(), X.transpose(), Z, eliminate_zeros, sorted_indices, threads);
    }

    /*
    Memory efficient method to stack <csr_t> matrices horizontally in parallel.
    expect stacked_matrix to be empty csr_t or spmm_mat_t
    */
    template <class MAT_T>
    void hstack_csr(const std::vector<csr_t>& matrices, MAT_T& stacked_matrix, int threads=-1) {
        typedef typename MAT_T::index_type ret_idx_t;
        typedef typename MAT_T::mem_index_type ret_indptr_t;

        // compute (nr_rows, total_cols, total_nnz) for memory allocation of the stacked_matrix matrix
        // all mat in matrices should have the same number of rows
        ret_idx_t nr_rows = matrices[0].rows;
        ret_idx_t total_cols = 0;
        ret_indptr_t total_nnz = 0;
        for(auto& mat : matrices) {
            total_cols += mat.cols;
            total_nnz += mat.get_nnz();
        }
        stacked_matrix.allocate(nr_rows, total_cols, total_nnz);

        set_threads(threads);
        // compute indptr row-wise independently for easy parallelism
#pragma omp parallel for
        for(ret_idx_t i = 0; i <= nr_rows; i++) {
            stacked_matrix.indptr[i] = 0;
            for(auto& mat : matrices) {
                stacked_matrix.indptr[i] += mat.indptr[i];
            }
        }

        // compute indices/data row-wise independently for easy parallelism
#pragma omp parallel for schedule(dynamic,64)
        for(ret_idx_t i = 0; i < nr_rows; i++) {
            // for row_i, column-wise stack mat
            ret_idx_t col_idx_offset = 0;
            ret_indptr_t cumulated_nnz = stacked_matrix.indptr[i];
            for(auto& mat : matrices) {
                const auto& x_i = mat.get_row(i);
                std::copy(
                    x_i.idx,
                    x_i.idx + x_i.nnz,
                    &stacked_matrix.indices[cumulated_nnz]
                );
                std::transform(
                    &stacked_matrix.indices[cumulated_nnz],
                    &stacked_matrix.indices[cumulated_nnz + x_i.nnz],
                    &stacked_matrix.indices[cumulated_nnz],
                    [&](ret_idx_t x) { return x + col_idx_offset; }
                );
                std::copy(
                    x_i.val,
                    x_i.val + x_i.nnz,
                    &stacked_matrix.data[cumulated_nnz]
                );
                col_idx_offset += mat.cols;
                cumulated_nnz += x_i.nnz;
            }
        }
    }


} // end namespace pecos

#endif // end of __MATRIX_H__
