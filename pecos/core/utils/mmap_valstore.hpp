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

#ifndef __MMAP_VALSTORE_H__
#define __MMAP_VALSTORE_H__

#include <omp.h>
#include "mmap_util.hpp"
// #include <iostream>


namespace pecos {
namespace mmap_valstore {

typedef uint64_t row_type;
typedef uint64_t col_type;


class Float32Store {
    typedef float float32_t;

    public:
        typedef float32_t value_type;

        Float32Store():
            n_row_(0),
            n_col_(0),
            vals_(nullptr)
        {}

        row_type n_row() {
            return n_row_;
        }

        col_type n_col() {
            return n_col_;
        }

        // View from external values pointer, does not hold memory
        void from_vals(const row_type n_row, const col_type n_col, const value_type* vals) {
            n_row_ = n_row;
            n_col_ = n_col;
            vals_ = vals;
        }

        void batch_get(const uint64_t n_sub_row, const uint64_t n_sub_col, const row_type* sub_rows, const col_type* sub_cols, value_type* ret, const int threads=1) {
            #pragma omp parallel for schedule(static, 1) num_threads(threads)
            for (uint64_t i=0; i<n_sub_row; ++i) {
                for (uint64_t j=0; j<n_sub_col; ++j) {
                    ret[i * n_sub_col + j] = vals_[sub_rows[i] * n_col_ + sub_cols[j]];
                }
            }
        }

        void save(const std::string& folderpath) {
            auto mmap_s = pecos::mmap_util::MmapStore();
            mmap_s.open(mmap_file_name(folderpath), "w");

            mmap_s.fput_one<row_type>(n_row_);
            mmap_s.fput_one<col_type>(n_col_);
            mmap_s.fput_multiple<value_type>(vals_, n_row_ * n_col_);

            mmap_s.close();
        }

        void load(const std::string& folderpath, const bool lazy_load) {
            mmap_store_.open(mmap_file_name(folderpath), lazy_load?"r_lazy":"r");

            n_row_ = mmap_store_.fget_one<row_type>();
            n_col_ = mmap_store_.fget_one<col_type>();
            vals_ = mmap_store_.fget_multiple<value_type>(n_row_ * n_col_);
        }


    private:
        row_type n_row_;
        col_type n_col_;
        const value_type* vals_;

        pecos::mmap_util::MmapStore mmap_store_;

        inline std::string mmap_file_name(const std::string& folderpath) const {
            return folderpath + "/numerical_float32_2d.mmap_store";
        }
};


class BytesStore {
    public:
        typedef uint32_t bytes_len_type;

        BytesStore():
            n_row_(0),
            n_col_(0)
        {}

        row_type n_row() {
            return n_row_;
        }

        col_type n_col() {
            return n_col_;
        }

        // In memory. Allocate and assign values
        void from_vals(const row_type n_row, const col_type n_col, const char* const* vals, const bytes_len_type* vals_lens) {
            n_row_ = n_row;
            n_col_ = n_col;

            // Allocation
            row_type n_total = n_row * n_col;
            row_type n_total_char = 0;
            for (row_type i=0; i<n_total; ++i) {
                n_total_char += vals_lens[i];
            }
            vals_.resize(n_total_char);
            vals_lens_.resize(n_total);
            vals_starts_.resize(n_total);

            // Assign the MmapVector
            row_type cur_start = 0;
            for (row_type i=0; i<n_total; ++i) {
                vals_lens_[i] = vals_lens[i];
                vals_starts_[i] = cur_start;
                std::memcpy(vals_.data() + cur_start, vals[i], vals_lens[i]);
                cur_start += vals_lens[i];
            }
        }

        void batch_get(const uint64_t n_sub_row, const uint64_t n_sub_col, const row_type* sub_rows, const col_type* sub_cols,
            const bytes_len_type trunc_val_len, char* ret, bytes_len_type* ret_lens, const int threads=1) {
            #pragma omp parallel for schedule(static, 1) num_threads(threads)
            for (uint64_t i=0; i<n_sub_row; ++i) {
                for (uint64_t j=0; j<n_sub_col; ++j) {
                    uint64_t sub_idx = i * n_sub_col + j;
                    row_type idx = sub_rows[i] * n_col_ + sub_cols[j];
                    uint64_t ret_start_idx = sub_idx * trunc_val_len;
                    bytes_len_type cur_ret_len = std::min(trunc_val_len, vals_lens_[idx]);
                    ret_lens[sub_idx] = cur_ret_len;
                    std::memcpy(ret + ret_start_idx, vals_.data() + vals_starts_[idx], cur_ret_len);
                }
            }
        }

        void save(const std::string& folderpath) {
            auto mmap_s = pecos::mmap_util::MmapStore();
            mmap_s.open(mmap_file_name(folderpath), "w");

            mmap_s.fput_one<row_type>(n_row_);
            mmap_s.fput_one<col_type>(n_col_);

            vals_.save_to_mmap_store(mmap_s);
            vals_lens_.save_to_mmap_store(mmap_s);
            vals_starts_.save_to_mmap_store(mmap_s);

            mmap_s.close();
        }

        void load(const std::string& folderpath, const bool lazy_load) {
            mmap_store_.open(mmap_file_name(folderpath), lazy_load?"r_lazy":"r");

            n_row_ = mmap_store_.fget_one<row_type>();
            n_col_ = mmap_store_.fget_one<col_type>();

            vals_.load_from_mmap_store(mmap_store_);
            vals_lens_.load_from_mmap_store(mmap_store_);
            vals_starts_.load_from_mmap_store(mmap_store_);
        }


    private:
        row_type n_row_;
        col_type n_col_;
        mmap_util::MmapableVector<char> vals_;  // Concatenated big string
        mmap_util::MmapableVector<bytes_len_type> vals_lens_;  // Length for each string
        mmap_util::MmapableVector<row_type> vals_starts_;  // Start for each string in the concatenated big string

        pecos::mmap_util::MmapStore mmap_store_;

        inline std::string mmap_file_name(const std::string& folderpath) const {
            return folderpath + "/string_2d.mmap_store";
        }
};

} // end namespace mmap_valstore
} // end namespace pecos

#endif  // end of __MMAP_VALSTORE_H__
