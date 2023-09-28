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


namespace pecos {
namespace mmap_valstore {

typedef uint64_t row_type;
typedef uint32_t col_type;


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

        void get_submatrix(const uint32_t n_sub_row, const uint32_t n_sub_col, const row_type* sub_rows, const col_type* sub_cols, value_type* ret, const int threads=1) {
            #pragma omp parallel for schedule(static, 1) num_threads(threads)
            for (uint32_t i=0; i<n_sub_row; ++i) {
                for (uint32_t j=0; j<n_sub_col; ++j) {
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



} // end namespace mmap_valstore
} // end namespace pecos

#endif  // end of __MMAP_VALSTORE_H__
