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

#ifndef __PARALLEL_H__
#define  __PARALLEL_H__

#include <algorithm>
#include <iterator>
#include <numeric>
#include <omp.h>
// for __gnu_parallel
#include <parallel/algorithm>

namespace pecos {

    // ===== Thread Utility =====
    int set_threads(int threads) {
        if(threads == -1) {
            threads = omp_get_num_procs();
        }
        threads = std::min(threads, omp_get_num_procs());
        omp_set_num_threads(threads);
        return threads;
    }

    template<class InputIt, class OutputIt>
    void parallel_partial_sum(InputIt first, InputIt last, OutputIt out, int threads=1) {
        typedef typename std::iterator_traits<InputIt>::value_type value_type;
        typedef typename std::iterator_traits<InputIt>::difference_type difference_type;
        difference_type len = last - first;
        if(threads == 1 || len < threads) {
            std::partial_sum(first, last, out);
        } else {
            std::vector<value_type> offsets(threads + 1);
            difference_type workload = (len / threads) + (len % threads != 0);
#pragma omp parallel for schedule(static,1)
            for(int tid = 0; tid < threads; tid++) {
                auto local_first = first + std::min(tid * workload, len);
                auto local_last = first + std::min((tid + 1) * workload, len);
                auto local_len = std::distance(local_first, local_last);
                auto local_out = out + std::distance(first, local_first);
                if(local_len > 0) {
                    std::partial_sum(local_first, local_last, local_out);
                    offsets[tid + 1] = *(local_out + local_len - 1);
                }
            }

            std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

#pragma omp parallel for schedule(static,1) shared(offsets)
            for(int tid = 0; tid < threads; tid++) {
                auto local_first = out + std::min(tid * workload, len);
                auto local_last = out + std::min((tid + 1) * workload, len);
                auto local_len = std::distance(local_first, local_last);
                if(local_len > 0) {
                    std::for_each(
                        local_first,
                        local_last,
                        [&](value_type& x){ x += offsets[tid]; }
                    );
                }
            }
        }
    }

    template<class InputIt, class Compare>
    void parallel_sort(InputIt first, InputIt last, Compare comp, int threads=-1) {
        threads = set_threads(threads);
        typedef typename std::iterator_traits<InputIt>::difference_type difference_type;
        difference_type len = last - first;
        if(threads == 1 || len < threads) {
            std::sort(first, last, comp);
        } else {
            __gnu_parallel::multiway_mergesort_tag parallelism(threads);
            __gnu_parallel::sort(first, last, comp, parallelism);
        }
    }
} // end namespace pecos

#endif // end of __PARALLEL_H__
