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

#ifndef __RANDOM_H__
#define  __RANDOM_H__

#include <algorithm>
#include <limits>
#include <random>

namespace pecos {

    // ===== Random Number Generator: simulate the interface of python random module =====
    template<typename engine_t=std::mt19937>
    struct random_number_generator : public engine_t {
        typedef typename engine_t::result_type result_type;

        random_number_generator(unsigned seed=0): engine_t(seed) {}

        result_type randrange(result_type end=engine_t::max()) { return engine_t::operator()() % end; }
        template<class T=double, class T2=double> T uniform(T start=0.0, T2 end=1.0) {
            return std::uniform_real_distribution<T>(start, (T)end)(*this);
        }
        template<class T=double> T normal(T mean=0.0, T stddev=1.0) {
            return std::normal_distribution<T>(mean, stddev)(*this);
        }
        template<class T=int, class T2=T> T randint(T start=0, T2 end=std::numeric_limits<T>::max()) {
            return std::uniform_int_distribution<T>(start, end)(*this);
        }
        template<class RandIter> void shuffle(RandIter first, RandIter last) {
            std::shuffle(first, last, *this);
        }
    };

} // end namespace pecos

#endif // end of __RANDOM_H__
