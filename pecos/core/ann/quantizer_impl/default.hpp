/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
#include "common.hpp"

namespace pecos {

namespace ann {

    struct ProductQuantizer4Bits : ProductQuantizer4BitsBase {
        void pack_codebook_for_inference() {
            pack_codebook_for_inference_default();
        }

        void pad_parameters(index_type& max_degree, size_t& code_dimension) {
            pad_parameters_default(max_degree, code_dimension);
        }

        inline void approximate_neighbor_group_distance(size_t neighbor_size, float* ds, const char* neighbor_codes, uint8_t* lut_ptr, float scale, float bias) const {
            approximate_neighbor_group_distance_default(neighbor_size, ds, neighbor_codes, lut_ptr, scale, bias);
        }

        inline void setup_lut(float* query, uint8_t* lut_ptr, float& scale, float& bias) const {
            setup_lut_default(query, lut_ptr, scale, bias);
        }
    };

}  // end of namespace ann
}  // end of namespace pecos

