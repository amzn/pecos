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

#ifndef __MMAP_H__
#define __MMAP_H__

#define __ALIGN_BYTES 4

#include <fcntl.h>
#include <stdexcept>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include "file_util.hpp"

namespace pecos {

namespace mmap {

// Dump machine endian type to file
static void dump_endian_type(FILE *fp) {
    char endian_type = file_util::runtime();
    // For GCC vectorization optimization
    // Repeat bytes to align file pointers when loaded into memory
    for (size_t ii=0; ii<__ALIGN_BYTES; ++ii) {
        fwrite(&(endian_type), sizeof(endian_type), 1, fp);
    }
}

// Check whether the dumped endian type (read at FILE pointer) is the same with machine endian type
// Should be used pairly with `dump_endian_type`
static void check_endian_type(FILE *fp) {
    char endian_type[__ALIGN_BYTES];
    if (fread(&(endian_type), sizeof(endian_type[0]), __ALIGN_BYTES, fp) != __ALIGN_BYTES) {
        throw std::runtime_error("Read endian type failed.");
    }
    if (file_util::different_from_runtime(endian_type[0])) {
        throw std::runtime_error("Machine endian type is different from saved data, cannot memory-map load.");
    }
}

template <typename T>
class MemoryMappedArray {
    public:
        MemoryMappedArray() :
            _arr_ptr(nullptr),
            _arr_len(0),
            _mmap_ptr(nullptr),
            _mmap_bytes(0) {
        }
        ~MemoryMappedArray() {
            _free(); // This need a separate function cuz C++11 raises warning against exception in destructor
        }

        T * data() {
            return _arr_ptr;
        }

        size_t size() {
            return _arr_len;
        }

        // Dump a snapshot of the given array's memory into file
        // Filepointer `fp` will be automatically incremented.
        // Note:
        // If the array is a struct array, memory snapshot could possibly be different across platform and compiler
        // due to data struct padding, and user should be aware of it.
        static void dump_to_file(const T * arr, const size_t arr_len, FILE * fp) {
            if (arr_len <= 0) {
                throw std::runtime_error("Memory mapped array length should be positive, got: " + std::to_string(arr_len));
            }
            fwrite(arr, sizeof(T) * arr_len, 1, fp);
        }

        // Load memory-mapped array from file.
        // The target array should be saved with `dump_mmap_file` so that endian-type information is also saved.
        // Return read bytes that should be advanced in offset
        off64_t load(const size_t arr_len, const int fd, const off64_t offset, const bool pre_load) {
            if (arr_len <= 0) {
                throw std::runtime_error("Memory mapped array length should be positive, got: " + std::to_string(arr_len));
            }

            // Runtime get system pagesize
            const off64_t pagesize = sysconf(_SC_PAGESIZE);

            // Get the closest start of a page margin before offset
            // This is required by mmap that offset must be a multiple of pagesize
            const off64_t extra_bytes = offset % pagesize;
            const off64_t start = offset - extra_bytes;

            // Load memory map
            int mmap_flags = MAP_SHARED;
            if (pre_load) { // pre-fault all pages to load them into memory
                mmap_flags |= MAP_POPULATE;
            }
            off64_t arr_bytes = sizeof(T) * arr_len;
            off64_t mmap_bytes = arr_bytes + extra_bytes;
            void * mmap_ptr = mmap64(NULL, mmap_bytes, PROT_READ, mmap_flags, fd, start);
            if (mmap_ptr == MAP_FAILED) {
                throw std::runtime_error("Memory map failed.");
            }

            // Shift the pointer by extra bytes read
            char * arr_ptr = reinterpret_cast< char * > (mmap_ptr);
            arr_ptr += extra_bytes;

            // Assign member variables
            _arr_ptr = reinterpret_cast< T * > (arr_ptr);
            _arr_len = arr_len;
            _mmap_ptr = mmap_ptr;
            _mmap_bytes = mmap_bytes;

            // Return number of bytes that offset should advance
            return arr_bytes;
        }


    private:
        T * _arr_ptr; // pointer of memory-mapped array
        size_t _arr_len; // length of array

        void * _mmap_ptr; // pointer of start of actual mmap region
        size_t _mmap_bytes; // length of actual mmap region in bytes

        // Unmap the region
        void _free() {
            // if (_mmap_ptr) {
            //     auto res = munmap(_mmap_ptr, _mmap_bytes);
            //     if (res == EINVAL) {
            //         throw std::runtime_error("Free memory map failed.");
            //     }
            // }
        }
};

} // end namespace mmap

} // end namespace pecos

#endif  // end of __FILE_UTIL_H__
