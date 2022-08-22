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

#ifndef __MMAP_UTIL_H__
#define __MMAP_UTIL_H__

#define __PTR_ALIGN_BYTES 8 // Bytes to align all pointers to in a mmap file

#include <cmath>
#include <fcntl.h>
#include <stdexcept>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "file_util.hpp"


namespace pecos {

namespace mmap_util {

// Pad file with 0
void _pad_file(FILE * fp, const size_t n_padded_bytes) {
    if (n_padded_bytes <= 0) {
        return;
    }
    const char dummy = '\0';
    fwrite(&dummy, sizeof(dummy), n_padded_bytes, fp);
}

/*
 * Metadata for memory-mapped file
 */
class MmapFileMetadata {
    public:
        MmapFileMetadata() : _n_arr(0) {}

        inline size_t n_arr() const {
            return _n_arr;
        }

        inline uint64_t arr_elem_size(const size_t idx) const {
            return _arr_elem_size[idx];
        }

        inline uint64_t arr_len(const size_t idx) const {
            return _arr_len[idx];
        }

        inline uint64_t arr_offset(const size_t idx) const {
            return _arr_offset[idx];
        }

        // Append an array's metadata, array ptr is aligned in file
        void aligned_append(const uint64_t elem_size, const uint64_t len, size_t & n_padded_bytes) {
            // align offset
            uint64_t file_end = 0;
            if (_n_arr != 0) {
                file_end = _arr_offset.back() + _arr_elem_size.back() * _arr_len.back();
            }
            uint64_t aligned_offset = std::ceil(static_cast<double>(file_end) / __PTR_ALIGN_BYTES) * __PTR_ALIGN_BYTES;

            _n_arr++;
            _arr_elem_size.push_back(elem_size);
            _arr_len.push_back(len);
            _arr_offset.push_back(aligned_offset);

            // return number of padded bytes
            uint64_t n_padded_bytes_64 = aligned_offset - file_end;
            n_padded_bytes = static_cast<size_t>(n_padded_bytes_64);
        }

        // Load metadata from file
        void load(const std::string & file_name) {
            // Open file and seek metadata position
            FILE * fp = fopen(file_name.c_str(), "rb");
            if (!fp) {
                throw std::runtime_error("Load metadata: Open file failed.");
            }
            fseek(fp, -_MAX_METADATA_SIZE, SEEK_END);

            // check signature
            _check_signature(fp);

            // check endian
            _check_endian_type(fp);

            // load metadata
            if (fread(&(_n_arr), sizeof(_n_arr), 1, fp) != 1) {
                throw std::runtime_error("Read n_arr failed.");
            }
            if (_n_arr <= 0) {
                throw std::runtime_error("Should get positive n_arr, got: " + std::to_string(_n_arr));
            }
            _load_vec(fp, _arr_elem_size, _n_arr, "arr_elem_size");
            _load_vec(fp, _arr_len, _n_arr, "arr_len");
            _load_vec(fp, _arr_offset, _n_arr, "arr_offset");

            // Close file
            fclose(fp);
        }

        // Dump metadata into file
        void dump(FILE * fp) {
            // check for empty case
            if (_n_arr == 0) {
                throw std::runtime_error("Cannot dump empty MmapFileMetadata.");
            }

            // dump signature
            _dump_signature(fp);

            // dump endian
            _dump_endian_type(fp);

            // dump metadata
            fwrite(&(_n_arr), sizeof(_n_arr), 1, fp);
            _dump_vec(fp, _arr_elem_size);
            _dump_vec(fp, _arr_len);
            _dump_vec(fp, _arr_offset);

            // Pad metadata to _MAX_METADATA_SIZE
            size_t n_padded_bytes = _MAX_METADATA_SIZE - _get_metadata_size(_n_arr);
            if (n_padded_bytes < 0) {
                throw std::runtime_error("Written Metadata size: " + std::to_string(_get_metadata_size(_n_arr)) + " exceeds MAX_METADATA_SIZE, please check.");
            }
            _pad_file(fp, n_padded_bytes);
        }

    private:
        // Signature: "MMAP"
        static constexpr const char * _SIGNATURE = "MMAP";
        static const size_t _SIGN_LEN = 4;

        // Max number of arrays in mmap file
        static const size_t _MAX_N_ARR = 128;

        // Preset metadata size>=_get_metadata_size(_MAX_N_ARR), containing information of all arrays, written at end of file
        static const size_t _MAX_METADATA_SIZE = 4096;

        // Metadata
        size_t _n_arr;
        std::vector<uint64_t> _arr_elem_size;
        std::vector<uint64_t> _arr_len;
        std::vector<uint64_t> _arr_offset;


        void _dump_signature(FILE * fp) {
            fwrite(_SIGNATURE, sizeof(char), _SIGN_LEN, fp);
        }

        void _check_signature(FILE * fp) {
            std::string loaded_sign(_SIGN_LEN, '\0');
            if (fread(&(loaded_sign[0]), sizeof(char), _SIGN_LEN, fp) != _SIGN_LEN) {
                throw std::runtime_error("Read metadata signature failed.");
            }
            if (loaded_sign != std::string(_SIGNATURE)) {
                throw std::runtime_error("File is not in mmap format. Got signature: " + loaded_sign);
            }
        }

        // Dump machine endian type to file
        void _dump_endian_type(FILE * fp) {
            char endian_type = file_util::runtime();
            fwrite(&(endian_type), sizeof(endian_type), 1, fp);
        }

        // Check whether the dumped endian type (read at FILE pointer) is the same with machine endian type
        void _check_endian_type(FILE * fp) {
            char endian_type;
            if (fread(&(endian_type), sizeof(endian_type), 1, fp) != 1) {
                throw std::runtime_error("Read endian type failed.");
            }
            if (file_util::different_from_runtime(endian_type)) {
                throw std::runtime_error("Machine endian type is different from saved data, cannot memory-map load.");
            }
        }

        void _load_vec(FILE * fp, std::vector<uint64_t> & vec, const size_t vec_len, const std::string & vec_name) {
            vec.resize(vec_len);
            if (fread(vec.data(), sizeof(vec[0]), vec.size(), fp) != vec.size()) {
                throw std::runtime_error("Read " + vec_name + " failed.");
            }
        }

        void _dump_vec(FILE * fp, std::vector<uint64_t> & vec) {
            fwrite(vec.data(), sizeof(vec[0]), vec.size(), fp);
        }

        size_t _get_metadata_size(const size_t n_arr) const {
            // signature (4 * char) + endian type (char) + n_arr (size_t) + arr infos (3 * n_arr * (uint64_t))
            return sizeof(char) * _SIGN_LEN + sizeof(char) + sizeof(size_t) + 3 * n_arr * sizeof(uint64_t);
        }
}; // end class MmapFileMetadata


/*
 * Class to load a file in memory-mapped format.
 */
class MemoryMappedFile {
    public:
        MemoryMappedFile() :
            _iter(0),
            _mmap_size(0),
            _mmap_ptr(nullptr) {
        }

        ~MemoryMappedFile() {
            _free_mmap(); // This need a separate function cuz C++11 raises warning against exception in destructor
        }

        // load an instance from a file
        void load(const std::string & file_name, const bool pre_load) {
            // Free previous mmap
            _free_mmap();

            _metadata.load(file_name);
            _load_mmap(file_name, pre_load);
            _assign_mmap_arr_ptrs();

            // Initialize array iterator
            _iter = 0;
        }

        // Get the array pointer at current iterator, the parameter `arr_len` is only for sanity check
        template <typename T>
        T * get_arr(const uint64_t arr_len) {
            // Metadata Sanity checks
            if (_iter >= _metadata.n_arr()) {
                throw std::runtime_error("Already got all arrays, please check save/load implementation");
            }
            auto cur_arr_elem_size = _metadata.arr_elem_size(_iter);
            if (cur_arr_elem_size != sizeof(T)) {
                throw std::runtime_error("Should get array element size: " + std::to_string(cur_arr_elem_size) + ", but user requests: " + std::to_string(sizeof(T)));
            }
            auto cur_arr_len = _metadata.arr_len(_iter);
            if (cur_arr_len != arr_len) {
                throw std::runtime_error("Should get array length: " + std::to_string(cur_arr_len) + ", but user requests: " + std::to_string(arr_len));
            }

            // Cast array pointer
            T * cur_arr_ptr = reinterpret_cast< T * >(_mmap_arr_ptr[_iter]);

            // Move forward array iterator
            _iter++;

            return cur_arr_ptr;
        }

        // Check if all arrays have been retrieved
        void check_all_arrs_retrieved() {
            if (_iter != _metadata.n_arr()) {
                throw std::runtime_error("Didn't retrieve all arrays. Number of arrays: " + std::to_string(_metadata.n_arr()) + ", retrieved: " + std::to_string(_iter));
            }
        }

    private:
        // Metadata
        MmapFileMetadata _metadata;

        // Internal array iterator
        size_t _iter;

        // mmap pointers
        uint64_t _mmap_size;
        void * _mmap_ptr;
        std::vector<void *> _mmap_arr_ptr; // mmap pointer for each array


        void _load_mmap(const std::string & file_name, const bool pre_load) {
            // mmap flag
            int mmap_flags = MAP_SHARED;
            if (pre_load) { // pre-fault all pages to load them into memory
                mmap_flags |= MAP_POPULATE;
            }

            // Open file
            int fd = open(file_name.c_str(), O_RDONLY);
            if (fd == -1) {
                throw std::runtime_error("Load Mmap file: Open file failed.");
            }

            // File size
            struct stat file_stat;
            fstat(fd, &file_stat);
            _mmap_size = file_stat.st_size;
            if (_mmap_size <= 0) {
                throw std::runtime_error("Memory mapped file size should be positive, got: " + std::to_string(_mmap_size));
            }

            // mmap
            _mmap_ptr = mmap(NULL, _mmap_size, PROT_READ, mmap_flags, fd, 0);
            if (_mmap_ptr == MAP_FAILED) {
                throw std::runtime_error("Memory map failed.");
            }

            // Close file
            if (close(fd) < 0) {
                throw std::runtime_error("Close file failed.");
            }
        }

        void _assign_mmap_arr_ptrs() {
            _mmap_arr_ptr.resize(_metadata.n_arr());
            for (size_t idx = 0; idx < _metadata.n_arr(); ++idx) {
                char * cur_arr_ptr = reinterpret_cast< char * >(_mmap_ptr);
                _mmap_arr_ptr[idx] = reinterpret_cast< void * >(cur_arr_ptr + _metadata.arr_offset(idx));
            }
        }

        // Unmap the region
        void _free_mmap() {
            if (_mmap_ptr) {
                auto res = munmap(_mmap_ptr, _mmap_size);
                if (res == EINVAL) {
                    throw std::runtime_error("Free memory map failed.");
                }
                _mmap_size = 0;
                _mmap_arr_ptr.clear();
                _iter = 0;
            }
        }
}; // end class MemoryMappedFile


/*
 * Dump one or several arrays into a memory-mapped file format
 */
class DumpToMmapFile {
    public:
        DumpToMmapFile(const std::string & file_name) {
            _fp = fopen(file_name.c_str(), "wb");
            _check_fp_open();
        }

        ~DumpToMmapFile() {
            if (_fp) {
                fclose(_fp);
            }
        }

        // Dump a snapshot of the given array's memory into file
        // Note:
        // If the array is a struct array, memory snapshot could possibly be different across platform and compiler
        // due to data struct padding, and user should be aware of it.
        template <typename T>
        void dump_arr(const T * arr, const uint64_t arr_len) {
            _check_fp_open();

            if (arr_len <= 0) {
                throw std::runtime_error("Array length should be positive, got: " + std::to_string(arr_len));
            }

            size_t n_padded_bytes = 0;
            _metadata.aligned_append(sizeof(T), arr_len, n_padded_bytes);

            // Pad last array
            _pad_file(_fp, n_padded_bytes);

            // Dump array memory snapshot
            fwrite(arr, sizeof(T) * arr_len, 1, _fp);
        }

        // Wrapper for vector
        template <typename T>
        void dump_vec(const std::vector<T> & vec) {
            dump_arr(vec.data(), vec.size());
        }

        void finalize() {
            _check_fp_open();
            // Dump metadata
            _metadata.dump(_fp);

            fflush(_fp);
            fsync(fileno(_fp));
            if (fclose(_fp) != 0) {
                throw std::runtime_error("DumpToMmapFile finalize close file failed.");
            }
            _fp = nullptr;
        }

    private:
        MmapFileMetadata _metadata;
        FILE * _fp;

        void _check_fp_open() {
            if (_fp == nullptr) {
                throw std::runtime_error("File is not opened, cannot dump mmap file.");
            }
        }

}; // end class DumpMemoryMappedFile

} // end namespace mmap_util

} // end namespace pecos

#endif  // end of __MMAP_UTIL_H__
