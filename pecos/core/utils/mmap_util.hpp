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

/* Memory-mapped module version.
 * Must be incremented by 1 everytime when non-backward-compatible change is made. */
#define __MMAP_UTIL_VER 1


#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include "file_util.hpp"


namespace pecos {

namespace mmap_util {

namespace _details { // namespace for Module Private classes

/* Type constraint for mmap save/load.
 * Please beware, this is only a minimum requirement and not a guarantee that the type is platform-portable for mmap.
 * If the class/struct members need padding, please double-check mmap file portability on your own. */
template<class Type>
constexpr bool IsPlainOldData = std::is_trivially_copyable<Type>::value && std::is_standard_layout<Type>::value;

template<typename Type, typename Ret=bool>
using if_simple_serializable = std::enable_if_t<IsPlainOldData<Type>, Ret>;


/*
 * Module Private Class to store signature and metadata of all data blocks in mainbody for a memory-mapped file.
 */
class _MmapFileMetadata {
    public:
        struct MetaInfo{
            uint64_t offset; // Starting position of each block in the file
            uint64_t size; // Number of bytes of each block's data
            MetaInfo(uint64_t offset=0, uint64_t size=0) : offset(offset), size(size) {}
        };

        /* Save metadata to file.
         * Parameters:
         *      fp (FILE *) : File pointer.
         *          Must points to the end of an opened file and waits for dumping metadata.
         */
        void save(FILE * fp) const {
            uint64_t meta_offset = static_cast<uint64_t>(ftell(fp)); // Record metadata offset
            _Signature sig = _Signature(meta_offset);

            file_util::fput_one<uint64_t>(_info.size(), fp);
            file_util::fput_multiple<MetaInfo>(_info.data(), _info.size(), fp);

            sig.save(fp);
        }

        /* Load metadata from file.
         * Parameters:
         *      fp (FILE *) : File pointer.
         *          Load signature first, then rewind to load metadata.
         */
        void load(FILE * fp) {
            _Signature sig;
            sig.load(fp);

            fseek(fp, sig.meta_offset, SEEK_SET); // Rewind fp for loading metadata
            uint64_t n_block = file_util::fget_one<uint64_t>(fp);
            _info.resize(n_block);
            file_util::fget_multiple<MetaInfo>(_info.data(), n_block, fp);
        }

        /* Append a new block's metadata with starting position aligned to at least multiple of sizeof(int)
         * Parameters:
         *      size (uint64_t): Number of bytes for the new block's data.
         * Return:
         *      n_bytes_to_pad (uint64_t): Number of bytes to pad before saving the new block.
         */
        uint64_t aligned_append(const uint64_t size) {
            const uint64_t align_bytes = 16; // All blocks are aligned to multiple of the value

            uint64_t last_end = 0;
            if (_info.size() > 0) {
                auto last_info = _info.back();
                last_end = last_info.offset + last_info.size;
            }

            uint64_t n_bytes_to_pad = (align_bytes - (last_end % align_bytes)) % align_bytes;
            _info.emplace_back(last_end + n_bytes_to_pad, size);

            return n_bytes_to_pad;
        }

        /* Iterator for retrieving next block's offset/size and auto advance */
        MetaInfo next() {
            auto cur_iter = _iter;
            _iter += 1;
            return MetaInfo(_info[cur_iter].offset, _info[cur_iter].size);
        }

    private:
        /* Struct to store signature for various configuration info of a memory-mapped file */
        struct _Signature {
            const uint8_t SIG_SIZE = 16u;                              // Size of signature is 16 bytes
            // Signture data
            const uint8_t MAGIC[6] = {0x93u, 'P', 'E', 'C', 'O', 'S'}; // 6 bytes
            uint8_t endianness;                                        // 1 bytes
            const uint8_t version = __MMAP_UTIL_VER;                   // 1 bytes
            uint64_t meta_offset;                                      // 8 bytes

            _Signature(uint64_t meta_offset=0) : endianness(get_endianess()), meta_offset(meta_offset) {}

            /* Save signature to file.
             * Signature must be saved at the end of file,
             * i.e. save when everything else has already been saved. */
            void save(FILE *fp) const {
                file_util::fput_multiple<uint8_t>(&MAGIC[0], sizeof(MAGIC), fp);
                file_util::fput_one<uint8_t>(endianness, fp);
                file_util::fput_one<uint8_t>(version, fp);
                file_util::fput_one<uint64_t>(meta_offset, fp);
            }

            /* Load signature from last `SIG_SIZE` bytes of file. */
            void load(FILE *fp) {
                // Seek last `SIG_SIZE` bytes
                fseek(fp, 0 - static_cast<long>(SIG_SIZE), SEEK_END);
                // Check MAGIC
                uint8_t loaded_magic[sizeof(MAGIC)];
                file_util::fget_multiple<uint8_t>(&loaded_magic[0], sizeof(MAGIC), fp);
                if(std::memcmp(&MAGIC[0], &loaded_magic[0], sizeof(MAGIC)) != 0) {
                    throw std::runtime_error("File is not a valid PECOS MMAP file");
                }
                // Check endianness
                uint8_t loaded_endianness = file_util::fget_one<uint8_t>(fp);
                if(loaded_endianness != get_endianess()) {
                    throw std::runtime_error("Inconsistent endianness between runtime and the mmap file.");
                }
                // Check version
                uint8_t loaded_version = file_util::fget_one<uint8_t>(fp);
                if (loaded_version != version) {
                    throw std::runtime_error("Inconsistent version between code and the mmap file.");
                }
                // Load meta offset
                meta_offset = file_util::fget_one<uint64_t>(fp);
            }

            /* Obtain current machine endianness */
            uint8_t get_endianess() const {
                return static_cast<uint8_t>(file_util::runtime());
            }
        };

        std::vector<MetaInfo> _info; // offset and size of each block in the file
        uint64_t _iter = 0; // Iterator index for retrieving metadata info in blocks order
}; // end class _MmapFileMetadata


/*
 * Module Private Class to load data from a memory-mapped file.
 */
class _MmapFileLoad {
    public:
        /* Constructor loads from memory-mapped file name
         * Parameters:
         *      file_name (const std::string &): Name of file to load from.
         *      pre_load (const bool): Whether to pre-fault all pages into memory before accessing.
         */
        _MmapFileLoad(const std::string & file_name, const bool pre_load) {
            // Load metadata first
            FILE * fp = fopen(file_name.c_str(), "rb");
            if (!fp) {
                throw std::runtime_error("Load metadata: Open file failed.");
            }
            _metadata.load(fp);
            if (fclose(fp) != 0) {
                throw std::runtime_error("Load metadata: Close file failed.");
            }

            // Load mmap
            int fd = open(file_name.c_str(), O_RDONLY);
            if (fd == -1) {
                throw std::runtime_error("Load Mmap file: Open file failed.");
            }
            _load_mmap(fd, pre_load);
            if (close(fd) < 0) {
                throw std::runtime_error("Load Mmap file: Close file failed.");
            }
        }
        /* Destructor frees the memory-mapped region */
        ~_MmapFileLoad() { _free_mmap(); }

        /* Load a simple-serializable vector(array)
         * Parameters:
         *      n_elements (uint64_t): Number of elements to load for the array.
         */
        template<class T, class TT=T, if_simple_serializable<TT> = true>
        T * fget_multiple(uint64_t n_elements) {
            auto meta_info = _metadata.next();

            // Verification
            if (n_elements * static_cast<uint64_t>(sizeof(T)) != meta_info.size) {
                throw std::runtime_error(
                    "This block contains " + std::to_string(meta_info.size) +
                    " bytes data, retrieving " + std::to_string(n_elements * static_cast<uint64_t>(sizeof(T))) +
                    " bytes not equal. Please double check.");
            }

            return reinterpret_cast< T * >(reinterpret_cast< char * >(_mmap_ptr) + meta_info.offset);
        }

        /* Load one element */
        template<class T>
        T fget_one() { // Call fget_multiple
            return * fget_multiple<T>(1u);
        }

    private:
        _MmapFileMetadata _metadata; // Metadata
        void * _mmap_ptr = nullptr; // Memory-mapped pointer
        uint64_t _mmap_size = 0;// Memory-mapped file size

        /* Create memory-mapped region */
        void _load_mmap(const int fd, const bool pre_load) {
            // Get file size
            struct stat file_stat;
            fstat(fd, &file_stat);
            _mmap_size = file_stat.st_size;
            if (_mmap_size <= 0) {
                throw std::runtime_error("Memory mapped file size should be positive, got: " + std::to_string(_mmap_size));
            }

            // Creat mmap
            int mmap_flags = MAP_SHARED;
            if (pre_load) { // pre-fault all pages to load them into memory
                mmap_flags |= MAP_POPULATE;
            }
            _mmap_ptr = mmap(NULL, _mmap_size, PROT_READ, mmap_flags, fd, 0);
            if (_mmap_ptr == MAP_FAILED) {
                throw std::runtime_error("Memory map failed.");
            }
        }

        /* Free memory-mapped region */
        void _free_mmap() {
            if (_mmap_ptr) {
                auto res = munmap(_mmap_ptr, _mmap_size);
                if (res == EINVAL) {
                    throw std::runtime_error("Free memory map failed.");
                }
                _mmap_ptr = nullptr;
                _mmap_size = 0;
            }
        }
}; // end class MmapFileLoad


/*
 * Module Private Class to save a series of data, along with metadata and signature into a memory-mapped file.
 */
class _MmapFileSave {
    public:
        /* Constructor to open a file and save to given filename */
        _MmapFileSave(const std::string & file_name) {
            _fp = fopen(file_name.c_str(), "wb");
            if (!_fp) {
                throw std::runtime_error("MmapFileSave: Open file failed.");
            }
        }

        /* Destructor to save metadata and close opened file after save */
        ~_MmapFileSave() { _finalize(); }

        /* Save a simple-serializable vector(array) */
        template<class T, class TT=T, if_simple_serializable<TT> = true>
        void fput_multiple(const T * src, const uint64_t n_elements) {
            auto n_bytes_to_pad = _metadata.aligned_append(sizeof(T) * n_elements);

            // Pad previous block end with dummy bytes
            const char dummy = '\0';
            for(auto i = 0u; i < n_bytes_to_pad; i++){
                file_util::fput_one(dummy, _fp);
            }

            // Save array
            file_util::fput_multiple<T>(src, n_elements, _fp);
        }

        /* Save one element */
        template<class T>
        void fput_one(const T& src) { // Call fput_multiple
            fput_multiple<T>(&src, 1u);
        }

    private:
        _MmapFileMetadata _metadata; // Metadata
        FILE * _fp; // File pointer

        /* Save metadata to end of file after all data blocks have been saveed.
         * Also, file pointer is closed at the end.
         */
        void _finalize() {
            _metadata.save(_fp);
            if (fclose(_fp) != 0) {
                throw std::runtime_error("MmapFileSave: Close file failed.");
            }
        }
}; // end class MmapFileSave

} // end namespace _details

/*
 * Wrapper of _details::_MmapFileSave and _details::_MmapFileLoad to facilitate usage
 */
class MmapFile {
    public:
        // Disable copy/move for single ownership semantics
        MmapFile(const MmapFile&) = delete;
        MmapFile(MmapFile&&);
        MmapFile& operator=(const MmapFile&) = delete;
        MmapFile& operator=(MmapFile&&);

        /* Dummy Constructor */
        MmapFile() { }
        /* Destructor to delete any existing instance */
        ~MmapFile() { close(); }

        /* Open a filename with given mode read/write and options
         * Parameters:
         *      file_name (string): Path of the file
         *      mode_str (string): Open for read ("r"), read with pre-fault all pages ("rp"), or for write("w")
         */
        void open(const std::string & file_name, const std::string & mode_str) {
            if (_mode != _Mode::UNINIT) {
                throw std::runtime_error("Should close existing file before open new one.");
            }
            if (mode_str == "r") {
                _mmap_r = new _details::_MmapFileLoad(file_name, false);
                _mode = _Mode::READONLY;
            }
            else if (mode_str == "rp") { // pre-load all pages
                _mmap_r = new _details::_MmapFileLoad(file_name, true);
                _mode = _Mode::READONLY;
            }
            else if (mode_str == "w") {
                _mmap_w = new _details::_MmapFileSave(file_name);
                _mode = _Mode::WRITEONLY;
            }
            else {
                throw std::runtime_error("Unrecogonized mode. Should be either 'r' or 'w'.");
            }
        }

        /* Close any opened files
         * If open for read, should NOT close until the data of the file is not visited anymore.
         */
        void close() {
            if (_mode == _Mode::UNINIT) {
                return;
            }
            else if (_mode == _Mode::READONLY) {
                delete _mmap_r;
                _mmap_r = nullptr;
            }
            else if (_mode == _Mode::WRITEONLY) {
                delete _mmap_w;
                _mmap_w = nullptr;
            }
            _mode = _Mode::UNINIT;
        }

        /* Expose get/put functions */
        template<class T>
        T fget_one() {
            _check_get();
            return _mmap_r->fget_one<T>();
        }

        template<class T>
        T* fget_multiple(const uint64_t n_elements) {
            _check_get();
            return _mmap_r->fget_multiple<T>(n_elements);
        }

        template<class T>
        void fput_one(const T& src) {
            _check_put();
            _mmap_w->fput_one<T>(src);
        }

        template<class T>
        void fput_multiple(const T * src, const uint64_t n_elements) {
            _check_put();
            _mmap_w->fput_multiple<T>(src, n_elements);
        }

    private:
        enum class _Mode {
            UNINIT = 0,             // Uninitialized
            READONLY = 1,           // "r"
            WRITEONLY = 2,          // "w"
        };

        _details::_MmapFileSave * _mmap_w = nullptr; // Used for write mode
        _details::_MmapFileLoad * _mmap_r = nullptr; // Used for read mode
        _Mode _mode = _Mode::UNINIT; // Indicator for current opened mode

        /* Check if opened as read mode for calling get functions */
        void _check_get() {
            if (!_mmap_r) {
                throw std::runtime_error("Not opened for read mode, cannot call get.");
            }
        }

        /* Check if opened as write mode for calling put functions */
        void _check_put() {
            if (!_mmap_w) {
                throw std::runtime_error("Not opened for write mode, cannot call put.");
            }
        }
}; // end class MmapFile


/* Extended Vector class that can be used as a std::vector or as a memory-view from part of mmap file
 * For std::vector case, it own the memory for data storage.
 * For mmap view case, it does not own any memory, but serve as a view for a piece of memory owned by MmapFile.
 * By default, it is initialized as empty std::vector that can be resized or loaded as mmap view.
 * Once loaded as mmap view, it cannot go back to std::vector case unless clear() is called.
 */
template<class T, class TT = T, _details::if_simple_serializable<TT> = true>
class MmapableVector {
    public:
        MmapableVector(uint64_t size=0) { resize(size); }

        /* Functions to match std::vector interface */
        uint64_t size() const { return _size; }

        const T* data() const { return _data; }

        T& operator[](uint64_t idx) { return _data[idx]; }
        const T& operator[](uint64_t idx) const { return _data[idx]; }

        bool empty() const { return static_cast<bool> (_size == 0); }

        void resize(uint64_t new_size) {
            if(empty() || _is_self_allocated()) { // Valid for empty or self-allocated vector
                _store.resize(new_size);
                _data = _store.data();
                _size = _store.size();
            }
            else { // raises error for mmap view
                throw std::runtime_error("Cannot resize for mmap view case.");
            }
        }

        void clear() { // Clear any storage
            _size = 0;
            _data = nullptr;
            _store.clear();
        }

        /* Mmap save & load with MmapFile */
        void save_mmap(MmapFile & mmap_f) const {
            mmap_f.fput_one<uint64_t>(_size);
            mmap_f.fput_multiple<T>(_data, _size);
        }

        void load_mmap(MmapFile & mmap_f) {
            clear(); // Clean any previous storage
            _size = mmap_f.fget_one<uint64_t>();
            _data = mmap_f.fget_multiple<T>(_size);
        }

    private:
        uint64_t _size = 0; // Number of elements of the data
        T* _data = nullptr; // Pointer to actual data
        std::vector<T> _store; // Actual data storage for self-allocated case

        /* Whether data storage is non-empty self-allocated.
         * True indicates non-empty vector; False indicates either empty or mmap view. */
        bool _is_self_allocated() const {
            return static_cast<bool> (_size != 0 && _data == _store.data());
        }

}; // end class MmapableVector

} // end namespace mmap_util

} // end namespace pecos

#endif  // end of __MMAP_UTIL_H__
