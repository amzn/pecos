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

namespace details_ { // namespace for Module Private classes

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
class MmapMetadata {
    public:
        struct MetaInfo{
            uint64_t offset; // Starting position of each block in the file
            uint64_t size; // Number of bytes of each block's data
            MetaInfo(uint64_t offset=0, uint64_t size=0) : offset(offset), size(size) {}
        };

        /* Save metadata to file.
         * Parameters:
         *      fp (FILE*) : File pointer.
         *          Must points to the end of an opened file and waits for dumping metadata.
         */
        void save(FILE* fp) const {
            uint64_t meta_offset = static_cast<uint64_t>(ftell(fp)); // Record metadata offset
            Signature sig = Signature(meta_offset);

            file_util::fput_one<uint64_t>(info_.size(), fp);
            file_util::fput_multiple<MetaInfo>(info_.data(), info_.size(), fp);

            sig.save(fp);
        }

        /* Load metadata from file.
         * Parameters:
         *      fp (FILE*) : File pointer.
         *          Load signature first, then rewind to load metadata.
         */
        void load(FILE* fp) {
            Signature sig;
            sig.load(fp);

            fseek(fp, sig.meta_offset, SEEK_SET); // Rewind fp for loading metadata
            uint64_t n_block = file_util::fget_one<uint64_t>(fp);
            info_.resize(n_block);
            file_util::fget_multiple<MetaInfo>(info_.data(), n_block, fp);
        }

        /* Get number of bytes to pad in order to align new block's beginning to multiple of N_ALIGN_BYTES_
         * Return:
         *      n_bytes_to_pad (uint64_t): Number of bytes to pad before saving the new block.
         */
        uint64_t get_n_bytes_padding_to_align() const {
            uint64_t last_end = get_last_block_end_();
            uint64_t n_bytes_to_pad = (N_ALIGN_BYTES_ - (last_end % N_ALIGN_BYTES_)) % N_ALIGN_BYTES_;
            return n_bytes_to_pad;
        }

        /* Append a new block's metadata giving size. Offset is auto calculated.
         * Parameters:
         *      size (uint64_t): Number of bytes for the new block's data.
         */
        void append(const uint64_t size) {
            uint64_t last_end = get_last_block_end_();
            uint64_t n_bytes_to_pad = get_n_bytes_padding_to_align();
            info_.emplace_back(last_end + n_bytes_to_pad, size);
        }

        /* Iterator for retrieving next block's offset/size and auto advance */
        MetaInfo next() {
            auto cur_iter = iter_;
            iter_ += 1;
            return MetaInfo(info_[cur_iter].offset, info_[cur_iter].size);
        }

    private:
        /* Struct to store signature for various configuration info of a memory-mapped file */
        struct Signature {
            const uint8_t SIG_SIZE = 16u;                              // Size of signature is 16 bytes
            // Signture data
            const uint8_t MAGIC[6] = {0x93u, 'P', 'E', 'C', 'O', 'S'}; // 6 bytes
            uint8_t endianness;                                        // 1 bytes
            const uint8_t version = __MMAP_UTIL_VER;                   // 1 bytes
            uint64_t meta_offset;                                      // 8 bytes

            Signature(uint64_t meta_offset=0) : endianness(get_endianess()), meta_offset(meta_offset) {}

            /* Save signature to file.
             * Signature must be saved at the end of file,
             * i.e. save when everything else has already been saved. */
            void save(FILE* fp) const {
                file_util::fput_multiple<uint8_t>(&MAGIC[0], sizeof(MAGIC), fp);
                file_util::fput_one<uint8_t>(endianness, fp);
                file_util::fput_one<uint8_t>(version, fp);
                file_util::fput_one<uint64_t>(meta_offset, fp);
            }

            /* Load signature from last `SIG_SIZE` bytes of file. */
            void load(FILE* fp) {
                // Seek last `SIG_SIZE` bytes
                fseek(fp, 0 - static_cast<long>(SIG_SIZE), SEEK_END);
                // Check MAGIC
                uint8_t loaded_magic[sizeof(MAGIC)];
                file_util::fget_multiple<uint8_t>(&loaded_magic[0], sizeof(MAGIC), fp);
                if(std::memcmp(&MAGIC[0], &loaded_magic[0], sizeof(MAGIC)) != 0) {
                    throw std::runtime_error("File is not a valid PECOS MMAP file.");
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

        const uint64_t N_ALIGN_BYTES_ = 16; // All blocks are aligned to multiple of the value
        std::vector<MetaInfo> info_; // offset and size of each block in the file
        uint64_t iter_ = 0; // Iterator index for retrieving metadata info in blocks order

        /* Return end of last block in file */
        uint64_t get_last_block_end_() const {
            uint64_t last_end = 0;
            if (info_.size() > 0) {
                auto last_info = info_.back();
                last_end = last_info.offset + last_info.size;
            }
            return last_end;
        }
}; // end class MmapMetadata


/*
 * Module Private Class to load data from a memory-mapped file.
 */
class MmapStoreLoad {
    public:
        /* Constructor loads from memory-mapped file name
         * Parameters:
         *      file_name (const std::string&): Name of file to load from.
         *      lazy_load (const bool): If false, pre-fault all pages into memory before accessing.
         */
        MmapStoreLoad(const std::string& file_name, const bool lazy_load) {
            // Load metadata first
            FILE* fp = fopen(file_name.c_str(), "rb");
            if (!fp) {
                throw std::runtime_error("Load metadata: Open file failed.");
            }
            metadata_.load(fp);
            if (fclose(fp) != 0) {
                throw std::runtime_error("Load metadata: Close file failed.");
            }

            // Load mmap
            int fd = open(file_name.c_str(), O_RDONLY);
            if (fd == -1) {
                throw std::runtime_error("Load Mmap file: Open file failed.");
            }
            load_mmap_(fd, lazy_load);
            if (close(fd) < 0) {
                throw std::runtime_error("Load Mmap file: Close file failed.");
            }
        }
        /* Destructor frees the memory-mapped region */
        ~MmapStoreLoad() { free_mmap_(); }

        /* Load a simple-serializable vector(array)
         * Parameters:
         *      n_elements (uint64_t): Number of elements to load for the array.
         */
        template<class T, class TT=T, if_simple_serializable<TT> = true>
        T* fget_multiple(uint64_t n_elements) {
            auto meta_info = metadata_.next();

            // Verification
            if (n_elements * static_cast<uint64_t>(sizeof(T)) != meta_info.size) {
                throw std::runtime_error(
                    "This block contains " + std::to_string(meta_info.size) +
                    " bytes data, retrieving " + std::to_string(n_elements * static_cast<uint64_t>(sizeof(T))) +
                    " bytes not equal. Please double check.");
            }

            return reinterpret_cast< T* >(reinterpret_cast< char * >(mmap_ptr_) + meta_info.offset);
        }

        /* Load one element */
        template<class T, class TT=T, if_simple_serializable<TT> = true>
        T fget_one() { // Call fget_multiple
            return *fget_multiple<T>(1u);
        }

    private:
        MmapMetadata metadata_; // Metadata
        void* mmap_ptr_ = nullptr; // Memory-mapped pointer
        uint64_t mmap_size_ = 0; // Memory-mapped file size

        /* Create memory-mapped region */
        void load_mmap_(const int fd, const bool lazy_load) {
            // Get file size
            struct stat file_stat;
            fstat(fd, &file_stat);
            mmap_size_ = file_stat.st_size;
            if (mmap_size_ <= 0) {
                throw std::runtime_error("Memory mapped file size should be positive, got: " + std::to_string(mmap_size_));
            }

            // Creat mmap
            int mmap_flags = MAP_SHARED;
            if (!lazy_load) { // pre-fault all pages to load them into memory
                mmap_flags |= MAP_POPULATE;
            }
            mmap_ptr_ = mmap(NULL, mmap_size_, PROT_READ, mmap_flags, fd, 0);
            if (mmap_ptr_ == MAP_FAILED) {
                throw std::runtime_error("Memory map failed.");
            }
        }

        /* Free memory-mapped region */
        void free_mmap_() {
            if (mmap_ptr_) {
                auto res = munmap(mmap_ptr_, mmap_size_);
                if (res == EINVAL) {
                    throw std::runtime_error("Free memory map failed.");
                }
                mmap_ptr_ = nullptr;
                mmap_size_ = 0;
            }
        }
}; // end class MmapStoreLoad


/*
 * Module Private Class to save a series of data, along with metadata and signature into a memory-mapped file.
 */
class MmapStoreSave {
    public:
        /* Constructor to open a file and save to given filename */
        MmapStoreSave(const std::string& file_name) {
            fp_ = fopen(file_name.c_str(), "wb");
            if (!fp_) {
                throw std::runtime_error("MmapStoreSave: Open file failed.");
            }
        }

        /* Destructor to save metadata and close opened file after save */
        ~MmapStoreSave() { finalize_(); }

        /* Save a simple-serializable vector(array) */
        template<class T, class TT=T, if_simple_serializable<TT> = true>
        void fput_multiple(const T* src, const uint64_t n_elements) {
            // Data size
            uint64_t size = sizeof(T) * n_elements;
            // Pad previous block with dummy bytes to make sure new block is aligned
            auto n_bytes_to_pad = metadata_.get_n_bytes_padding_to_align();
            const char dummy = '\0';
            for(auto i = 0u; i < n_bytes_to_pad; i++){
                file_util::fput_one(dummy, fp_);
            }
            // Save array
            metadata_.append(size);
            file_util::fput_multiple<T>(src, n_elements, fp_);
        }

        /* Save one element */
        template<class T, class TT=T, if_simple_serializable<TT> = true>
        void fput_one(const T& src) { // Call fput_multiple
            fput_multiple<T>(&src, 1u);
        }

    private:
        MmapMetadata metadata_; // Metadata
        FILE* fp_; // File pointer

        /* Save metadata to end of file after all data blocks have been saveed.
         * Also, file pointer is closed at the end.
         */
        void finalize_() {
            metadata_.save(fp_);
            if (fclose(fp_) != 0) {
                throw std::runtime_error("MmapStoreSave: Close file failed.");
            }
        }
}; // end class MmapStoreSave

} // end namespace details_


/*
 * Wrapper of details_::MmapStoreSave and details_::MmapStoreLoad to facilitate usage
 */
class MmapStore {
    public:
        // Disable copy/move for single ownership semantics
        MmapStore(const MmapStore&) = delete;
        MmapStore(MmapStore&&);
        MmapStore& operator=(const MmapStore&) = delete;
        MmapStore& operator=(MmapStore&&);

        /* Dummy Constructor */
        MmapStore() { }
        /* Destructor to delete any existing instance */
        ~MmapStore() { close(); }

        /* Open a filename with given mode read/write and options
         * Parameters:
         *      file_name (string): Path of the file
         *      mode_str (string): Open for following options:
         *          * read ("r") : pre-fault all pages
         *          * read with lazy load ("r_lazy") : only create mmap, loading when accessing
         *          * write ("w")
         */
        void open(const std::string& file_name, const std::string& mode_str) {
            if (mode_ != Mode::UNINIT) {
                throw std::runtime_error("Should close existing file before open new one.");
            }
            if (mode_str == "r") { // lazy_load=false, pre-load all pages
                mmap_r_ = new details_::MmapStoreLoad(file_name, false);
                mode_ = Mode::READONLY;
            } else if (mode_str == "r_lazy") { // lazy_load=true
                mmap_r_ = new details_::MmapStoreLoad(file_name, true);
                mode_ = Mode::READONLY;
            } else if (mode_str == "w") {
                mmap_w_ = new details_::MmapStoreSave(file_name);
                mode_ = Mode::WRITEONLY;
            } else {
                throw std::runtime_error("Unrecogonized mode. Should be either 'r', 'r_lazy' or 'w'.");
            }
        }

        /* Close any opened files
         * If open for read, should NOT close until the data of the file is not visited anymore.
         */
        void close() {
            if (mode_ == Mode::UNINIT) {
                return;
            } else if (mode_ == Mode::READONLY) {
                delete mmap_r_;
                mmap_r_ = nullptr;
            } else if (mode_ == Mode::WRITEONLY) {
                delete mmap_w_;
                mmap_w_ = nullptr;
            }
            mode_ = Mode::UNINIT;
        }

        /* Expose get/put functions */
        template<class T, class TT=T, details_::if_simple_serializable<TT> = true>
        T fget_one() {
            check_get_();
            return mmap_r_->fget_one<T>();
        }

        template<class T, class TT=T, details_::if_simple_serializable<TT> = true>
        T* fget_multiple(const uint64_t n_elements) {
            check_get_();
            return mmap_r_->fget_multiple<T>(n_elements);
        }

        template<class T, class TT=T, details_::if_simple_serializable<TT> = true>
        void fput_one(const T& src) {
            check_put_();
            mmap_w_->fput_one<T>(src);
        }

        template<class T, class TT=T, details_::if_simple_serializable<TT> = true>
        void fput_multiple(const T* src, const uint64_t n_elements) {
            check_put_();
            mmap_w_->fput_multiple<T>(src, n_elements);
        }

    private:
        enum class Mode {
            UNINIT = 0,             // Uninitialized
            READONLY = 1,           // "r"
            WRITEONLY = 2,          // "w"
        };

        details_::MmapStoreSave* mmap_w_ = nullptr; // Used for write mode
        details_::MmapStoreLoad* mmap_r_ = nullptr; // Used for read mode
        Mode mode_ = Mode::UNINIT; // Indicator for current opened mode

        /* Check if opened as read mode for calling get functions */
        void check_get_() {
            if (!mmap_r_) {
                throw std::runtime_error("Not opened for read mode, cannot call get.");
            }
        }

        /* Check if opened as write mode for calling put functions */
        void check_put_() {
            if (!mmap_w_) {
                throw std::runtime_error("Not opened for write mode, cannot call put.");
            }
        }
}; // end class MmapStore


/* Extended Vector class that can be used as a std::vector or as a memory-view from part of mmap file
 * For std::vector case, it own the memory for data storage.
 * For mmap view case, it does not own any memory, but serve as a view for a piece of memory owned by MmapStore.
 * By default, it is initialized as empty std::vector that can be resized or loaded as mmap view.
 * Once loaded as mmap view, it cannot go back to std::vector case unless clear() or convertion is called.
 */
template<class T, class TT = T, details_::if_simple_serializable<TT> = true>
class MmapableVector {
    public:
        /* Constructor */
        MmapableVector(uint64_t size=0, const T& value=T()) { resize(size, value); }

        /* Copy constructor and assignment */
        MmapableVector(const MmapableVector<T>& other): size_(other.size_), data_(other.data_) {
            if (other.is_self_allocated_()) { // non-empty vector
                store_ = other.store_;
                data_ = store_.data();
            }
        }

        MmapableVector& operator=(const MmapableVector<T>& other) {
            if(this != &other) {
                MmapableVector temp(other);
                swap_(*this, temp);
            }
            return *this;
        }

        /* Move Constructor and Assignment */
        MmapableVector(MmapableVector<T>&& other): MmapableVector() {
            swap_(*this, other);
        }

        MmapableVector& operator=(MmapableVector<T>&& other) {
            swap_(*this, other);
            return *this;
        }

        /* Functions to match std::vector interface */
        uint64_t size() const { return size_; }

        T* data() { return data_; }
        const T* data() const { return data_; }

        T& operator[](uint64_t idx) { return data_[idx]; }
        const T& operator[](uint64_t idx) const { return data_[idx]; }

        bool empty() const { return static_cast<bool> (size_ == 0); }

        void resize(uint64_t size, const T& value=T()) {
            if(empty() || is_self_allocated_()) { // Valid for empty or self-allocated vector
                store_.resize(size, value);
                data_ = store_.data();
                size_ = store_.size();
            } else { // raises error for mmap view
                throw std::runtime_error("Cannot resize for mmap view case.");
            }
        }

        void clear() { // Clear any storage
            size_ = 0;
            data_ = nullptr;
            store_.clear();
        }

        /* Mmap save/load with MmapStore */
        void save_to_mmap_store(MmapStore& mmap_s) const {
            mmap_s.fput_one<uint64_t>(size_);
            mmap_s.fput_multiple<T>(data_, size_);
        }

        void load_from_mmap_store(MmapStore& mmap_s) {
            if (is_self_allocated_()) { // raises error for non-empty self-allocated vector
                throw std::runtime_error("Cannot load for non-empty vector case.");
            }
            size_ = mmap_s.fget_one<uint64_t>();
            data_ = mmap_s.fget_multiple<T>(size_);
        }

        /* Convert (from mmap view) into self-allocated vector by copying data.
         * To be noted, this is only a shallow copy and only good for POD without pointer members. */
        void to_self_alloc_vec() {
            if (!is_self_allocated_()) {
                store_.resize(size_);
                for (uint64_t i = 0; i < size_; ++i) {
                    store_[i] = data_[i];
                }
                data_ = store_.data();
            }
        }

    private:
        uint64_t size_ = 0; // Number of elements of the data
        T* data_ = nullptr; // Pointer to data. The same as store_.data() for self-allocated vector case
        std::vector<T> store_; // Actual data storage for self-allocated vector case

        /* Whether data storage is non-empty self-allocated vector.
         * True indicates non-empty vector case; False indicates either empty or mmap view. */
        bool is_self_allocated_() const {
            return (!empty() && static_cast<bool> (data_ == store_.data()));
        }

        /* Swap function for copy and move assignments */
        void swap_(MmapableVector& lhs, MmapableVector& rhs) {
            std::swap(lhs.size_, rhs.size_);
            std::swap(lhs.data_, rhs.data_);
            std::swap(lhs.store_, rhs.store_);
        }

}; // end class MmapableVector

} // end namespace mmap_util

} // end namespace pecos

#endif  // end of __MMAP_UTIL_H__
