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

#ifndef __MMAP_ANKERL_HASHMAP_H__
#define __MMAP_ANKERL_HASHMAP_H__

#include <omp.h>
#include "third_party/ankerl/unordered_dense.h"
#include "mmap_util.hpp"


namespace pecos {
namespace mmap_hashmap {

namespace details_ { // namespace for Module Private classes

/* For all memory-mappable vectors, when calling write functions,
assume the underlying storage is in memory, otherwise the code will fail. */

// Memory-mappable vector of std::pair<StrView, uint64_t> for Ankerl
// This vector takes/gets std::string_view as the key, but emplace back as the special mmap format StrView
class AnkerlStr2IntMmapableVector {
    template <bool IsConst>
    class iter_t;

    struct StrView {
        uint64_t offset;
        uint32_t len;

        StrView(uint64_t offset=0, uint32_t len=0) :
            offset(offset),
            len(len) { }
    };

    public:
        using key_type = std::string_view;
        using value_type = std::pair<StrView, uint64_t>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using allocator_type = std::allocator<value_type>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = allocator_type::pointer;
        using const_pointer = allocator_type::const_pointer;
        // Custom iterator
        using iterator = iter_t<false>;
        using const_iterator = iter_t<true>;

        AnkerlStr2IntMmapableVector() = default;
        AnkerlStr2IntMmapableVector(allocator_type alloc)
            : store_(alloc) {}

        value_type* data() { return data_; }
        const value_type* data() const { return data_; }

        value_type& operator[](uint64_t idx) { return data_[idx]; }
        const value_type& operator[](uint64_t idx) const { return data_[idx]; }

        /* Functions to match std::vector interface */
        auto get_allocator() { return store_.get_allocator(); }

        constexpr auto back() -> reference { return data_[size_ - 1]; }
        constexpr auto begin() -> iterator { return {data_}; }
        constexpr auto cbegin() -> const_iterator { return {data_}; }
        constexpr auto end() -> iterator { return {data_ + size_}; }
        constexpr auto end() const -> const_iterator { return {data_ + size_}; }
        constexpr auto cend() const -> const_iterator{ return {data_ + size_}; }

        // ----- Write funcs start -----
        void shrink_to_fit() { store_.shrink_to_fit(); }
        void reserve(size_t new_capacity) { store_.reserve(new_capacity); }

        /* Emplace string-like key and int value as std::pair<StrView, uint64_t>*/
        template <typename K, typename... Args>
        auto emplace_back(std::piecewise_construct_t, std::tuple<K> key, std::tuple<Args...> args) {
            // Extract key
            key_type k = std::get<0>(key);

            // Emplace back std::pair<StrView, uint64_t>
            auto eb_val = store_.emplace_back(
                std::piecewise_construct,
                std::forward_as_tuple(str_size_, k.size()),
                std::forward< std::tuple<Args...> >(args));

            // Append key string
            str_store_.insert(str_store_.end(), k.data(), k.data() + k.size());

            // Update pointers
            size_ = store_.size();
            data_ = store_.data();
            str_size_ = str_store_.size();
            str_data_ = str_store_.data();

            return eb_val;
        }

        void pop_back() {
            throw std::runtime_error("Not implemented for deletion");
        }
        // ----- Write funcs end -----

        size_type size() const { return size_; }

        bool empty() const { return static_cast<bool> (size_ == 0); }

        /* Get key for given member */
        key_type get_key(value_type const& vt) const {
            auto str_view = vt.first;
            return key_type(str_data_ + str_view.offset, str_view.len);
        }

        /* Mmap save/load with MmapStore */
        void save_to_mmap_store(pecos::mmap_util::MmapStore& mmap_s) const {
            mmap_s.fput_one<size_type>(size_);
            mmap_s.fput_one<size_type>(str_size_);
            mmap_s.fput_multiple<value_type>(data_, size_);
            mmap_s.fput_multiple<char>(str_data_, str_size_);
        }

        void load_from_mmap_store(pecos::mmap_util::MmapStore& mmap_s) {
            if (is_self_allocated_()) { // raises error for non-empty self-allocated vector
                throw std::runtime_error("Cannot load for non-empty vector case.");
            }
            size_ = mmap_s.fget_one<size_type>();
            str_size_ = mmap_s.fget_one<size_type>();
            data_ = mmap_s.fget_multiple<value_type>(size_);
            str_data_ = mmap_s.fget_multiple<char>(str_size_);
        }


    private:
        // Number of elements of the data
        size_type size_ = 0;
        size_type str_size_ = 0;

        // Pointer to data
        value_type* data_ = nullptr;
        char* str_data_ = nullptr;

        // Actual data storage for in-memory case
        std::vector<value_type> store_;
        std::vector<char> str_store_;

        /* Whether data storage is non-empty self-allocated vector.
         * True indicates non-empty vector case; False indicates either empty or mmap view. */
        bool is_self_allocated_() const {
            return static_cast<bool> (store_.size() > 0);
        }

        /**
         * Iterator class doubles as const_iterator and iterator
         */
        template <bool IsConst>
        class iter_t {
            using ptr_t = typename std::conditional_t<IsConst,
                AnkerlStr2IntMmapableVector::const_pointer, AnkerlStr2IntMmapableVector::pointer>;
            ptr_t iter_data_{};

            template <bool B>
            friend class iter_t;

            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type = AnkerlStr2IntMmapableVector::difference_type;
                using value_type = AnkerlStr2IntMmapableVector::value_type;
                using reference = typename std::conditional_t<IsConst,
                    value_type const&, value_type&>;
                using pointer = typename std::conditional_t<IsConst,
                    AnkerlStr2IntMmapableVector::const_pointer, AnkerlStr2IntMmapableVector::pointer>;

                iter_t() noexcept = default;

                template <bool OtherIsConst, typename = typename std::enable_if<IsConst && !OtherIsConst>::type>
                constexpr iter_t(iter_t<OtherIsConst> const& other) noexcept
                    : iter_data_(other.iter_data_) {}

                constexpr iter_t(ptr_t data) noexcept
                    : iter_data_(data) {}

                template <bool OtherIsConst, typename = typename std::enable_if<IsConst && !OtherIsConst>::type>
                constexpr auto operator=(iter_t<OtherIsConst> const& other) noexcept -> iter_t& {
                    iter_data_ = other.iter_data_;
                    return *this;
                }

                constexpr auto operator++() noexcept -> iter_t& {
                    ++iter_data_;
                    return *this;
                }

                constexpr auto operator+(difference_type diff) noexcept -> iter_t {
                    return {iter_data_ + diff};
                }

                template <bool OtherIsConst>
                constexpr auto operator-(iter_t<OtherIsConst> const& other) noexcept -> difference_type {
                    return static_cast<difference_type>(iter_data_ - other.iter_data_);
                }

                constexpr auto operator*() const noexcept -> reference {
                    return *iter_data_;
                }

                constexpr auto operator->() const noexcept -> pointer {
                    return iter_data_;
                }

                template <bool O>
                constexpr auto operator==(iter_t<O> const& o) const noexcept -> bool {
                    return iter_data_ == o.iter_data_;
                }

                template <bool O>
                constexpr auto operator!=(iter_t<O> const& o) const noexcept -> bool {
                    return !(*this == o);
                }
        };
};


// Memory-mappable vector of std::pair<uint64_t, uint64_t> for Ankerl
class AnkerlInt2IntMmapableVector : public pecos::mmap_util::MmapableVector<std::pair<uint64_t, uint64_t>> {
    template <bool IsConst>
    class iter_t;

    public:
        using key_type = uint64_t;
        using value_type = std::pair<uint64_t, uint64_t>;
        using mem_vec_type = std::vector<value_type>;
        using allocator_type = typename mem_vec_type::allocator_type;
        using size_type = typename mem_vec_type::size_type;
        using difference_type = typename mem_vec_type::difference_type;
        using reference = typename mem_vec_type::reference;
        using const_reference = typename mem_vec_type::const_reference;
        using pointer = typename mem_vec_type::pointer;
        using const_pointer = typename mem_vec_type::const_pointer;
        // Custom iterator
        using iterator = iter_t<false>;
        using const_iterator = iter_t<true>;

        AnkerlInt2IntMmapableVector() = default;
        AnkerlInt2IntMmapableVector(allocator_type alloc)
            : pecos::mmap_util::MmapableVector<value_type>(alloc) {}

        auto get_allocator() { return this->store_.get_allocator(); }

        constexpr auto back() -> reference { return this->data_[this->size_ - 1]; }
        constexpr auto begin() -> iterator { return {this->data_}; }
        constexpr auto cbegin() -> const_iterator { return {this->data_}; }
        constexpr auto end() -> iterator { return {this->data_ + this->size_}; }
        constexpr auto end() const -> const_iterator { return {this->data_ + this->size_}; }
        constexpr auto cend() const -> const_iterator{ return {this->data_ + this->size_}; }

        // ----- Write funcs start -----
        void shrink_to_fit() { this->store_.shrink_to_fit(); }
        void reserve(size_t new_capacity) { this->store_.reserve(new_capacity); }

        template <class... Args>
        auto emplace_back(Args&&... args) {
            auto eb_val = this->store_.emplace_back(std::forward<Args>(args)...);
            this->size_ = this->store_.size();
            this->data_ = this->store_.data();
            return eb_val;
        }

        void pop_back() {
            this->store_.pop_back();
            this->size_ = this->store_.size();
            this->data_ = this->store_.data();
        }
        // ----- Write funcs end -----

        /* Get key for member */
        key_type get_key(value_type const& vt) const {
            return vt.first;
        }


    private:
        /**
         * Iterator class doubles as const_iterator and iterator
         */
        template <bool IsConst>
        class iter_t {
            using ptr_t = typename std::conditional_t<IsConst,
                AnkerlInt2IntMmapableVector::const_pointer, AnkerlInt2IntMmapableVector::pointer>;
            ptr_t iter_data_{};

            template <bool B>
            friend class iter_t;

            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type = AnkerlInt2IntMmapableVector::difference_type;
                using value_type = AnkerlInt2IntMmapableVector::value_type;
                using reference = typename std::conditional_t<IsConst,
                    value_type const&, value_type&>;
                using pointer = typename std::conditional_t<IsConst,
                    AnkerlInt2IntMmapableVector::const_pointer, AnkerlInt2IntMmapableVector::pointer>;

                iter_t() noexcept = default;

                template <bool OtherIsConst, typename = typename std::enable_if<IsConst && !OtherIsConst>::type>
                constexpr iter_t(iter_t<OtherIsConst> const& other) noexcept
                    : iter_data_(other.iter_data_) {}

                constexpr iter_t(ptr_t data) noexcept
                    : iter_data_(data) {}

                template <bool OtherIsConst, typename = typename std::enable_if<IsConst && !OtherIsConst>::type>
                constexpr auto operator=(iter_t<OtherIsConst> const& other) noexcept -> iter_t& {
                    iter_data_ = other.iter_data_;
                    return *this;
                }

                constexpr auto operator++() noexcept -> iter_t& {
                    ++iter_data_;
                    return *this;
                }

                constexpr auto operator+(difference_type diff) noexcept -> iter_t {
                    return {iter_data_ + diff};
                }

                template <bool OtherIsConst>
                constexpr auto operator-(iter_t<OtherIsConst> const& other) noexcept -> difference_type {
                    return static_cast<difference_type>(iter_data_ - other.iter_data_);
                }

                constexpr auto operator*() const noexcept -> reference {
                    return *iter_data_;
                }

                constexpr auto operator->() const noexcept -> pointer {
                    return iter_data_;
                }

                template <bool O>
                constexpr auto operator==(iter_t<O> const& o) const noexcept -> bool {
                    return iter_data_ == o.iter_data_;
                }

                template <bool O>
                constexpr auto operator!=(iter_t<O> const& o) const noexcept -> bool {
                    return !(*this == o);
                }
        };

};
} // end namespace details_


class Str2IntMap {
public:
    void insert(const char* key, uint32_t key_len, uint64_t val) {
        map[std::string_view(key, key_len)] = val;
    }

    uint64_t get(const char* key, uint32_t key_len) {
        return map.at(std::string_view(key, key_len));
    }

    uint64_t get_w_default(const char* key, uint32_t key_len, uint64_t def_val) {
        try {
            return get(key, key_len);
        } catch (...) { return def_val;}
    }

    void batch_get_w_default(const uint32_t n_key, const char* const* keys, const uint32_t* keys_lens, const uint64_t def_val, uint64_t* vals, const int threads) {
        #pragma omp parallel for schedule(static, 1) num_threads(threads)
        for (uint32_t i=0; i<n_key; ++i) {
            vals[i] = get_w_default(keys[i], keys_lens[i], def_val);
        }
    }

    bool contains(const char* key, uint32_t key_len) {
        return map.contains(std::string_view(key, key_len));
    }

    size_t size() { return map.size(); }

    void save(const std::string& map_dir) { map.save_mmap(map_dir); }
    void load(const std::string& map_dir, const bool lazy_load) { map.load_mmap(map_dir, lazy_load); }

private:
    ankerl::unordered_dense::map<
        std::string_view, uint64_t,
        ankerl::unordered_dense::v4_0_0::hash<std::string_view>,
        std::equal_to<std::string_view>,
        details_::AnkerlStr2IntMmapableVector
    > map;
};

class Int2IntMap {
public:
    void insert(uint64_t key, uint64_t val) { map[key] = val; }
    uint64_t get(uint64_t key) { return map.at(key); }

    uint64_t get_w_default(uint64_t key, uint64_t def_val) {
        try {
            return get(key);
        } catch (...) { return def_val;}
    }

    void batch_get_w_default(const uint32_t n_key, const uint64_t* keys, const uint64_t def_val, uint64_t* vals, const int threads) {
        #pragma omp parallel for schedule(static, 1) num_threads(threads)
        for (uint32_t i=0; i<n_key; ++i) {
            vals[i] = get_w_default(keys[i], def_val);
        }
    }

    bool contains(uint64_t key) { return map.contains(key); }

    size_t size() { return map.size(); }

    void save(const std::string& folderpath) { map.save_mmap(folderpath); }
    void load(const std::string& folderpath, const bool lazy_load) { map.load_mmap(folderpath, lazy_load); }

private:
    ankerl::unordered_dense::map<
        uint64_t, uint64_t,
        ankerl::unordered_dense::v4_0_0::hash<uint64_t>,
        std::equal_to<uint64_t>,
        details_::AnkerlInt2IntMmapableVector
    > map;
};

} // end namespace ankerl_mmap_hashmap
} // end namespace pecos

#endif  // end of __MMAP_ANKERL_HASHMAP_H__
