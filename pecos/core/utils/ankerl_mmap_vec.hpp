#include "mmap_util.hpp"


// Memory-mappable vector of fixed length type T for Ankerl
// When calling write methods, the assumption is that the underlying storage is in memory, i.e. std::vector
template<class T>
class AnkerlMmapableVector : public pecos::mmap_util::MmapableVector<T> {
    template <bool IsConst>
    class iter_t;

    public:
        using mem_vec_type = std::vector<T>;
        using allocator_type = typename mem_vec_type::allocator_type;
        using value_type = typename mem_vec_type::value_type;
        using size_type = typename mem_vec_type::size_type;
        using difference_type = typename mem_vec_type::difference_type;
        using reference = typename mem_vec_type::reference;
        using const_reference = typename mem_vec_type::const_reference;
        using pointer = typename mem_vec_type::pointer;
        using const_pointer = typename mem_vec_type::const_pointer;
        // Custom iterator
        using iterator = iter_t<false>;
        using const_iterator = iter_t<true>;

        AnkerlMmapableVector() = default;
        AnkerlMmapableVector(allocator_type alloc)
            : pecos::mmap_util::MmapableVector<T>(alloc) {}

        auto get_allocator() { return this->store_.get_allocator(); }

        constexpr auto back() -> reference { return this->data_[this->size_ - 1]; }
        constexpr auto begin() -> iterator { return {this->data_}; }
        constexpr auto cbegin() -> const_iterator { return {this->data_}; }
        constexpr auto end() -> iterator { return {this->data_ + this->size_}; }
        constexpr auto cend() -> const_iterator{ return {this->data_ + this->size_}; }

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


    private:
        /**
         * Iterator class doubles as const_iterator and iterator
         */
        template <bool IsConst>
        class iter_t {
            using ptr_t = typename std::conditional_t<IsConst,
                AnkerlMmapableVector::const_pointer, AnkerlMmapableVector::pointer>;
            ptr_t iter_data_{};

            template <bool B>
            friend class iter_t;

            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type = AnkerlMmapableVector::difference_type;
                using value_type = T;
                using reference = typename std::conditional_t<IsConst,
                    value_type const&, value_type&>;
                using pointer = typename std::conditional_t<IsConst,
                    AnkerlMmapableVector::const_pointer, AnkerlMmapableVector::pointer>;

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
