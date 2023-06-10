#include "../third_party/ankerl/unordered_dense.h"
#include "ankerl_int2int_mmap_vec.hpp"
#include "ankerl_str2int_mmap_vec.hpp"

namespace pecos {
namespace ankerl_mmap_hashmap {

class Str2IntMap {
public:
    void insert(std::string_view key, uint64_t val) { map[key] = val; }
    uint64_t get(std::string_view key) { return map[key]; }
    auto size() { return map.size(); }

    void save(const std::string& folderpath) { map.save_mmap(folderpath); }
    void load(const std::string& folderpath, const bool lazy_load) { map.load_mmap(folderpath, lazy_load); }

private:
    ankerl::unordered_dense::map<
        std::string_view, uint64_t,
        ankerl::unordered_dense::v4_0_0::hash<std::string_view>,
        std::equal_to<std::string_view>,
        AnkerlStr2IntMmapableVector
    > map;
};

class Int2IntMap {
public:
    void insert(uint64_t key, uint64_t val) { map[key] = val; }
    uint64_t get(uint64_t key) { return map[key]; }
    auto size() { return map.size(); }

    void save(const std::string& folderpath) { map.save_mmap(folderpath); }
    void load(const std::string& folderpath, const bool lazy_load) { map.load_mmap(folderpath, lazy_load); }

private:
    ankerl::unordered_dense::map<
        uint64_t, uint64_t,
        ankerl::unordered_dense::v4_0_0::hash<uint64_t>,
        std::equal_to<uint64_t>,
        AnkerlInt2IntMmapableVector
    > map;
};

} // end namespace mmap_util
} // end namespace pecos
