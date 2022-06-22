#include <fcntl.h>
#include <sys/mman.h>
#include <iostream>
#include <unistd.h>


// Load array memory snapshot from disk as memory mapped file
// Note: if the array is a struct array, memory snapshot would be different across platform and compiler due to data struct padding
template <typename T>
T * load_arr_mmap(const size_t arr_len, const int fd, const off_t offset) {
    if (arr_len <= 0) {
        std::cerr << "Memory mapped array length should be positive, got: " << arr_len << std::endl;
        exit(1);
    }
    // Runtime get pagesize
    const off_t pagesize = sysconf(_SC_PAGESIZE);

    // Get the closest page containing contents from offset
    const off_t extra_bytes = offset % pagesize;
    const off_t start = offset - extra_bytes;

    // Load memory map
    void * arr_mmap = mmap64(NULL, sizeof(T) * arr_len + extra_bytes, PROT_READ, MAP_SHARED, fd, start);
    if (arr_mmap == MAP_FAILED) {
        std::cerr << "Memory map failed." << std::endl;
        exit(1);
    }

    // Shift the pointer by extra bytes read
    char * arr_mmap_bytes = reinterpret_cast< char * > (arr_mmap);
    arr_mmap_bytes += extra_bytes;

    return reinterpret_cast< T * > (arr_mmap_bytes);
}

// Free array's memory map
// template <typename T>
// void free_arr_mmap(T * arr, const size_t arr_len) {
//     auto res = munmap(arr, sizeof(T) * arr_len);
//     if (res == EINVAL) {
//         std::cerr << "Free memory map failed." << std::endl;
//         exit(1);
//     }
// }

// Dump a snapshot of the given array's memory into file
// Filepointer fp will be automatically incremented
template <typename T>
void save_arr_mmap(const T * arr, const size_t arr_len, FILE * fp) {
    if (arr_len <= 0) {
        std::cerr << "Memory mapped array length should be positive, got: " << arr_len << std::endl;
        exit(1);
    }
    fwrite(arr, sizeof(T) * arr_len, 1, fp);
}
