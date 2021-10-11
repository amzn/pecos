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

#ifndef __SCIPY_LOADER_H__
#define  __SCIPY_LOADER_H__

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "utils/file_util.hpp"

namespace pecos {

//https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
template<typename T>
class NpyArray {

public:
    typedef T value_type;
    typedef std::vector<value_type> array_t;
    typedef std::vector<uint64_t> shape_t;

    shape_t shape;
    array_t array;
    size_t num_elements;
    bool fortran_order;

    NpyArray() {}

    NpyArray(const std::string& filename, uint64_t offset=0) { load(filename, offset); }

    NpyArray(const std::vector<uint64_t>& shape, value_type default_value=0) { resize(shape, default_value); }

    /* load an NpyArry<T> starting from the `offset`-th byte in the file with `filename` */
    NpyArray<T>& load(const std::string& filename, uint64_t offset=0) {
        //https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
        FILE *fp = fopen(filename.c_str(), "rb");
        fseek(fp, offset, SEEK_SET);

        // check magic string
        std::vector<uint8_t> magic = {0x93u, 'N', 'U', 'M', 'P', 'Y'};
        for(size_t i = 0; i < magic.size(); i++) {
            if (pecos::file_util::fget_one<uint8_t>(fp) != magic[i]) {
                throw std::runtime_error("file is not a valid NpyFile");
            }
        }

        // load version
        uint8_t major_version = pecos::file_util::fget_one<uint8_t>(fp);
        uint8_t minor_version = pecos::file_util::fget_one<uint8_t>(fp);

        // load header len
        uint64_t header_len;
        if(major_version == 1) {
            header_len = pecos::file_util::fget_one<uint16_t>(fp);
        } else if (major_version == 2) {
            header_len = pecos::file_util::fget_one<uint32_t>(fp);
        } else {
            throw std::runtime_error("unsupported NPY major version");
        }

        if(minor_version != 0) {
            throw std::runtime_error("unsupported NPY minor version");
        }

        // load header
        std::vector<char> header(header_len + 1, (char) 0);
        pecos::file_util::fget_multiple<char>(&header[0], header_len, fp);
        char endian_code, type_code;
        uint32_t word_size;
        std::string dtype;
        this->parse_header(header, endian_code, type_code, word_size, dtype);

        // load array content
        this->load_content(fp, word_size, dtype);

        fclose(fp);
        return *this;
    }

    void resize(const std::vector<uint64_t>& new_shape, value_type default_value=value_type()) {
        shape = new_shape;
        size_t num_elements = 1;
        for(auto& dim : shape) {
            num_elements *= dim;
        }
        array.resize(num_elements);
        std::fill(array.begin(), array.end(), default_value);
    }

    size_t ndim() const { return shape.size(); }
    size_t size() const { return num_elements; }
    value_type* data() { return &array[0]; }
    value_type& at(size_t idx) { return array[idx]; }
    const value_type& at(size_t idx) const { return array[idx]; }
    value_type& operator[](size_t idx) { return array[idx]; }
    const value_type& operator[](size_t idx) const { return array[idx]; }

private:

    void parse_header(const std::vector<char>& header, char& endian_code, char& type_code, uint32_t& word_size, std::string& dtype) {
        char value_buffer[1024] = {0};
        const char* header_cstr = &header[0];

        // parse descr in a str form
        if(1 != sscanf(strstr(header_cstr, "'descr'"), "'descr': '%[^']' ", value_buffer)) {
            throw std::runtime_error("invalid NPY header (descr)");
        }
        dtype = std::string(value_buffer);
        if(3 != sscanf(value_buffer, "%c%c%u", &endian_code, &type_code, &word_size)) {
            throw std::runtime_error("invalid NPY header (descr parse)");
        }

        // parse fortran_order in a boolean form [False, True]
        if(1 != sscanf(strstr(header_cstr, "'fortran_order'"), "'fortran_order': %[FalseTrue] ", value_buffer)) {
            throw std::runtime_error("invalid NPY header (fortran_order)");
        }
        this->fortran_order = std::string(value_buffer) == "True";

        // parse shape in a tuple form
        if (0 > sscanf(strstr(header_cstr, "'shape'"), "'shape': (%[^)]) ", value_buffer)) {
            throw std::runtime_error("invalid NPY header (shape)");
        }

        char *ptr = &value_buffer[0];
        int offset;
        uint64_t dim;
        num_elements = 1;
        shape.clear();
        while(sscanf(ptr, "%lu, %n", &dim, &offset) == 1) {
            ptr += offset;
            shape.push_back(dim);
            num_elements *= dim;
        }
        // handle the case with single element case: shape=()
        if(shape.size() == 0 && num_elements == 1) {
            shape.push_back(1);
        }
    }

    template<typename U=value_type, typename std::enable_if<std::is_arithmetic<U>::value, U>::type* = nullptr>
    void load_content(FILE *fp, uint32_t& word_size, const std::string& dtype) {
        array.resize(num_elements);
        auto type_code = dtype.substr(1);
#define IF_CLAUSE_FOR(np_type_code, c_type) \
        if(type_code == np_type_code) { \
            bool byte_swap = pecos::file_util::different_from_runtime(dtype[0]); \
            size_t batch_size = 32768; \
            std::vector<c_type> batch(batch_size); \
            for(size_t i = 0; i < num_elements; i += batch_size) { \
                size_t num = std::min(batch_size, num_elements - i); \
                pecos::file_util::fget_multiple<c_type>(batch.data(), num, fp, byte_swap); \
                for(size_t b = 0; b < num; b++) { \
                    array[i + b] = static_cast<value_type>(batch[b]); \
                } \
            } \
        }

        IF_CLAUSE_FOR("f4", float)
        else IF_CLAUSE_FOR("f8", double)
        else IF_CLAUSE_FOR("f16", long double)
        else IF_CLAUSE_FOR("i1", int8_t)
        else IF_CLAUSE_FOR("i2", int16_t)
        else IF_CLAUSE_FOR("i4", int32_t)
        else IF_CLAUSE_FOR("i8", int64_t)
        else IF_CLAUSE_FOR("u1", uint8_t)
        else IF_CLAUSE_FOR("u2", uint16_t)
        else IF_CLAUSE_FOR("u4", uint32_t)
        else IF_CLAUSE_FOR("u8", uint64_t)
        else IF_CLAUSE_FOR("b1", uint8_t)
#undef IF_CLAUSE_FOR
    }

    template<typename U=value_type, typename std::enable_if<std::is_same<value_type, std::basic_string<typename U::value_type>>::value, U>::type* = nullptr>
    void load_content(FILE *fp, const uint32_t& word_size, const std::string& dtype) {
        array.resize(num_elements);
        auto type_code = dtype[1];
#define IF_CLAUSE_FOR(np_type_code, c_type, char_size) \
        if(type_code == np_type_code) { \
            std::vector<c_type> char_buffer(word_size); \
            bool byte_swap = pecos::file_util::different_from_runtime(dtype[0]); \
            for(size_t i = 0; i < num_elements; i++) { \
                pecos::file_util::fget_multiple<c_type>(&char_buffer[0], word_size, fp, byte_swap); \
                array[i] = value_type(reinterpret_cast<typename T::value_type*>(&char_buffer[0]), word_size * char_size); \
            } \
        }

        // numpy uses UCS4 to encode unicode https://numpy.org/devdocs/reference/c-api/dtype.html?highlight=ucs4
        IF_CLAUSE_FOR('U', char32_t, 4)
        else IF_CLAUSE_FOR('S', char, 1)
#undef IF_CLAUSE_FOR
    }
};


class ReadOnlyZipArchive {

private:

    struct FileInfo {
        std::string name;
        uint64_t offset_of_content;
        uint64_t offset_of_header;
        uint64_t uncompressed_size;
        uint64_t compressed_size;
        uint16_t compression_method;
        uint32_t signature;
        uint16_t version;
        uint16_t bit_flag;
        uint16_t last_modified_time;
        uint16_t last_modified_date;
        uint32_t crc_32;

        FileInfo() {}

        static bool valid_start(FILE *fp) {
            return pecos::file_util::fpeek<uint32_t>(fp) == 0x04034b50; // local header
        }

        // https://en.wikipedia.org/wiki/Zip_(file_format)#ZIP64
        // https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
        static FileInfo get_one_from(FILE *fp) {
            // ReadOnlyZipArchive always uses little endian
            bool byte_swap = pecos::file_util::different_from_runtime('<');

            FileInfo info;
            info.offset_of_header = ftell(fp);
            info.signature = pecos::file_util::fget_one<uint32_t>(fp, byte_swap);
            info.version = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);
            info.bit_flag = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);
            info.compression_method = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);
            info.last_modified_time = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);
            info.last_modified_date = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);
            info.crc_32 = pecos::file_util::fget_one<uint32_t>(fp, byte_swap);

            if(info.compression_method != 0) {
                throw std::runtime_error("only uncompressed zip archive is supported.");
            }

            info.compressed_size = pecos::file_util::fget_one<uint32_t>(fp, byte_swap);
            info.uncompressed_size = pecos::file_util::fget_one<uint32_t>(fp, byte_swap);
            auto filename_length = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);
            auto extra_field_length = pecos::file_util::fget_one<uint16_t>(fp, byte_swap);

            std::vector<char> filename(filename_length, (char)0);
            std::vector<char> extra_field(extra_field_length, (char)0);
            pecos::file_util::fget_multiple<char>(&filename[0], filename_length, fp, byte_swap);
            pecos::file_util::fget_multiple<char>(&extra_field[0], extra_field_length, fp, byte_swap);

            info.name = std::string(&filename[0], filename_length);
            info.offset_of_content = ftell(fp);

            // handle zip64 extra field to obtain proper size information
            uint64_t it = 0;
            while(it < extra_field.size()) {
                uint16_t header_id = *reinterpret_cast<uint16_t*>(&extra_field[it]);
                it += sizeof(header_id);
                uint16_t data_size = *reinterpret_cast<uint16_t*>(&extra_field[it]);
                it += sizeof(data_size);
                if(header_id == 0x0001) { // zip64 extended information in extra field
                    info.uncompressed_size = *reinterpret_cast<uint64_t*>(&extra_field[it]);
                    info.compressed_size = *reinterpret_cast<uint64_t*>(&extra_field[it + sizeof(info.uncompressed_size)]);
                }
                it += data_size;
            }

            // skip the actual content
            fseek(fp, info.compressed_size, SEEK_CUR);

            // skip the data descriptor if bit 3 of the general purpose bit flag is set
            if (info.bit_flag & 8) {
                fseek(fp, 12, SEEK_CUR);
            }

            return info;
        }
    };

    std::vector<FileInfo> file_info_array;
    std::unordered_map<std::string, FileInfo*> mapping;

public:

    ReadOnlyZipArchive(const std::string& zip_name) {
        FILE *fp = fopen(zip_name.c_str(), "rb");
        while(FileInfo::valid_start(fp)) {
            file_info_array.emplace_back(FileInfo::get_one_from(fp));
        }
        fclose(fp);
        for(auto& file : file_info_array) {
            mapping[file.name] = &file;
        }
    }

    FileInfo& operator[](const std::string& name) { return *mapping.at(name); }
    const FileInfo& operator[](const std::string& name) const { return *mapping.at(name); }
};


template<bool IsCsr, typename DataT, typename IndicesT = uint32_t, typename IndptrT = uint64_t, typename ShapeT = uint64_t>
struct ScipySparseNpz {
    NpyArray<IndicesT> indices;
    NpyArray<IndptrT> indptr;
    NpyArray<DataT> data;
    NpyArray<ShapeT> shape;
    NpyArray<std::string> format;

    ScipySparseNpz() {}

    ScipySparseNpz(const std::string& npz_filepath) { load(npz_filepath); }


    uint64_t size() const { return data.size(); }
    uint64_t rows() const { return shape[0]; }
    uint64_t cols() const { return shape[1]; }
    uint64_t nnz() const { return data.size(); }

    void load(const std::string& npz_filepath) {
        auto npz = ReadOnlyZipArchive(npz_filepath);
        format.load(npz_filepath, npz["format.npy"].offset_of_content);
        if(IsCsr && format[0] != "csr") {
            throw std::runtime_error(npz_filepath + " is not a valid scipy CSR npz");
        } else if (!IsCsr && format[0] != "csc") {
            throw std::runtime_error(npz_filepath + " is not a valid scipy CSC npz");
        }
        indices.load(npz_filepath, npz["indices.npy"].offset_of_content);
        data.load(npz_filepath, npz["data.npy"].offset_of_content);
        indptr.load(npz_filepath, npz["indptr.npy"].offset_of_content);
        shape.load(npz_filepath, npz["shape.npy"].offset_of_content);
    }

    void fill_ones(size_t rows, size_t cols) {
        shape.resize({2});
        shape[0] = rows;
        shape[1] = cols;

        uint64_t nnz = rows * cols;
        data.resize({nnz}, DataT(1));
        indices.resize({nnz});
        format.resize({1});
        if(IsCsr) {
            format[0] = "csr";
            indptr.resize({rows + 1});
            for(size_t r = 0; r < rows; r++) {
                for(size_t c = 0; c < cols; c++) {
                    indices[r * cols + c] = c;
                }
                indptr[r + 1] = indptr[r] + cols;
            }
        } else {
            format[0] = "csc";
            indptr.resize({cols + 1});
            for(size_t c = 0; c < cols; c++) {
                for(size_t r = 0; r < rows; r++) {
                    indices[c * rows + r] = r;
                }
                indptr[c + 1] = indptr[c] + rows;
            }
        }
    }
};

} // end namespace pecos

#endif // end of __SCIPY_LOADER_H__
