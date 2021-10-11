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

#ifndef __FILE_UTIL_H__
#define __FILE_UTIL_H__

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

namespace pecos {

namespace file_util {

using std::string;
using std::vector;

// return '<' for little endian and '>' for big endian
static char runtime () {
    uint32_t x = 1U;
    return (reinterpret_cast<uint8_t*>(&x)[0]) ? '<' : '>';
}

// return true if the given byte_order code (>, <, |, =) is different from the byte order of runtime
static bool different_from_runtime(char byte_order) {
    if(byte_order == '|' || byte_order == '=' || byte_order == file_util::runtime()) {
        return false;
    } else {
        return true;
    }
}

template<class T>
static T byte_swap(T& src) {
    if(sizeof(T) == 1) {
        return src;
    }
    typename std::remove_const<T>::type dst;
    auto src_ptr = reinterpret_cast<const std::uint8_t*>(&src);
    auto dst_ptr = reinterpret_cast<std::uint8_t*>(&dst);
    std::reverse_copy(src_ptr, src_ptr + sizeof(T), dst_ptr);
    return dst;
}

template<class T>
inline T* fget_multiple(T* dst, size_t num, FILE *stream, bool byte_swap=false) {
    if(num != fread(dst, sizeof(T), num, stream)) {
        throw std::runtime_error("Cannot read enough data from the stream");
    }
    // swap the endianness
    if(byte_swap) {
        for(size_t i = 0; i < num; i++) {
            dst[i] = file_util::byte_swap(dst[i]);
        }
    }
    return dst;
}

template<class T>
inline T fget_one(FILE *stream, bool byte_swap=false) {
    T x;
    file_util::fget_multiple<T>(&x, 1U, stream, byte_swap);
    return x;
}

template<class T>
inline void fput_multiple(const T* src, size_t num, FILE *stream, bool byte_swap=false) {
    if(byte_swap) {
        for (size_t i = 0; i < num; i++) {
            T src_copy = file_util::byte_swap(src[i]);
            if(1U != fwrite(&src_copy, sizeof(T), 1U, stream)) {
                throw std::runtime_error("Cannot write enough data from the stream");
            }
        }
    } else {
        if(num != fwrite(src, sizeof(T), num, stream)) {
            throw std::runtime_error("Cannot write enough data from the stream");
        }
    }
}

template<class T>
inline void fput_one(const T& src, FILE *stream, bool byte_swap=false) {
    file_util::fput_multiple<T>(&src, 1U, stream, byte_swap);
}

template<class T>
inline T fpeek(FILE *stream, bool byte_swap=false) {
    T x = file_util::fget_one<T>(stream, byte_swap);
    fseek(stream, -sizeof(T), SEEK_CUR);
    return x;
}

// get file size in bytes
size_t get_filesize(const string& filename) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        fprintf(stderr, "Error getting file size: can't read %s!!\n", filename.c_str());
        return 0;
    }
    fseek(fp, 0, SEEK_END);
    size_t filesize = ftell(fp);
    fclose(fp);
    return filesize;
}

// count number of lines in file
size_t get_linecount(const string& filename, size_t start_pos=0, size_t end_pos=0) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        fprintf(stderr, "Error getting line count: can't read %s!!\n", filename.c_str());
        return 0;
    }
    const int chunksize = 10240;
    char buf[chunksize];
    if(end_pos == 0) {
        fseek(fp, 0, SEEK_END);
        end_pos = ftell(fp);
    }
    size_t linecount = 0;
    fseek(fp, start_pos, SEEK_SET);
    for(size_t cnt = start_pos; cnt < end_pos; cnt += chunksize) {
        size_t buf_len = std::min((size_t)chunksize, end_pos-cnt);
        size_t tmp;
        if(buf_len != (tmp=fread(&buf[0], sizeof(char), buf_len, fp))) {
            fprintf(stderr, "Error: something wrong in linecount() expect %ld bytes but read %ld instead!!\n", tmp, buf_len);
        }
        for(size_t i = 0; i < buf_len; i++) {
            if(buf[i] == '\n') {
                linecount++;
            }
        }
    }
    fclose(fp);
    return linecount;
}

// load file into memory buffer with extra '\0' at end
// buffer size should > end_pos - start_pos + 1
// return the actual overwritten buffer size
size_t load_file_block(const string& filename, char* buffer, size_t start_pos=0, size_t end_pos=0) {
    if(end_pos < start_pos) {
        throw std::invalid_argument("got end_pos < start_pos");
    }
    FILE *fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        fprintf(stderr, "Error loading file: can't read %s!!\n", filename.c_str());
    }
    if(end_pos == 0) {
        fseek(fp, 0, SEEK_END);
        end_pos = ftell(fp);
    }
    fseek(fp, start_pos, SEEK_SET);
    size_t read_len = end_pos - start_pos;
    if(fread(buffer, sizeof(char), read_len, fp) != read_len){
        fprintf(stderr, "Error: error reading %s!!\n", filename.c_str());
    }
    fclose(fp);
    // set proper end char for loaded buffer
    buffer[read_len] = '\0';
    return read_len + 1;
}

// split file into chunks with sizes approximately equal to given chunk_size
// chunk i=0,1,2... start from position (chunk_size * i) and search for immediate next \n symbol as corresponding offsets
// result chunk_offset = {0, offset_1, offset_2, ..., offset_n} where offset_n = filesize
void get_file_offset(const string& filename, size_t chunk_size, vector<size_t>& chunk_offset) {
    size_t file_size = get_filesize(filename);
    chunk_size = std::min(chunk_size, file_size);
    // Generate offset for each block
    size_t n_chunks = (file_size + chunk_size - 1) / chunk_size;
    chunk_offset.resize(n_chunks + 1);
    chunk_offset[0] = 0; chunk_offset[n_chunks] = file_size;
    FILE *src_fp = fopen(filename.c_str(), "rb");
    for(size_t i = 1; i < n_chunks; i++) {
        chunk_offset[i] = chunk_offset[i - 1] + chunk_size;
        if(chunk_offset[i] >= file_size) {
            chunk_offset[i] = file_size;
            chunk_offset.resize(i + 1);
            break;
        }
        fseek(src_fp, chunk_offset[i] - 1, SEEK_SET);
        while(!feof(src_fp) && fgetc(src_fp) != '\n');
        chunk_offset[i] = ftell(src_fp);
    }
    fclose(src_fp);
}

} // end namespace file_util
} // end namespace pecos

#endif  // end of __FILE_UTIL_H__
