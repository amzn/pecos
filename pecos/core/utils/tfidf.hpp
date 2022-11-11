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

#ifndef __TFIDF_H__
#define __TFIDF_H__

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

// string_view available since c++17
// only in experimental/string_view in c++14
#if __cplusplus >= 201703L
#include <string_view>
#else
#include <experimental/string_view>
namespace std{
  using std::experimental::string_view;
}
#endif

#include "parallel.hpp"
#include "file_util.hpp"
#include "third_party/nlohmann_json/json.hpp"
#include "third_party/robin_hood_hashing/robin_hood.h"


namespace pecos {

namespace tfidf {


using std::string;
using std::vector;
using std::string_view;
using robin_hood::unordered_map;
using robin_hood::unordered_set;
using robin_hood::hash;
using robin_hood::pair;

// ===== C Interface for Structure/Types =====

extern "C" {

    // Because TfidfBaseVectorizerParam will be used to communicate with python ctypes,
    // its data structure need to be a "standard-layout type", see more at
    // http://www.cplusplus.com/reference/type_traits/is_standard_layout
    struct TfidfBaseVectorizerParam {

        TfidfBaseVectorizerParam() {}

        TfidfBaseVectorizerParam(
            int32_t min_ngram,
            int32_t max_ngram,
            int32_t max_length,
            int32_t max_feature,
            float min_df_ratio,
            float max_df_ratio,
            int32_t min_df_cnt,
            int32_t max_df_cnt,
            bool binary,
            bool use_idf,
            bool smooth_idf,
            bool add_one_idf,
            bool sublinear_tf,
            bool keep_frequent_feature,
            int32_t norm_p,
            int32_t tok_type
        ): min_ngram(min_ngram),
        max_ngram(max_ngram),
        max_length(max_length),
        max_feature(max_feature),
        min_df_ratio(min_df_ratio),
        max_df_ratio(max_df_ratio),
        min_df_cnt(min_df_cnt),
        max_df_cnt(max_df_cnt),
        binary(binary),
        use_idf(use_idf),
        smooth_idf(smooth_idf),
        add_one_idf(add_one_idf),
        sublinear_tf(sublinear_tf),
        keep_frequent_feature(keep_frequent_feature),
        norm_p(norm_p),
        tok_type(tok_type) {
            if(min_df_ratio < 0 || max_df_ratio <= min_df_ratio || max_df_ratio > 1.0) {
                throw std::invalid_argument("expect 0 <= min_df_ratio < max_df_ratio <= 1.0");
            }
            if(min_ngram <=0 || min_ngram > max_ngram) {
                throw std::invalid_argument("expect 0 < min_ngram <= max_ngram");
            }
        }

        int32_t min_ngram, max_ngram;
        int32_t max_length, max_feature;
        float min_df_ratio, max_df_ratio;
        int32_t min_df_cnt, max_df_cnt;
        bool binary, use_idf, smooth_idf, add_one_idf, sublinear_tf, keep_frequent_feature;
        int32_t norm_p, tok_type;

        void save(const string& filepath) const {
            nlohmann::json j_params = {
                {"type", "tfidf"},
                {"kwargs",{
                    {"ngram_range", {min_ngram, max_ngram}},
                    {"max_length", max_length},
                    {"max_feature", max_feature},
                    {"min_df_ratio", min_df_ratio},
                    {"max_df_ratio", max_df_ratio},
                    {"min_df_cnt", min_df_cnt},
                    {"max_df_cnt", max_df_cnt},
                    {"binary", binary},
                    {"use_idf", use_idf},
                    {"smooth_idf", smooth_idf},
                    {"add_one_idf", add_one_idf},
                    {"sublinear_tf", sublinear_tf},
                    {"keep_frequent_feature", keep_frequent_feature},
                    {"norm_p", norm_p == 1 ? "l1" : "l2"}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if(savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void load(const string& filepath) {
            std::ifstream loadfile(filepath);
            string json_str;
            if(loadfile.is_open()) {
                json_str.assign((std::istreambuf_iterator<char>(loadfile)),
                                   (std::istreambuf_iterator<char>()));
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            auto j_param = nlohmann::json::parse(json_str);
            string vectorizer_type = j_param["type"];
            if(vectorizer_type != "tfidf") {
                throw std::invalid_argument("Wrong vectorizer type: " + vectorizer_type);
            }
            auto kwargs = j_param["kwargs"];
            auto ngram_range = kwargs["ngram_range"];
            min_ngram = ngram_range[0];
            max_ngram = ngram_range[1];

            max_length = kwargs["max_length"];
            max_feature = kwargs["max_feature"];
            min_df_ratio = kwargs["min_df_ratio"];
            max_df_ratio = kwargs["max_df_ratio"];
            min_df_cnt = kwargs["min_df_cnt"];
            max_df_cnt = kwargs["max_df_cnt"];
            binary = kwargs["binary"];
            use_idf = kwargs["use_idf"];
            smooth_idf = kwargs["smooth_idf"];
            sublinear_tf = kwargs["sublinear_tf"];
            keep_frequent_feature = kwargs["keep_frequent_feature"];
            if(kwargs["norm_p"] == "l1") {
                norm_p = 1;
            } else if (kwargs["norm_p"] == "l2") {
                norm_p = 2;
            } else {
                throw std::invalid_argument("Unknown normalization type");
            }
	    // handle missing key by filling default value
            add_one_idf = kwargs.value("add_one_idf", false);
        }

    };

    // Because TfidfVectorizerParam will be used to communicate with python ctypes,
    // its data structure need to be a "standard-layout type", see more at
    // http://www.cplusplus.com/reference/type_traits/is_standard_layout
    struct TfidfVectorizerParam {

        TfidfVectorizerParam() {}

        TfidfVectorizerParam(
            TfidfBaseVectorizerParam* base_param_ptr,
            int32_t num_base_vect,
            int32_t norm_p
        ): base_param_ptr(base_param_ptr),
        num_base_vect(num_base_vect),
        norm_p(norm_p) {}

        TfidfBaseVectorizerParam* base_param_ptr;
        int32_t num_base_vect;
        int32_t norm_p; // only support 1 or 2

        void save(const string& filepath) const {
            nlohmann::json j_params = {
                {"type", "tfidf"},
                {"kwargs", {
                    {"num_base_vect", num_base_vect},
                    {"norm_p", norm_p}
                    }
                }
            };
            std::ofstream savefile(filepath, std::ofstream::trunc);
            if(savefile.is_open()) {
                savefile << j_params.dump(4);
                savefile.close();
            } else {
                throw std::runtime_error("Unable to save config file to " + filepath);
            }
        }

        void load(const string& filepath) {
            std::ifstream loadfile(filepath);
            string json_str;
            if(loadfile.is_open()) {
                json_str.assign((std::istreambuf_iterator<char>(loadfile)),
                                   (std::istreambuf_iterator<char>()));
            } else {
                throw std::runtime_error("Unable to open config file at " + filepath);
            }
            const auto& j_param = nlohmann::json::parse(json_str);
            const string& vectorizer_type = j_param["type"];
            if(vectorizer_type != "tfidf") {
                throw std::invalid_argument("Wrong vectorizer type: " + vectorizer_type);
            }
            const auto& kwargs = j_param["kwargs"];
            num_base_vect = kwargs["num_base_vect"];
            norm_p = kwargs["norm_p"];
        }

    };

} // end of extern C


typedef int idx_type; // special tokens use negative index
typedef vector<string> str_vec_t;
typedef vector<string_view> sv_vec_t;
typedef vector<idx_type> idx_vec_t;

enum {
    WORDTOKENIZER=10,
    CHARTOKENIZER=20,
    CHARWBTOKENIZER=30,
};

size_t DEFAULT_BUFFER_SIZE = size_t(2e8); // default max buffer size for file I/O

// Hash function for vector to support arbitrary length ngram feature
// REF: https://www.boost.org/doc/libs/1_74_0/doc/html/hash/reference.html#boost.hash_combine
template<class val_type>
struct VectorHasher {
    size_t operator()(vector<val_type> const& V) const {
        size_t val = V.size();
        for(auto &i : V) {
            val ^= hash<val_type>()(i) + 0x9e3779b9 + (val << 6) + (val >> 2);
        }
        return val;
    }
};

// split c str by '\n' and append reference to line_view
void append_lines_to_string_view(char* buffer, size_t buffer_size, sv_vec_t& line_view) {
    // split buffer with \n for lines
    size_t start = 0, end = 0;
    while(end < buffer_size) {
        if(buffer[end] == '\n') {
            string_view lv(buffer + start, end - start);
            line_view.push_back(lv);
            start = end + 1;
        }
        end++;
    }
    if(start < buffer_size && buffer[start] != '\0') {
        string_view lv(buffer + start, end - start);
        line_view.push_back(lv);
    }
}


class Tokenizer {
public:
    typedef unordered_map<string, idx_type> str2idx_map_t;
    typedef unordered_map<idx_vec_t, idx_type, VectorHasher<idx_type>> vec2idx_map_t;
    typedef unordered_set<string> str_set_t;

    static const idx_type UNK = -1; // unknown token index, will not appear in feature ngrams
    const string DELIMS = " "; // in US-ASCII, delimiters for word tokenizer

    str2idx_map_t vocab;
    int tok_type;

    Tokenizer(int tok_type=WORDTOKENIZER): tok_type(tok_type) {
        if(tok_type != WORDTOKENIZER && tok_type != CHARTOKENIZER && tok_type != CHARWBTOKENIZER) {
            throw std::invalid_argument("received unknown tok_type: " + std::to_string(tok_type));
        }
    }

    Tokenizer(const string& load_dir) { load(load_dir); }

    static nlohmann::json load_config(const string& filepath) {
        std::ifstream loadfile(filepath);
        string json_str;
        if(loadfile.is_open()) {
            json_str.assign((std::istreambuf_iterator<char>(loadfile)),
                               (std::istreambuf_iterator<char>()));
        } else {
            throw std::runtime_error("Unable to open config file at " + filepath);
        }
        auto j_param = nlohmann::json::parse(json_str);
        return j_param;
    }

    void save_config(const string& filepath) const {
        nlohmann::json j_params = {
            {"token_type", tok_type}
        };
        std::ofstream savefile(filepath, std::ofstream::trunc);
        if(savefile.is_open()) {
            savefile << j_params.dump(4);
            savefile.close();
        } else {
            throw std::runtime_error("Unable to save config file to " + filepath);
        }
    }

    void save(const string& save_dir) const {
        if(mkdir(save_dir.c_str(), 0777) == -1) {
            if(errno != EEXIST) {
                throw std::runtime_error("Unable to create save folder at " + save_dir);
            }
        }
        save_config(save_dir + "/config.json");
        std::ofstream savefile(save_dir + "/vocab.txt", std::ofstream::trunc);
        if(savefile.is_open()) {
            savefile << std::to_string(vocab.size()) << '\n';
            for(auto iter = vocab.begin(); iter != vocab.end(); iter++) {
                // (INDEX)<TAB>(KEY)\n
                savefile << std::to_string(iter->second) << '\t' << iter->first << '\n';
            }
            savefile.close();
        } else {
            throw std::runtime_error("Unable to save vocab file to " + save_dir + "/vocab.txt");
        }
    }

    void load(const string& load_dir) {
        auto config = load_config(load_dir + "/config.json");
        this->tok_type = config["token_type"];

        std::ifstream loadfile(load_dir + "/vocab.txt");
        if(loadfile.is_open()) {
            string line;
            // read number of keys
            getline(loadfile, line);
            size_t vocab_size = size_t(std::stoul(line));
            vocab.reserve(vocab_size);
            while(getline(loadfile, line)) {
                size_t pos = line.find('\t');
                if(pos == string::npos) {
                    throw std::runtime_error("Corrupted vocab file.");
                }
                idx_type idx = idx_type(std::stoi(line.substr(0, pos)));
                vocab[line.substr(pos + 1, line.size() - pos - 1)] = idx;
            }
            loadfile.close();
        } else {
            throw std::runtime_error("Unable to open tokenizer vocab file at " + load_dir + "/vocab.txt");
        }
    }

    // split given string_view into tokens, clear tokens if not empty
    void split_into_tokens(const string_view& sv, sv_vec_t& tokens, int override_tok_type=-1) const {
        if(override_tok_type < 0) {
            override_tok_type = tok_type;
        }
        tokens.clear();
        if(override_tok_type == WORDTOKENIZER) {
            for(auto first = sv.data(), second = sv.data(), last = first + sv.size();
                    second != last && first != last;
                    first = second + 1) {
                second = std::find_first_of(first, last, DELIMS.begin(), DELIMS.end());

                if(first != second) {
                    tokens.emplace_back(first, second - first);
                }
            }
        } else if(override_tok_type == CHARTOKENIZER || override_tok_type == CHARWBTOKENIZER) {
            auto first = sv.data(), last = first + sv.size();
            int char_size = 1;
            while(first != last) {
                // decide current char_size
                // REF: https://en.wikipedia.org/wiki/UTF-8#Encoding
                if((uint8_t)*first >= 0xe0) {
                    if((uint8_t)*first >= 0xf0) {
                        char_size = 4; // 11110XXX
                    } else {
                        char_size = 3; // 1110XXXX
                    }
                } else {
                    if((uint8_t)*first >= 0xc0) {
                        char_size = 2; // 110XXXXX
                    } else if((uint8_t)*first < 0x80) {
                        char_size = 1; // 0XXXXXXX
                    } else {
                        throw std::runtime_error("the string is not utf-8 encoded!"); // 10XXXXXX
                    }
                }
                tokens.emplace_back(first, char_size);
                first += char_size;
            }
        }
    }

    // convert string_view into token indices no longer than max_length
    // length constraint ignored if max_length < 0
    void tokenize(const string_view& sv, idx_vec_t& indices, int max_length) const {
        sv_vec_t tokens;
        split_into_tokens(sv, tokens);
        size_t actual_size = tokens.size();
        if(max_length > 0 && (size_t)max_length < actual_size) {
            actual_size = max_length;
        }
        indices.resize(actual_size);
        for(size_t i = 0; i < actual_size; i++) {
            string cur_token(tokens[i]);
            if(vocab.find(cur_token) != vocab.end()) {
                indices[i] = vocab.at(cur_token);
            } else {
                indices[i] = UNK;
            }
        }
    }

    // count ngrams appeared in given string_view
    void count_ngrams(const string_view& line_sv, vec2idx_map_t& ngram_cnt, int min_ngram, int max_ngram, int max_length) const {
        if(tok_type == CHARWBTOKENIZER) {
            // for character-word-boundary, n-grams at the edges of words are padded with space
            // REF: https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/feature_extraction/text.py#L278
            int length_left = max_length;
            sv_vec_t word_tokens;
            split_into_tokens(line_sv, word_tokens, WORDTOKENIZER);
            for(string_view& cur_word : word_tokens) {
                idx_vec_t tokens;
                tokenize(cur_word, tokens, length_left);
                length_left -= tokens.size();
                // find all features between min_ngram and max_ngram
                for(int cur_ngram = min_ngram; cur_ngram <= max_ngram; cur_ngram++) {
                    idx_type SPS = vocab.at(" ");
                    if(cur_ngram >= (int) tokens.size() + 2) { // count a short word only once
                        idx_vec_t feat_key(tokens.size() + 2, SPS);
                        std::copy(tokens.begin(), tokens.end(), feat_key.begin() + 1);
                        ngram_cnt[feat_key] += 1;
                        break;
                    } else {
                        idx_vec_t feat_key(cur_ngram, SPS);
                        // left pad
                        std::copy(tokens.begin(), tokens.begin() + cur_ngram - 1, feat_key.begin() + 1);
                        ngram_cnt[feat_key] += 1;
                        // real ngrams
                        for(int i = 0; i + cur_ngram <= (int) tokens.size(); i++) {
                            std::copy(tokens.begin() + i, tokens.begin() + i + cur_ngram, feat_key.begin());
                            ngram_cnt[feat_key] += 1;
                        }
                        // right pad
                        feat_key[cur_ngram - 1] = SPS;
                        std::copy(tokens.end() - cur_ngram + 1, tokens.end(), feat_key.begin());
                        ngram_cnt[feat_key] += 1;
                    }
                }
            }
        } else {
            idx_vec_t tokens;
            tokenize(line_sv, tokens, max_length);
            // find all features between min_ngram and max_ngram
            for(int cur_ngram = min_ngram; cur_ngram <= std::min(max_ngram, (int)tokens.size()); cur_ngram++) {
                for(int i = 0; i <= (int)tokens.size() - cur_ngram; i++) {
                    idx_vec_t feat_key(tokens.begin() + i, tokens.begin() + i + cur_ngram);
                    ngram_cnt[feat_key] += 1;
                }
            }
        }
    }

private:
    // build vocabulary and count doc_freq with given corpus chunk
    // results are added to vocab_chunk
    void incremental_train_chunk_(const sv_vec_t& corpus, str_set_t& vocab_chunk, size_t start_line=0, size_t end_line=0) {
        if(end_line <= start_line || end_line > corpus.size()) {
            end_line = corpus.size();
        }

        sv_vec_t tokens;
        for(auto line_sv = corpus.begin() + start_line; line_sv != corpus.begin() + end_line; line_sv++) {
            split_into_tokens(*line_sv, tokens);
            for(auto& cur_token : tokens) {
                string cur_str_token(cur_token.begin(), cur_token.end());
                vocab_chunk.insert(cur_str_token);
            }
        }
    }

    // parallel build vocabulary from single file to vocab_chunks
    void train_from_file_(const string& corpus_path, vector<str_set_t>& vocab_chunks, vector<vector<char>>& buffer, size_t chunk_size) {
        vector<size_t> chunk_offset;
        file_util::get_file_offset(corpus_path, chunk_size, chunk_offset);
        size_t n_chunks = chunk_offset.size() - 1;

#pragma omp parallel for schedule(dynamic,1)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            int proc_id = omp_get_thread_num();
            // load file chunk and parse lines to string_views
            sv_vec_t cur_corpus_sv;
            if(buffer[proc_id].size() <= chunk_offset[chunk + 1] - chunk_offset[chunk]) {
                // need to increase buffer size
                buffer[proc_id].resize(chunk_offset[chunk + 1] - chunk_offset[chunk] + 1);
            }
            size_t cache_size = file_util::load_file_block(corpus_path,
                    buffer[proc_id].data(), chunk_offset[chunk], chunk_offset[chunk + 1]);

            append_lines_to_string_view(buffer[proc_id].data(), cache_size, cur_corpus_sv);

            incremental_train_chunk_(cur_corpus_sv, vocab_chunks[proc_id]);
        }
    }

    // parallel build vocabulary from memory to vocab_chunks
    void train_from_mem_(const sv_vec_t& corpus, vector<str_set_t>& vocab_chunks) {
        size_t n_chunks = std::min(vocab_chunks.size(), corpus.size());
        size_t chunk_size = (corpus.size() + n_chunks - 1) / n_chunks;

#pragma omp parallel for schedule(dynamic,1)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            size_t start_line = chunk * chunk_size;
            if(start_line < corpus.size()) {
                size_t end_line = std::min((chunk + 1) * chunk_size, corpus.size());
                incremental_train_chunk_(corpus, vocab_chunks[chunk], start_line, end_line);
            }
        }
    }

    void merge_vocabs(vector<str_set_t>& vocab_chunks, int threads) {
        // upper bound for vocab size
        size_t max_vocab_size = 0;
        for(auto& cur_chunk : vocab_chunks) {
            max_vocab_size += cur_chunk.size();
        }
        // merge vocab_chunks to first chunk
        vocab_chunks[0].reserve(max_vocab_size);
        for(size_t i = 1; i < vocab_chunks.size(); i++) {
            auto& part = vocab_chunks[i];
            vocab_chunks[0].insert(part.begin(), part.end());
            // make sure release mem
            part.clear();
            str_set_t().swap(part);
        }
        // space token is needed in char_wb as padding token
        if(tok_type == CHARWBTOKENIZER) {
            vocab_chunks[0].insert(" ");
        }
        // sort tokens with their keys
        str_vec_t all_token_vec(vocab_chunks[0].begin(), vocab_chunks[0].end());
        vocab_chunks.clear();
        vector<size_t> tok_idcs(all_token_vec.size());
        for(size_t i = 0; i < tok_idcs.size(); i++) {
            tok_idcs[i] = i;
        }
        parallel_sort(tok_idcs.begin(), tok_idcs.end(), [&](const size_t& i, const size_t& j) -> bool {
                return all_token_vec[i] < all_token_vec[j];
            },
            threads
        );
        // merge token2count mappings
        for(size_t cur_idx = 0; cur_idx < tok_idcs.size(); cur_idx++) {
            string& cur_str = all_token_vec[tok_idcs[cur_idx]];
            vocab[cur_str] = static_cast<idx_type>(cur_idx);
        }
    }

public:
    // build vocabulary with corpus in memory
    void train(const sv_vec_t& corpus, int threads=-1) {
        threads = set_threads(threads);
        vector<str_set_t> vocab_chunks(threads);
        train_from_mem_(corpus, vocab_chunks);
        merge_vocabs(vocab_chunks, threads);
    }

    // build vocabulary from a single file
    void train_from_file(const string& corpus_file, size_t buffer_size=0, int threads=-1) {
        str_vec_t corpus_files{corpus_file};
        train_from_file(corpus_files, buffer_size, threads);
    }

    // build vocabulary from multiple files, one file at a time
    void train_from_file(const str_vec_t& corpus_files, size_t buffer_size=0, int threads=-1) {
        buffer_size = std::max(DEFAULT_BUFFER_SIZE, buffer_size);

        threads = set_threads(threads);
        vector<str_set_t> vocab_chunks(threads);

        // workspace memory to load data
        size_t proc_buf_size = buffer_size / threads;
        vector<vector<char>> buffer(threads);
        for(int i = 0; i < threads; i++) {
            buffer[i].resize(proc_buf_size);
        }

        for(auto& cur_corpus_file : corpus_files) {
            size_t cur_file_size = file_util::get_filesize(cur_corpus_file);
            if(cur_file_size < buffer[0].size() - 1) {
                // load whole file in buffer and parse to string_views
                sv_vec_t cur_corpus_sv;
                size_t cache_size = file_util::load_file_block(cur_corpus_file, buffer[0].data());
                append_lines_to_string_view(buffer[0].data(), cache_size, cur_corpus_sv);

                train_from_mem_(cur_corpus_sv, vocab_chunks);
            } else {
                // load file in chunks and parallel training
                // chunk_size may increase depending on the position of next \n
                // to avoid resizing buffer, use proc_buf_size / 2 as chunk_size
                train_from_file_(cur_corpus_file, vocab_chunks, buffer, proc_buf_size / 2);
            }
        }
        merge_vocabs(vocab_chunks, threads);
    }

}; // end Tokenizer

class BaseVectorizer {
public:
    typedef unordered_map<idx_type, float> idx2float_map_t;
    typedef unordered_map<idx_vec_t, idx_type, VectorHasher<idx_type>> vec2idx_map_t;
    typedef unordered_set<idx_vec_t, VectorHasher<idx_type>> vec_set_t;

    // arguments
    TfidfBaseVectorizerParam param;

    // TFIDF model
    Tokenizer tokenizer;
    vec2idx_map_t feature_vocab;
    idx2float_map_t idx_idf;

    BaseVectorizer() {};

    BaseVectorizer(const TfidfBaseVectorizerParam* param_ptr):
        param(*param_ptr), tokenizer(param_ptr->tok_type) { }

    BaseVectorizer(const string& filepath) { load(filepath); }

    size_t nr_features() const { return feature_vocab.size(); }

    void save(const string& save_dir) const {
        // save tokenizer
        tokenizer.save(save_dir + "/tokenizer");

        string vectorizer_folder = save_dir + "/vectorizer";
        if(mkdir(vectorizer_folder.c_str(), 0777) == -1) {
            if(errno != EEXIST) {
                throw std::runtime_error("Unable to create save folder at " + vectorizer_folder);
            }
        }
        // save param
        param.save(vectorizer_folder + "/config.json");
        // save feature_vocab and idx_idf

        auto model_filename = vectorizer_folder + "/tfidf-model.txt";
        FILE *fp = fopen(model_filename.c_str(), "w");
        if(fp == NULL) {
            throw std::runtime_error("Unable to save tfidf model file to " + model_filename);
        } else {
            fprintf(fp, "%ld\n", nr_features());
            for(auto iter = feature_vocab.begin(); iter != feature_vocab.end(); iter++) {
                auto& feat_ngram = iter->first;
                int32_t feat_id = iter->second;
                // feat_id<TAB>feat_idf<TAB>ngram_length<TAB>idx1 idx2...
                fprintf(fp, "%d\t%f\t%ld", feat_id, idx_idf.at(feat_id), feat_ngram.size());
                for(size_t tid = 0; tid < feat_ngram.size(); tid++) {
                    if(tid == 0) {
                        fprintf(fp, "\t%d", feat_ngram[tid]);
                    } else {
                        fprintf(fp, " %d", feat_ngram[tid]);
                    }
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    void load(const string& load_dir) {
        // load tokenizer
        tokenizer.load(load_dir + "/tokenizer");

        string vectorizer_folder = load_dir + "/vectorizer";
        // load param
        param.load(vectorizer_folder + "/config.json");
        // load feature_vocab and idx_idf
        auto model_filename = vectorizer_folder + "/tfidf-model.txt";

        FILE *fp = fopen(model_filename.c_str(), "r");
        if(fp == NULL) {
            throw std::runtime_error("Unable to load tfidf model file to " + model_filename);
        } else {
            size_t total_features = 0;
            if (1 != fscanf(fp, "%ld", &total_features)) {
                throw std::runtime_error("Invalid tfidf model file (total_features).");
            }
            feature_vocab.reserve(total_features);
            idx_idf.reserve(total_features);

            for(size_t f = 0; f < total_features; f++) {
                // feat_id<TAB>feat_idf<TAB>ngram_length<TAB>idx1 idx2...
                int32_t idx = 0;
                float32_t idf = 0.0;
                uint64_t ngram_len = 0;
                if (3 != fscanf(fp, "%d%f%ld", &idx, &idf, &ngram_len)) {
                    throw std::runtime_error("Invalid tfidf model file (idx, idf, ngram_len).");
                }
                idx_idf[idx] = idf;
                idx_vec_t ngram(ngram_len);
                for(size_t tid = 0; tid < ngram_len; tid++) {
                    int32_t tok_idx;
                    if (1 != fscanf(fp, "%d", &tok_idx)) {
                        throw std::runtime_error("Invalid tfidf model file (tok_idx).");
                    }
                    ngram[tid] = tok_idx;
                }
                feature_vocab[ngram] = idx;
            }

            fclose(fp);
        }
    }

private:
    // train from a data chunk and count document frequency into feat_df
    void train_feat_df_chunk_(
        const sv_vec_t& corpus,
        vec2idx_map_t& feat_df,
        size_t start_line = 0,
        size_t end_line = 0) {

        if(end_line <= start_line || end_line > corpus.size()) {
            end_line = corpus.size();
        }

        for(auto line_sv = corpus.begin() + start_line; line_sv != corpus.begin() + end_line; line_sv++) {
            vec2idx_map_t feat_cnt;
            tokenizer.count_ngrams(*line_sv, feat_cnt, param.min_ngram, param.max_ngram, param.max_length);
            // for document frequency, only record binarized count
            for(auto& feat_i : feat_cnt) {
                feat_df[feat_i.first] += 1;
            }
        }
    }

    // return the sorted feature vector of given string_view
    template<typename IDX_T, typename VAL_T>
    void get_sorted_feature(const string_view& line, vector<IDX_T>& feat_idx, vector<VAL_T>& feat_val) const {
        idx_vec_t tokens;
        tokenizer.tokenize(line, tokens, param.max_length);
        unordered_map<IDX_T, VAL_T> term_freq_map;
        // find all features between min_ngram and max_ngram
        idx_vec_t feat_key;
        feat_key.reserve(param.max_ngram);
        for(int cur_ngram = param.min_ngram; cur_ngram <= std::min(param.max_ngram, (int)tokens.size()); cur_ngram++) {
            feat_key.resize(cur_ngram);
            for(int i = 0; i <= (int) tokens.size() - cur_ngram; i++) {
                feat_key.assign(tokens.begin() + i, tokens.begin() + i + cur_ngram);
                auto feat_pair = feature_vocab.find(feat_key);
                if(feat_pair != feature_vocab.end()) {
                    term_freq_map[IDX_T(feat_pair->second)] += 1.0;
                }
            }
        }
        // sort indices
        vector<pair<IDX_T, VAL_T>> term_freq(term_freq_map.begin(), term_freq_map.end());
        std::sort(term_freq.begin(), term_freq.end());

        // convert term frequency to feature
        VAL_T normalizing_denominator = 0.0;
        for(auto& dim2feat : term_freq) {
            dim2feat.second = param.binary ? 1.0 : dim2feat.second;
            dim2feat.second = param.sublinear_tf ? log(dim2feat.second) + 1.0 : dim2feat.second;
            if(param.use_idf) {
                dim2feat.second *= idx_idf.at(dim2feat.first);
            }

            if(param.norm_p == 1) {
                normalizing_denominator += std::fabs(dim2feat.second);
            } else if(param.norm_p == 2) {
                normalizing_denominator += dim2feat.second * dim2feat.second;
            } else {
                throw std::invalid_argument("invalid normalize option, norm_p: [ 1| 2]");
            }
        }
        if(std::fabs(normalizing_denominator) < std::numeric_limits<float>::epsilon()) {
            normalizing_denominator = 1.0;
        } else if(param.norm_p == 2) {
            normalizing_denominator = std::sqrt(normalizing_denominator);
        }
        for(auto& dim2feat : term_freq) {
            dim2feat.second /= normalizing_denominator;
        }
        feat_idx.resize(term_freq.size());
        feat_val.resize(term_freq.size());
        for(size_t i = 0; i < term_freq.size(); i++) {
            feat_idx[i] = term_freq[i].first;
            feat_val[i] = term_freq[i].second;
        }
    }

    void train_from_file_(const string& corpus_path, vector<vec2idx_map_t>& feat_df_chunks, vector<vector<char>>& buffer, size_t chunk_size) {
        vector<size_t> chunk_offset;
        file_util::get_file_offset(corpus_path, chunk_size, chunk_offset);
        size_t n_chunks = chunk_offset.size() - 1;

#pragma omp parallel for schedule(dynamic,1)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            int proc_id = omp_get_thread_num();

            if(buffer[proc_id].size() <= chunk_offset[chunk + 1] - chunk_offset[chunk]) {
                // need to increase buffer size
                buffer[proc_id].resize(chunk_offset[chunk + 1] - chunk_offset[chunk] + 1);
            }

            // load file chunk and parse lines to string_views
            sv_vec_t cur_corpus_sv;
            size_t cache_size = file_util::load_file_block(corpus_path,
                    buffer[proc_id].data(), chunk_offset[chunk], chunk_offset[chunk + 1]);
            append_lines_to_string_view(buffer[proc_id].data(), cache_size, cur_corpus_sv);

            train_feat_df_chunk_(cur_corpus_sv, feat_df_chunks[proc_id]);
        }
    }

    // train tfidf vectorizer from corpus in memory
    void train_from_mem_(const sv_vec_t& corpus, vector<vec2idx_map_t>& feat_df_chunks) {
        size_t nr_doc = corpus.size();
        size_t n_chunks = std::min(feat_df_chunks.size(), corpus.size());
        size_t chunk_size = (nr_doc + n_chunks - 1) / n_chunks;

#pragma omp parallel for schedule(dynamic,1)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            size_t start_line = chunk * chunk_size;
            if(start_line < nr_doc) {
                size_t end_line = std::min(start_line + chunk_size, nr_doc);
                train_feat_df_chunk_(corpus, feat_df_chunks[chunk], start_line, end_line);
            }
        }
    }

    // merge and sort features
    void merge_df_chunks(vector<vec2idx_map_t>& feat_df_chunks, size_t nr_doc, int threads) {
        size_t n_chunks = feat_df_chunks.size();

        // only keep (real_min_df_cnt, real_max_df_cnt)
        size_t real_min_df_cnt = (size_t)std::round(param.min_df_ratio * nr_doc);
        real_min_df_cnt = std::max((size_t)param.min_df_cnt, real_min_df_cnt);
        size_t real_max_df_cnt = (size_t)std::round(param.max_df_ratio * nr_doc);
        real_max_df_cnt = param.max_df_cnt > 0 ? std::min((size_t)param.max_df_cnt, real_max_df_cnt) : real_max_df_cnt;

        // merge df counts to its first chunk
        auto& final_chunk =  feat_df_chunks[0];
        for(size_t ck_idx = 1; ck_idx < n_chunks; ck_idx++) {
            for(auto& cur_cnt : feat_df_chunks[ck_idx]) {
                auto ngram_feat = cur_cnt.first;
                auto df_cnt = cur_cnt.second;
                final_chunk[ngram_feat] += df_cnt;
            }
            // make sure memory is released
            feat_df_chunks[ck_idx].clear();
            vec2idx_map_t().swap(feat_df_chunks[ck_idx]);
        }
        // filtering features with min_df, max_df
        for(auto it = final_chunk.begin(); it != final_chunk.end();) {
            auto fidx = static_cast<size_t>(it->second);
            if(fidx < real_min_df_cnt || fidx > real_max_df_cnt) {
                // remove this feature
                it = final_chunk.erase(it);
            } else {
                it++;
            }
        }

        // vector of cnt2ptr for sort
        typedef vec2idx_map_t::value_type* vec2idx_pair_p;
        vector<vec2idx_pair_p> ptr_vec;
        for(auto it = final_chunk.begin(); it != final_chunk.end(); ++it) {
            // keep this feature
            vec2idx_pair_p ptr = &(*it);
            ptr_vec.push_back(ptr);
        }

        size_t nr_feat = 0;
        if(param.max_feature > 0) {
            nr_feat = std::min(size_t(param.max_feature), ptr_vec.size());
        } else {
            nr_feat = ptr_vec.size();
        }
        // sort the remaining features by their cnt in ascending order
        // if cnt equals, sort by ngram length in ascending order
        // if ngram length equals, sort by token indices in ascending order
        parallel_sort(ptr_vec.begin(), ptr_vec.end(),
            [&](const vec2idx_pair_p& lx, const vec2idx_pair_p& rx) -> bool {
                if(lx->second != rx->second) { // using the count to compare
                    return lx->second < rx->second;
                } else if(lx->first.size() != rx->first.size()) {
                    return lx->first.size() < rx->first.size(); // compare ngram length
                } else {// compare ngram token idx
                    for(auto i = 0U; i < lx->first.size() - 1; i++) {
                        if(lx->first[i] != rx->first[i]) {
                            return lx->first[i] < rx->first[i];
                        }
                    }
                    return lx->first.back() < rx->first.back();
                }
            },
            threads
        );

        // trim features by removing
        // the least frequent ones (if keep_frequent_feature)
        // or the most frequent ones
        // convert feat_key2df to feat_key2idx + idx2idf
        feature_vocab.reserve(nr_feat);
        idx_idf.reserve(nr_feat);

        idx_type start_idx = 0;
        if(param.keep_frequent_feature) {
            start_idx = ptr_vec.size() - nr_feat;
        }

        for(size_t cur_idx = 0; cur_idx < nr_feat; cur_idx++) {
            auto& cur_ptr = ptr_vec[cur_idx + start_idx];
            auto cur_df = final_chunk[cur_ptr->first];
            feature_vocab[cur_ptr->first] = static_cast<idx_type>(cur_idx);
            idx_idf[cur_idx] = std::max(log(float(nr_doc) / (cur_df + (size_t)param.smooth_idf)), 0.0) + float(param.add_one_idf);
        }
    }

public:
    // train from corpus in memory
    void train(const char** corpus, const size_t* doc_lens, size_t nr_doc, int threads=-1) {
        // create reference with string_view, no copy
        sv_vec_t corpus_sv_vec(nr_doc);
        for(size_t i = 0; i < nr_doc; i++) {
            string_view cur_doc(corpus[i], doc_lens[i]);
            corpus_sv_vec[i] = cur_doc;
        }
        train(corpus_sv_vec, threads);
    }

    void train(const sv_vec_t& corpus, int threads=-1) {
        tokenizer.train(corpus, threads);

        threads = set_threads(threads);
        vector<vec2idx_map_t> feat_df_chunks(threads);

        train_from_mem_(corpus, feat_df_chunks);
        merge_df_chunks(feat_df_chunks, corpus.size(), threads);
    }

    // train from a single file
    void train_from_file(const string& corpus_file, size_t buffer_size=0, int threads=-1) {
        str_vec_t corpus_files{corpus_file};
        train_from_file(corpus_files, buffer_size, threads);
    }

    // train from multiple files
    void train_from_file(const char** corpus_files, const size_t* fname_lens, size_t nr_files, size_t buffer_size=0, int threads=-1) {
        str_vec_t corpus_files_vec(nr_files);
        for(size_t i = 0; i < nr_files; i++) {
            string cur_fname(corpus_files[i], fname_lens[i]);
            corpus_files_vec[i] = cur_fname;
        }
        train_from_file(corpus_files_vec, buffer_size, threads);
    }

    void train_from_file(const str_vec_t& corpus_files, size_t buffer_size=0, int threads=-1) {
        buffer_size = std::max(DEFAULT_BUFFER_SIZE, buffer_size);
        // train tokenizer and build vocabulary
        tokenizer.train_from_file(corpus_files, buffer_size, threads);

        threads = set_threads(threads);
        vector<vec2idx_map_t> feat_df_chunks(threads);
        size_t nr_doc = 0;

        // workspace memory to load data
        size_t proc_buf_size = buffer_size / threads;
        vector<vector<char>> buffer(threads);
        for(int i = 0; i < threads; i++) {
            buffer[i].resize(proc_buf_size);
        }

        for(auto& cur_corpus_file : corpus_files) {
            nr_doc += file_util::get_linecount(cur_corpus_file);
            size_t cur_file_size = file_util::get_filesize(cur_corpus_file);
            if(cur_file_size < buffer[0].size() - 1) {
                // load file chunk and parse lines to string_views
                sv_vec_t cur_corpus_sv;
                size_t cache_size = file_util::load_file_block(cur_corpus_file, buffer[0].data());
                append_lines_to_string_view(buffer[0].data(), cache_size, cur_corpus_sv);
                train_from_mem_(cur_corpus_sv, feat_df_chunks);
            } else {
                // load file in chunks and parallel training
                // chunk_size may increase depending on the position of next \n
                // to avoid resizing buffer, use proc_buf_size / 2 as chunk_size
                train_from_file_(cur_corpus_file, feat_df_chunks, buffer, proc_buf_size / 2);
            }
        }
        merge_df_chunks(feat_df_chunks, nr_doc, threads);
    }

    // batch inference from single file
    // expect res to be empty csr_t or spmm_mat_t
    template<class MAT_T>
    void predict_from_file(const char* corpus_file, const size_t fname_len, MAT_T& res, size_t buffer_size=0, int threads=-1) const {
        // create reference with string_view, no copy
        string corpus_file_str(corpus_file, fname_len);
        predict_from_file(corpus_file_str, res, buffer_size, threads);
    }

    template<class MAT_T>
    void predict_from_file(const string& corpus_file, MAT_T& res, size_t buffer_size=0, int threads=-1) const {
        typedef typename MAT_T::index_type ret_idx_t;
        typedef typename MAT_T::value_type ret_val_t;
        typedef typename MAT_T::mem_index_type ret_indptr_t;

        buffer_size = std::max(DEFAULT_BUFFER_SIZE, buffer_size);
        threads = set_threads(threads);
        // workspace memory to load data
        size_t proc_buf_size = buffer_size / threads;
        vector<vector<char>> buffer(threads);
        for(int i = 0; i < threads; i++) {
            buffer[i].resize(proc_buf_size);
        }
        vector<size_t> chunk_offset;
        file_util::get_file_offset(corpus_file, proc_buf_size / 2, chunk_offset);
        size_t n_chunks = chunk_offset.size() - 1;

        vector<size_t> chunk_nnz(n_chunks + 1, 0);

        vector<vector<ret_indptr_t>> feat_indptr_vec(n_chunks);
        vector<vector<ret_idx_t>> feat_indices_vec(n_chunks);
        vector<vector<ret_val_t>> feat_data_vec(n_chunks);
        vector<size_t> chunk_nr_doc(n_chunks);

#pragma omp parallel for schedule(dynamic)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            int proc_id = omp_get_thread_num();
            size_t start_pos = chunk_offset[chunk];
            size_t end_pos = chunk_offset[chunk + 1];

            if(buffer[proc_id].size() <= end_pos - start_pos) {
                // need to increase buffer size
                buffer[proc_id].resize(end_pos - start_pos + 1);
            }
            sv_vec_t cur_corpus_sv;
            size_t cache_size = file_util::load_file_block(corpus_file,
                    buffer[proc_id].data(), chunk_offset[chunk], chunk_offset[chunk + 1]);

            append_lines_to_string_view(buffer[proc_id].data(), cache_size, cur_corpus_sv);
            chunk_nr_doc[chunk] = cur_corpus_sv.size();

            for(size_t qi = 0; qi < cur_corpus_sv.size(); qi++) {
                vector<ret_idx_t> feat_indices;
                vector<ret_val_t> feat_data;
                get_sorted_feature(cur_corpus_sv[qi], feat_indices, feat_data);

                size_t cur_nnz = feat_data.size();

                chunk_nnz[chunk + 1] += cur_nnz;
                feat_indptr_vec[chunk].push_back(cur_nnz);
                feat_indices_vec[chunk].insert(feat_indices_vec[chunk].end(), feat_indices.begin(), feat_indices.end());
                feat_data_vec[chunk].insert(feat_data_vec[chunk].end(), feat_data.begin(), feat_data.end());
            }
        }

        size_t nr_doc = std::accumulate(chunk_nr_doc.begin(), chunk_nr_doc.end(), 0);
        std::partial_sum(chunk_nnz.begin(), chunk_nnz.end(), chunk_nnz.begin());
        vector<ret_indptr_t> feat_sizes(nr_doc + 1, 0);
        size_t cp_ptr = 1;
        // merge feat_indptr_vec to one
        for(auto& cur_feat_sizes : feat_indptr_vec) {
            std::copy(cur_feat_sizes.begin(), cur_feat_sizes.end(), feat_sizes.begin() + cp_ptr);
            cp_ptr += cur_feat_sizes.size();
            // make sure memory is cleared
            cur_feat_sizes.clear();
            vector<ret_indptr_t>().swap(cur_feat_sizes);
        }
        parallel_partial_sum(feat_sizes.begin(), feat_sizes.end(), feat_sizes.begin(), threads);

        size_t total_nnz = feat_sizes[nr_doc];

        res.allocate(nr_doc, idx_idf.size(), total_nnz);
        std::memcpy(res.indptr, feat_sizes.data(), sizeof(ret_indptr_t) * (nr_doc + 1));

#pragma omp parallel for schedule(dynamic)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            size_t start = chunk_nnz[chunk];
            size_t end = chunk_nnz[chunk + 1];
            std::memcpy(&res.data[start], feat_data_vec[chunk].data(), sizeof(ret_val_t) * (end - start));
            std::memcpy(&res.indices[start], feat_indices_vec[chunk].data(), sizeof(ret_idx_t) * (end - start));
        }
    }

    // batch inference with corpus in memory
    // expect res to be empty csr_t or spmm_mat_t
    template<class MAT_T>
    void predict(const char** corpus, const size_t* doc_lens, size_t nr_doc, MAT_T& res, int threads=-1) const {
        // create reference with string_view, no copy
        sv_vec_t corpus_vec(nr_doc);
        for(size_t i = 0; i < nr_doc; i++) {
            string_view cur_doc(corpus[i], doc_lens[i]);
            corpus_vec[i] = cur_doc;
        }
        predict(corpus_vec, res, threads);
    }

    template<class MAT_T>
    void predict(const sv_vec_t& corpus, MAT_T& res, int threads=-1) const {
        typedef typename MAT_T::index_type ret_idx_t;
        typedef typename MAT_T::value_type ret_val_t;
        typedef typename MAT_T::mem_index_type ret_indptr_t;

        threads = set_threads(threads);

        size_t nr_doc = corpus.size();
        size_t n_chunks = threads;
        size_t chunk_size = (nr_doc + n_chunks - 1) / n_chunks;

        vector<ret_indptr_t> feat_sizes(nr_doc + 1, 0);
        vector<size_t> chunk_nnz(n_chunks + 1, 0);
        vector<vector<ret_idx_t>> feat_indices_vec(n_chunks);
        vector<vector<ret_val_t>> feat_data_vec(n_chunks);

#pragma omp parallel for schedule(static)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            size_t start_line = chunk * chunk_size;
            size_t end_line = std::min(start_line + chunk_size, nr_doc);

            for(size_t qi = start_line; qi < end_line; qi++) {
                vector<ret_idx_t> feat_indices;
                vector<ret_val_t> feat_data;
                get_sorted_feature(corpus[qi], feat_indices, feat_data);

                size_t cur_nnz = feat_data.size();
                feat_sizes[qi + 1] = cur_nnz;

                chunk_nnz[chunk + 1] += cur_nnz;
                feat_indices_vec[chunk].insert(feat_indices_vec[chunk].end(), feat_indices.begin(), feat_indices.end());
                feat_data_vec[chunk].insert(feat_data_vec[chunk].end(), feat_data.begin(), feat_data.end());
            }
        }
        parallel_partial_sum(feat_sizes.begin(), feat_sizes.end(), feat_sizes.begin(), threads);
        // chunk_nnz only need single thread partial_sum
        std::partial_sum(chunk_nnz.begin(), chunk_nnz.end(), chunk_nnz.begin());
        size_t total_nnz = feat_sizes[nr_doc];

        res.allocate(corpus.size(), idx_idf.size(), total_nnz);
        std::memcpy(res.indptr, feat_sizes.data(), sizeof(ret_indptr_t) * (nr_doc + 1));

#pragma omp parallel for schedule(static)
        for(size_t chunk = 0; chunk < n_chunks; chunk++) {
            size_t start = chunk_nnz[chunk];
            size_t end = chunk_nnz[chunk + 1];
            std::memcpy(&res.data[start], feat_data_vec[chunk].data(), sizeof(ret_val_t) * (end - start));
            std::memcpy(&res.indices[start], feat_indices_vec[chunk].data(), sizeof(ret_idx_t) * (end - start));
        }
    }

    // transform input string to feature vector
    // returns feature vector with sorted indices
    template<class MAT_T>
    void predict(const string_view& line, MAT_T& res) const {
        typedef typename MAT_T::index_type ret_idx_t;
        typedef typename MAT_T::value_type ret_val_t;

        vector<ret_idx_t> feature_idx;
        vector<ret_val_t> feature_val;
        get_sorted_feature(line, feature_idx, feature_val);
        size_t data_size = feature_idx.size();

        // construct result csr vector
        res.allocate(1, idx_idf.size(), data_size);
        res.indptr[0] = 0;
        res.indptr[1] = data_size;
        std::memcpy(res.data, feature_val.data(), sizeof(ret_val_t) * data_size);
        std::memcpy(res.indices, feature_idx.data(), sizeof(ret_idx_t) * data_size);
    }

}; // end BaseVectorizer

class Vectorizer {

public:

    // arguments
    TfidfVectorizerParam param;
    vector<BaseVectorizer> vectorizer_arr;

    Vectorizer() {};

    Vectorizer(const TfidfVectorizerParam* param_ptr): param(*param_ptr) {
        for (int i = 0; i < param.num_base_vect; i++) {
            vectorizer_arr.emplace_back(BaseVectorizer(&param.base_param_ptr[i]));
        }
    }

    Vectorizer(const string& filepath) { load(filepath); }

    void save(const string& save_dir) const {
        // save TfidfVectorizerParam
        string meta_json_path = save_dir + "/meta.json";
        param.save(meta_json_path);

        // save TfidfBaseVectorizer
        for (int i = 0; i < param.num_base_vect; i++) {
            string base_vect_dir = save_dir + "/" + std::to_string(i) + ".base";
            if(mkdir(base_vect_dir.c_str(), 0777) == -1) {
                if(errno != EEXIST) {
                    throw std::runtime_error("Unable to create base_vect_dir at " + base_vect_dir);
                }
            }
            vectorizer_arr[i].save(base_vect_dir);
        }
    }

    void load(const string& load_dir) {
        // check whether "load_dir/meta.json" exists.
        // If not exist, load_dir is saved from BaseVectorizer
        string meta_json_path = load_dir + "/meta.json";
        std::ifstream file_stream(meta_json_path);
        if (!file_stream.is_open()) {
            param.num_base_vect = 1;
            vectorizer_arr.resize(param.num_base_vect);
            vectorizer_arr[0].load(load_dir);
            param.norm_p = vectorizer_arr[0].param.norm_p;
            return;
        }
        // Otherwise, load_dir is saved from Vectorizer
        param.load(load_dir + "/meta.json");
        vectorizer_arr.resize(param.num_base_vect);

        for (int i = 0; i < param.num_base_vect; i++) {
            string base_vect_dir = load_dir + "/" + std::to_string(i) + ".base";
            vectorizer_arr[i].load(base_vect_dir);
        }
    }

    size_t nr_features() const {
        size_t total_features = 0;
        for(const auto& vectorizer : vectorizer_arr) {
            total_features += vectorizer.nr_features();
        }
        return total_features;
    }

    // train from corpus in memory
    void train(const char** corpus, const size_t* doc_lens, size_t nr_doc, int threads=-1) {
        // create reference with string_view, no copy
        sv_vec_t corpus_sv_vec(nr_doc);
        for(size_t i = 0; i < nr_doc; i++) {
            string_view cur_doc(corpus[i], doc_lens[i]);
            corpus_sv_vec[i] = cur_doc;
        }
        train(corpus_sv_vec, threads);
    }

    void train(const sv_vec_t& corpus, int threads=-1) {
        for(auto& vectorizer : vectorizer_arr) {
            vectorizer.train(corpus, threads);
        }
    }

    // train from a single file
    void train_from_file(const string& corpus_file, size_t buffer_size=0, int threads=-1) {
        str_vec_t corpus_files{corpus_file};
        train_from_file(corpus_files, buffer_size, threads);
    }

    // train from multiple files
    void train_from_file(const char** corpus_files, const size_t* fname_lens, size_t nr_files, size_t buffer_size=0, int threads=-1) {
        str_vec_t corpus_files_vec(nr_files);
        for(size_t i = 0; i < nr_files; i++) {
            string cur_fname(corpus_files[i], fname_lens[i]);
            corpus_files_vec[i] = cur_fname;
        }
        train_from_file(corpus_files_vec, buffer_size, threads);
    }

    void train_from_file(const str_vec_t& corpus_files, size_t buffer_size=0, int threads=-1) {
        for(auto& vectorizer : vectorizer_arr) {
            vectorizer.train_from_file(corpus_files, buffer_size, threads);
        }
    }

    // expect res to be empty csr_t or spmm_mat_t
    template<class MAT_T>
    void normalize_csr(MAT_T& res, int norm_p, int threads) const {
        typedef typename MAT_T::value_type ret_val_t;
        set_threads(threads);
        if(norm_p == 1) {
#pragma omp parallel for schedule(dynamic)
            for(unsigned i = 0; i < res.rows; i++) {
                ret_val_t normalizing_denominator = 0.0;
                for(auto j = res.indptr[i]; j < res.indptr[i + 1]; j++) {
                    normalizing_denominator += std::fabs(res.data[j]);
                }
                if(std::fabs(normalizing_denominator) < std::numeric_limits<float>::epsilon()) {
                    normalizing_denominator = 1.0;
                }
                for(auto j = res.indptr[i]; j < res.indptr[i + 1]; j++) {
                    res.data[j] /= normalizing_denominator;
                }
            }
        } else if(norm_p == 2) {
#pragma omp parallel for schedule(dynamic)
            for(unsigned i = 0; i < res.rows; i++) {
                ret_val_t normalizing_denominator = 0.0;
                for(auto j = res.indptr[i]; j < res.indptr[i + 1]; j++) {
                    normalizing_denominator += res.data[j] * res.data[j];
                }
                if(std::fabs(normalizing_denominator) < std::numeric_limits<float>::epsilon()) {
                    normalizing_denominator = 1.0;
                } else{
                    normalizing_denominator = std::sqrt(normalizing_denominator);
                }
                for (auto j = res.indptr[i]; j < res.indptr[i + 1]; j++) {
                    res.data[j] /= normalizing_denominator;
                }
            }
        } else {
            throw std::invalid_argument("invalid normalize option, norm_p: [ 1| 2]");
        }
    }

    // batch inference from single file
    // expect res to be empty csr_t or spmm_mat_t
    template<class MAT_T>
    void predict_from_file(const char* corpus_file, const size_t fname_len, MAT_T& res, size_t buffer_size=0, int threads=-1) const {
        // create reference with string_view, no copy
        string corpus_file_str(corpus_file, fname_len);
        predict_from_file(corpus_file_str, res, buffer_size, threads);
    }

    template<class MAT_T>
    void predict_from_file(const string& corpus_file, MAT_T& res, size_t buffer_size=0, int threads=-1) const {
        // base case, no need to do the extra column-wise concatenate work
        if(param.num_base_vect == 1) {
            vectorizer_arr[0].predict_from_file(corpus_file, res, buffer_size, threads);
            if(param.norm_p != vectorizer_arr[0].param.norm_p) {
                normalize_csr(res, param.norm_p, threads);
            }
            return;
        }

        // do prediction from each TfidfBaseVectorizer
        // and save each feat_mat in feat_mat_arr
        vector<csr_t> feat_mat_arr(param.num_base_vect);
        for(int i = 0; i < param.num_base_vect; i++) {
            vectorizer_arr[i].predict_from_file(corpus_file, feat_mat_arr[i], buffer_size, threads);
        }
        // hstack feature sub-matrices, copy into res, and normalize
        hstack_csr(feat_mat_arr, res, threads);
        normalize_csr(res, param.norm_p, threads);

        for (auto& feat_mat: feat_mat_arr) {
            feat_mat.free_underlying_memory();
        }
    }

    // batch inference with corpus in memory
    // expect res to be empty csr_t or spmm_mat_t
    template<class MAT_T>
    void predict(const char** corpus, const size_t* doc_lens, size_t nr_doc, MAT_T& res, int threads=-1) const {
        // create reference with string_view, no copy
        sv_vec_t corpus_vec(nr_doc);
        for(size_t i = 0; i < nr_doc; i++) {
            string_view cur_doc(corpus[i], doc_lens[i]);
            corpus_vec[i] = cur_doc;
        }
        predict(corpus_vec, res, threads);
    }

    template<class MAT_T>
    void predict(const sv_vec_t& corpus, MAT_T& res, int threads=-1) const {
        // base case, no need to do the extra column-wise concatenate work
        if(param.num_base_vect == 1) {
            vectorizer_arr[0].predict(corpus, res, threads);
            if(param.norm_p != vectorizer_arr[0].param.norm_p) {
                normalize_csr(res, param.norm_p, threads);
            }
            return;
        }

        // do prediction from each TfidfBaseVectorizer
        // and save each feat_mat in feat_mat_arr
        vector<csr_t> feat_mat_arr(param.num_base_vect);
        for(int i = 0; i < param.num_base_vect; i++) {
            vectorizer_arr[i].predict(corpus, feat_mat_arr[i], threads);
        }
        // hstack feature sub-matrices, copy into res, and normalize
        hstack_csr(feat_mat_arr, res, threads);
        normalize_csr(res, param.norm_p, threads);

        for(auto& feat_mat: feat_mat_arr) {
            feat_mat.free_underlying_memory();
        }
    }

    // transform input string to feature vector
    // returns feature vector with sorted indices
    template<class MAT_T>
    void predict(const string_view& line, MAT_T& res) const {
        // threads = 1 to simulate online inference setting
        int threads = 1;

        // base case, no need to do the extra column-wise concatenate work
        if(param.num_base_vect == 1) {
            vectorizer_arr[0].predict(line, res);
            if(param.norm_p != vectorizer_arr[0].param.norm_p) {
                normalize_csr(res, param.norm_p, threads);
            }
            return;
        }

        // do prediction from each TfidfBaseVectorizer
        // and save each feat_mat in feat_mat_arr
        vector<csr_t> feat_mat_arr(param.num_base_vect);
        for(int i = 0; i < param.num_base_vect; i++) {
            vectorizer_arr[i].predict(line, feat_mat_arr[i]);
        }
        // hstack feature sub-matrices, copy into res, and normalize
        hstack_csr(feat_mat_arr, res, threads);
        normalize_csr(res, param.norm_p, threads);

        for(auto& feat_mat: feat_mat_arr) {
            feat_mat.free_underlying_memory();
        }
    }

}; // end Vectorizer


} // end namespace tfidf
} // end namespace pecos
#endif  // end of __TFIDF_H__
