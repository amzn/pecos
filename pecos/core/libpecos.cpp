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

#include "utils/clustering.hpp"
#include "utils/matrix.hpp"
#include "utils/tfidf.hpp"
#include "xmc/inference.hpp"
#include "xmc/linear_solver.hpp"

// ===== C Interface of Functions ======
// C Interface of Types/Structures can be found in utils/matrix.hpp

extern "C" {
    // ==== C Interface of XMC Models ====
    void* c_xlinear_load_model_from_disk(const char* model_path) {
        auto model = new pecos::HierarchicalMLModel(model_path);
        return static_cast<void*>(model);
    }

    void* c_xlinear_load_model_from_disk_ext(const char* model_path,
        int weight_matrix_type) {
        pecos::layer_type_t type = static_cast<pecos::layer_type_t>(weight_matrix_type);
        auto model = new pecos::HierarchicalMLModel(model_path, type);
        return static_cast<void*>(model);
    }

    void c_xlinear_destruct_model(void* ptr) {
        pecos::HierarchicalMLModel* mc = static_cast<pecos::HierarchicalMLModel*>(ptr);
        delete mc;
    }

    // Obtain attribute values of the model.
    // Allowed attr: depth, nr_features, nr_labels, nr_codes
    uint32_t c_xlinear_get_int_attr(void* ptr, const char* attr) {
        pecos::HierarchicalMLModel* mc = static_cast<pecos::HierarchicalMLModel*>(ptr);
        return mc->get_int_attr(attr);
    }

    pecos::layer_type_t c_xlinear_get_layer_type(void* ptr, int layer_depth) {
        pecos::HierarchicalMLModel* model = static_cast<pecos::HierarchicalMLModel*>(ptr);
        pecos::HierarchicalMLModel::ISpecializedModelLayer* layer = model->get_model_layers()[layer_depth];
        return layer->get_type();
    }

    #define C_XLINEAR_PREDICT(SUFFIX, PY_MAT, C_MAT) \
    void c_xlinear_predict ## SUFFIX( \
        void* ptr, \
        const PY_MAT* input_x, \
        const uint32_t overridden_beam_size, \
        const char* overridden_post_processor_str, \
        const uint32_t overridden_only_topk, \
        const int threads, \
        py_sparse_allocator_t pred_alloc) { \
        pecos::HierarchicalMLModel* mc = static_cast<pecos::HierarchicalMLModel*>(ptr); \
        C_MAT X(input_x); \
        pecos::csr_t result; \
        mc->predict(X, result, overridden_beam_size, overridden_post_processor_str, \
        overridden_only_topk, threads); \
        result.create_pycsr(pred_alloc); \
        result.free_underlying_memory(); \
    }
    C_XLINEAR_PREDICT(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_XLINEAR_PREDICT(_drm_f32, ScipyDrmF32, pecos::drm_t)


    #define C_XLINEAR_PREDICT_ON_SELECTED_OUTPUTS(SUFFIX, PY_MAT, C_MAT) \
    void c_xlinear_predict_on_selected_outputs ## SUFFIX( \
        void* ptr, \
        const PY_MAT* input_x, \
        const ScipyCsrF32* selected_outputs_csr, \
        const char* overridden_post_processor_str, \
        const int threads, \
        py_sparse_allocator_t pred_alloc) { \
        pecos::HierarchicalMLModel* mc = static_cast<pecos::HierarchicalMLModel*>(ptr); \
        C_MAT X(input_x); \
        pecos::csr_t curr_outputs_csr = pecos::csr_t(selected_outputs_csr).deep_copy(); \
        pecos::csr_t result; \
        mc->predict_on_selected_outputs(X, curr_outputs_csr, result, overridden_post_processor_str, \
        threads); \
        result.create_pycsr(pred_alloc); \
        result.free_underlying_memory(); \
        curr_outputs_csr.free_underlying_memory(); \
    }
    C_XLINEAR_PREDICT_ON_SELECTED_OUTPUTS(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_XLINEAR_PREDICT_ON_SELECTED_OUTPUTS(_drm_f32, ScipyDrmF32, pecos::drm_t)


    #define C_XLINEAR_SINGLE_LAYER_PREDICT(SUFFIX, PY_MAT, C_MAT) \
    void c_xlinear_single_layer_predict ## SUFFIX( \
        const PY_MAT* input_x, \
        const ScipyCsrF32* csr_codes, \
        ScipyCscF32* W, \
        ScipyCscF32* C, \
        const char* post_processor_str, \
        const uint32_t only_topk, \
        const int num_threads, \
        const float bias, \
        py_sparse_allocator_t pred_alloc) { \
        C_MAT X(input_x); \
        pecos::csr_t last_layer_pred; \
        bool is_first_layer; \
        if (csr_codes) { \
            last_layer_pred = pecos::csr_t(csr_codes).deep_copy(); \
            is_first_layer = false; \
        } else { \
            last_layer_pred.fill_ones(X.rows, 1); \
            is_first_layer = true; \
        } \
        pecos::csc_t C_; \
        C_ = pecos::csc_t(C); \
        pecos::csr_t cur_layer_pred; \
        pecos::csc_t W_ = pecos::csc_t(W); \
        pecos::MLModelMetadata metadata(bias, only_topk, post_processor_str); \
        pecos::MLModel<pecos::csc_t> layer(W_, C_, 0, false, metadata); \
        layer.predict(X, last_layer_pred, is_first_layer, only_topk, \
            post_processor_str, cur_layer_pred, num_threads); \
        cur_layer_pred.create_pycsr(pred_alloc); \
        cur_layer_pred.free_underlying_memory(); \
        last_layer_pred.free_underlying_memory(); \
    }
    C_XLINEAR_SINGLE_LAYER_PREDICT(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_XLINEAR_SINGLE_LAYER_PREDICT(_drm_f32, ScipyDrmF32, pecos::drm_t)


    #define C_XLINEAR_SINGLE_LAYER_PREDICT_ON_SELECTED_OUTPUTS(SUFFIX, PY_MAT, C_MAT) \
    void c_xlinear_single_layer_predict_on_selected_outputs ## SUFFIX( \
        const PY_MAT* input_x, \
        const ScipyCsrF32* selected_outputs_csr, \
        const ScipyCsrF32* csr_codes, \
        ScipyCscF32* W, \
        ScipyCscF32* C, \
        const char* post_processor_str, \
        const int num_threads, \
        const float bias, \
        py_sparse_allocator_t pred_alloc) { \
        C_MAT X(input_x); \
        pecos::csr_t curr_outputs_csr = pecos::csr_t(selected_outputs_csr).deep_copy(); \
        pecos::csr_t prev_layer_pred; \
        bool is_first_layer; \
        if (csr_codes) { \
            prev_layer_pred = pecos::csr_t(csr_codes).deep_copy(); \
            is_first_layer = false; \
        } else { \
            prev_layer_pred.fill_ones(X.rows, 1); \
            is_first_layer = true; \
        } \
        pecos::csc_t C_; \
        C_ = pecos::csc_t(C); \
        pecos::csr_t cur_layer_pred; \
        pecos::csc_t W_ = pecos::csc_t(W); \
        pecos::MLModelMetadata metadata(bias, 0, post_processor_str); \
        pecos::MLModel<pecos::csc_t> layer(W_, C_, 0, false, metadata); \
        layer.predict_on_selected_outputs(X, curr_outputs_csr, prev_layer_pred, is_first_layer, \
            post_processor_str, cur_layer_pred, num_threads); \
        cur_layer_pred.create_pycsr(pred_alloc); \
        cur_layer_pred.free_underlying_memory(); \
        curr_outputs_csr.free_underlying_memory(); \
        prev_layer_pred.free_underlying_memory(); \
    }
    C_XLINEAR_SINGLE_LAYER_PREDICT_ON_SELECTED_OUTPUTS(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_XLINEAR_SINGLE_LAYER_PREDICT_ON_SELECTED_OUTPUTS(_drm_f32, ScipyDrmF32, pecos::drm_t)


    #define C_XLINEAR_SINGLE_LAYER_TRAIN(SUFFIX, PY_MAT, C_MAT) \
    void c_xlinear_single_layer_train ## SUFFIX( \
        const PY_MAT *pX, \
        const ScipyCscF32 *pY, \
        const ScipyCscF32 *pC, \
        const ScipyCscF32 *pM, \
        const ScipyCscF32 *pR, \
        py_coo_allocator_t coo_alloc, \
        double threshold, \
        uint32_t max_nonzeros_per_label, \
        int solver_type, \
        double Cp, \
        double Cn, \
        size_t max_iter, \
        double eps, \
        double bias, \
        int threads) { \
        const C_MAT feat_mat(pX); \
        const pecos::csc_t Y(pY); \
        const pecos::csc_t& C = (pC == NULL) ? pecos::csc_t() : pecos::csc_t(pC); \
        const pecos::csc_t& M = (pM == NULL) ? pecos::csc_t() : pecos::csc_t(pM); \
        const pecos::csc_t& R = (pR == NULL) ? pecos::csc_t() : pecos::csc_t(pR); \
        pecos::linear_solver::SVMParameter param(solver_type, Cp, Cn, max_iter, eps, bias); \
        pecos::coo_t model; \
        pecos::linear_solver::multilabel_train_with_codes(\
            &feat_mat, \
            &Y, \
            (pC == NULL) ? NULL : &C, \
            (pM == NULL) ? NULL : &M, \
            (pR == NULL) ? NULL : &R, \
            &model, \
            threshold, \
            max_nonzeros_per_label, \
            &param, \
            threads \
        ); \
        model.create_pycoo(coo_alloc); \
    }
    C_XLINEAR_SINGLE_LAYER_TRAIN(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_XLINEAR_SINGLE_LAYER_TRAIN(_drm_f32, ScipyDrmF32, pecos::drm_t)

    // ==== C Interface of Sparse Matrix/Vector Operations ====

    #define C_SPARSE_MATMUL(SUFFIX, PY_MAT, C_MAT) \
    void c_sparse_matmul ## SUFFIX( \
        const PY_MAT* pX, \
        const PY_MAT* pY, \
        py_sparse_allocator_t pred_alloc, \
        const bool eliminate_zeros, \
        const bool sorted_indices, \
        int threads) { \
        C_MAT X(pX); \
        C_MAT Y(pY); \
        pecos::spmm_mat_t<C_MAT::IS_COLUMN_MAJORED> Z(pred_alloc); \
        smat_x_smat(X, Y, Z, eliminate_zeros, sorted_indices, threads); \
    }
    C_SPARSE_MATMUL(_csc_f32, ScipyCscF32, pecos::csc_t)
    C_SPARSE_MATMUL(_csr_f32, ScipyCsrF32, pecos::csr_t)


    #define C_SPARSE_INNER_PRODUCTS(SUFFIX, PY_MAT, C_MAT) \
    void c_sparse_inner_products ## SUFFIX( \
        const PY_MAT *pX, \
        const ScipyCscF32 *pW, \
        uint64_t len, \
        uint32_t *X_row_idx, \
        uint32_t *W_col_idx, \
        float32_t *val, \
        int threads) { \
        C_MAT X(pX); \
        pecos::csc_t W(pW); \
        compute_sparse_entries_from_rowmajored_X_and_colmajored_M( \
            X, W, len, X_row_idx, W_col_idx, val, threads \
        ); \
    }
    C_SPARSE_INNER_PRODUCTS(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_SPARSE_INNER_PRODUCTS(_drm_f32, ScipyDrmF32, pecos::drm_t)

    // ==== C Interface of Clustering ====

    #define C_RUN_CLUSTERING(SUFFIX, PY_MAT, C_MAT) \
    void c_run_clustering ## SUFFIX( \
        const PY_MAT* py_mat_ptr, \
        uint32_t depth, \
        uint32_t partition_algo, \
        int seed, \
        uint32_t max_iter, \
        int threads, \
        uint32_t* label_codes) { \
        C_MAT feat_mat(py_mat_ptr); \
        pecos::clustering::Tree tree(depth); \
        tree.run_clustering(feat_mat, partition_algo, seed, label_codes, max_iter, threads); \
    }
    C_RUN_CLUSTERING(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_RUN_CLUSTERING(_drm_f32, ScipyDrmF32, pecos::drm_t)

    // ==== C Interface of TFIDF vectorizer ====

    void* c_tfidf_train_from_file(
        void* corpus_files_ptr,
        const size_t* fname_lens,
        size_t nr_files,
        const pecos::tfidf::TfidfVectorizerParam* param_ptr,
        size_t buffer_size,
        int threads) {
        const char** corpus_files = static_cast<const char**>(corpus_files_ptr);
        pecos::tfidf::Vectorizer* vect = new pecos::tfidf::Vectorizer(param_ptr);
        vect->train_from_file(corpus_files, fname_lens, nr_files, buffer_size, threads);
        return static_cast<void*>(vect);
    }

    void* c_tfidf_train(
        void* corpus_ptr,
        const size_t* doc_lens,
        size_t nr_doc,
        const pecos::tfidf::TfidfVectorizerParam* param_ptr,
        int threads) {
        const char** corpus = static_cast<const char**>(corpus_ptr);
        pecos::tfidf::Vectorizer* vect = new pecos::tfidf::Vectorizer(param_ptr);
        vect->train(corpus, doc_lens, nr_doc, threads);
        return static_cast<void*>(vect);
    }

    void* c_tfidf_load(const char* model_dir) {
        pecos::tfidf::Vectorizer* vect = new pecos::tfidf::Vectorizer(model_dir);
        return static_cast<void*>(vect);
    }

    void c_tfidf_save(void* ptr, const char* model_dir) {
        pecos::tfidf::Vectorizer* vect = static_cast<pecos::tfidf::Vectorizer*>(ptr);
        vect->save(model_dir);
    }

    void c_tfidf_destruct(void* ptr) {
        pecos::tfidf::Vectorizer* vect = static_cast<pecos::tfidf::Vectorizer*>(ptr);
        delete vect;
    }

    void c_tfidf_predict_from_file(
        void* ptr,
        void* corpus_fname_ptr,
        size_t fname_len,
        size_t buffer_size,
        int threads,
        py_sparse_allocator_t pred_alloc) {
        pecos::tfidf::Vectorizer* vect = static_cast<pecos::tfidf::Vectorizer*>(ptr);
        const char* corpus = static_cast<const char*>(corpus_fname_ptr);
        pecos::spmm_mat_t<false> feat_mat(pred_alloc);
        vect->predict_from_file(corpus, fname_len, feat_mat, buffer_size, threads);
    }

    void c_tfidf_predict(
        void* ptr,
        void* corpus_ptr,
        const size_t* doc_lens,
        size_t nr_doc,
        int threads,
        py_sparse_allocator_t pred_alloc) {
        pecos::tfidf::Vectorizer* vect = static_cast<pecos::tfidf::Vectorizer*>(ptr);
        const char** corpus = static_cast<const char**>(corpus_ptr);
        pecos::spmm_mat_t<false> feat_mat(pred_alloc);
        if(nr_doc > 1) {
            vect->predict(corpus, doc_lens, nr_doc, feat_mat, threads);
        } else if(nr_doc == 1) {
            std::string_view cur_doc(corpus[0], doc_lens[0]);
            vect->predict(cur_doc, feat_mat);
        } else {
            throw std::runtime_error("Invalid nr_doc " + std::to_string(nr_doc));
        }
    }
}