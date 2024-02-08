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
#include "utils/mmap_hashmap.hpp"
#include "utils/mmap_valstore.hpp"
#include "utils/tfidf.hpp"
#include "utils/parallel.hpp"
#include "xmc/inference.hpp"
#include "xmc/linear_solver.hpp"
#include "ann/feat_vectors.hpp"
#include "ann/hnsw.hpp"
#include "ann/pairwise.hpp"

// ===== C Interface of Functions ======
// C Interface of Types/Structures can be found in utils/matrix.hpp

extern "C" {
    // ==== C Interface of MLModels ====
    // Only implemented for w_matrix_t = pecos::csc_t
    typedef pecos::csc_t MLMODEL_MAT_T;
    void c_mlmodel_compile_mmap_model(const char* model_path, const char* mmap_model_path) {
        auto model = new pecos::MLModel<MLMODEL_MAT_T>(model_path, 0);
        model->save_mmap(mmap_model_path);
	delete model;
    }
    void* c_mlmodel_load_mmap_model(const char* model_path, const bool lazy_load) {
        auto mlm = new pecos::MLModel<MLMODEL_MAT_T>(model_path, 0, lazy_load);
        return static_cast<void*>(mlm);
    }
    void c_mlmodel_destruct_model(void* ptr) {
        pecos::MLModel<MLMODEL_MAT_T>* mlm = static_cast<pecos::MLModel<MLMODEL_MAT_T>*>(ptr);
        delete mlm;
    }
    // Allowed attr: nr_labels, nr_codes, nr_features
    uint32_t c_mlmodel_get_int_attr(void* ptr, const char* attr) {
        pecos::MLModel<MLMODEL_MAT_T>* mlm = static_cast<pecos::MLModel<MLMODEL_MAT_T>*>(ptr);
        return mlm->get_int_attr(attr);
    }

    #define C_MLMODEL_PREDICT(SUFFIX, PY_MAT, C_MAT) \
    void c_mlmodel_predict ## SUFFIX( \
        void* ptr, \
        const PY_MAT* input_x, \
        const ScipyCsrF32* csr_codes, \
        const char* overridden_post_processor, \
        const uint32_t overridden_only_topk, \
        const int num_threads, \
        py_sparse_allocator_t pred_alloc) { \
        pecos::MLModel<MLMODEL_MAT_T>* mlm = static_cast<pecos::MLModel<MLMODEL_MAT_T>*>(ptr); \
        C_MAT X(input_x); \
        pecos::csr_t prev_layer_pred; \
        bool no_prev_pred; \
	if (csr_codes) { \
	    prev_layer_pred = pecos::csr_t(csr_codes).deep_copy(); \
	    no_prev_pred = false; \
	} else { \
	    prev_layer_pred.fill_ones(X.rows, mlm->code_count()); \
	    no_prev_pred = true; \
	} \
        pecos::csr_t cur_layer_pred; \
        mlm->predict(X, prev_layer_pred, no_prev_pred, \
            overridden_only_topk, overridden_post_processor, \
            cur_layer_pred, num_threads); \
        cur_layer_pred.create_pycsr(pred_alloc); \
        cur_layer_pred.free_underlying_memory(); \
        prev_layer_pred.free_underlying_memory(); \
    }
    C_MLMODEL_PREDICT(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_MLMODEL_PREDICT(_drm_f32, ScipyDrmF32, pecos::drm_t)

    #define C_MLMODEL_PREDICT_ON_SELECTED_OUTPUTS(SUFFIX, PY_MAT, C_MAT) \
    void c_mlmodel_predict_on_selected_outputs ## SUFFIX( \
        void* ptr, \
        const PY_MAT* input_x, \
        const ScipyCsrF32* selected_outputs_csr, \
        const ScipyCsrF32* csr_codes, \
        const char* overridden_post_processor, \
        const int num_threads, \
        py_sparse_allocator_t pred_alloc) { \
        pecos::MLModel<MLMODEL_MAT_T>* mlm = static_cast<pecos::MLModel<MLMODEL_MAT_T>*>(ptr); \
        C_MAT X(input_x); \
        pecos::csr_t curr_outputs_csr = pecos::csr_t(selected_outputs_csr).deep_copy(); \
        pecos::csr_t prev_layer_pred; \
        bool no_prev_pred; \
	if (csr_codes) { \
	    prev_layer_pred = pecos::csr_t(csr_codes).deep_copy(); \
	    no_prev_pred = false; \
	} else { \
	    prev_layer_pred.fill_ones(X.rows, mlm->code_count()); \
	    no_prev_pred = true; \
	} \
        pecos::csr_t cur_layer_pred; \
        mlm->predict_on_selected_outputs(X, curr_outputs_csr, prev_layer_pred, no_prev_pred, \
            overridden_post_processor, cur_layer_pred, num_threads); \
        cur_layer_pred.create_pycsr(pred_alloc); \
        cur_layer_pred.free_underlying_memory(); \
        curr_outputs_csr.free_underlying_memory(); \
        prev_layer_pred.free_underlying_memory(); \
    }
    C_MLMODEL_PREDICT_ON_SELECTED_OUTPUTS(_csr_f32, ScipyCsrF32, pecos::csr_t)
    C_MLMODEL_PREDICT_ON_SELECTED_OUTPUTS(_drm_f32, ScipyDrmF32, pecos::drm_t)

    // ==== C Interface of XLinearModels ====
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

    void* c_xlinear_load_mmap_model_from_disk(const char* model_path, const bool lazy_load) {
        auto model = new pecos::HierarchicalMLModel(model_path, lazy_load);
        return static_cast<void*>(model);
    }

    void c_xlinear_compile_mmap_model(const char* model_path, const char* mmap_model_path) {
        // Only implemented for bin_search_chunked
        auto model = new pecos::HierarchicalMLModel(model_path, pecos::layer_type_t::LAYER_TYPE_BINARY_SEARCH_CHUNKED);
        model->save_mmap(mmap_model_path);
	delete model;
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
        pecos::csr_t prev_layer_pred; \
        bool no_prev_pred; \
        if (csr_codes) { \
            prev_layer_pred = pecos::csr_t(csr_codes).deep_copy(); \
            no_prev_pred = false; \
        } else { \
            prev_layer_pred.fill_ones(X.rows, C->cols); \
            no_prev_pred = true; \
        } \
        pecos::csc_t C_; \
        C_ = pecos::csc_t(C); \
        pecos::csr_t cur_layer_pred; \
        pecos::csc_t W_ = pecos::csc_t(W); \
        pecos::MLModelMetadata metadata(bias, only_topk, post_processor_str); \
        pecos::MLModel<pecos::csc_t> layer(W_, C_, 0, false, metadata); \
        layer.predict(X, prev_layer_pred, no_prev_pred, only_topk, \
            post_processor_str, cur_layer_pred, num_threads); \
        cur_layer_pred.create_pycsr(pred_alloc); \
        cur_layer_pred.free_underlying_memory(); \
        prev_layer_pred.free_underlying_memory(); \
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
        bool no_prev_pred; \
        if (csr_codes) { \
            prev_layer_pred = pecos::csr_t(csr_codes).deep_copy(); \
            no_prev_pred = false; \
        } else { \
            prev_layer_pred.fill_ones(X.rows, C->cols); \
            no_prev_pred = true; \
        } \
        pecos::csc_t C_; \
        C_ = pecos::csc_t(C); \
        pecos::csr_t cur_layer_pred; \
        pecos::csc_t W_ = pecos::csc_t(W); \
        pecos::MLModelMetadata metadata(bias, 0, post_processor_str); \
        pecos::MLModel<pecos::csc_t> layer(W_, C_, 0, false, metadata); \
        layer.predict_on_selected_outputs(X, curr_outputs_csr, prev_layer_pred, no_prev_pred, \
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


    #define C_SPARSE_INNER_PRODUCTS(SUFFIX, PX_MAT, CX_MAT, PW_MAT, CW_MAT) \
    void c_sparse_inner_products ## SUFFIX( \
        const PX_MAT *pX, \
        const PW_MAT *pW, \
        uint64_t len, \
        uint32_t *X_row_idx, \
        uint32_t *W_col_idx, \
        float32_t *val, \
        int threads) { \
        CX_MAT X(pX); \
        CW_MAT W(pW); \
        compute_sparse_entries_from_rowmajored_X_and_colmajored_M( \
            X, W, len, X_row_idx, W_col_idx, val, threads \
        ); \
    }
    C_SPARSE_INNER_PRODUCTS(_csr2csc_f32, ScipyCsrF32, pecos::csr_t, ScipyCscF32, pecos::csc_t)
    C_SPARSE_INNER_PRODUCTS(_drm2csc_f32, ScipyDrmF32, pecos::drm_t, ScipyCscF32, pecos::csc_t)
    C_SPARSE_INNER_PRODUCTS(_csr2dcm_f32, ScipyCsrF32, pecos::csr_t, ScipyDcmF32, pecos::dcm_t)
    C_SPARSE_INNER_PRODUCTS(_drm2dcm_f32, ScipyDrmF32, pecos::drm_t, ScipyDcmF32, pecos::dcm_t)

    // ==== C Interface of Clustering ====

    #define C_RUN_CLUSTERING(SUFFIX, PY_MAT, C_MAT) \
    void c_run_clustering ## SUFFIX( \
        const PY_MAT* py_mat_ptr, \
        size_t depth, \
        pecos::clustering::ClusteringParam* param_ptr, \
        uint32_t* label_codes) { \
        C_MAT feat_mat(py_mat_ptr); \
        pecos::clustering::Tree tree(depth); \
        tree.run_clustering(feat_mat, param_ptr, label_codes); \
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

    // ==== C Interface of HNSW ====

    typedef pecos::ann::HNSW<float, pecos::ann::FeatVecSparseIPSimd<uint32_t, float>> hnsw_csr_ip_t;
    typedef pecos::ann::HNSW<float, pecos::ann::FeatVecSparseL2Simd<uint32_t, float>> hnsw_csr_l2_t;
    typedef pecos::ann::HNSW<float, pecos::ann::FeatVecDenseIPSimd<float>> hnsw_drm_ip_t;
    typedef pecos::ann::HNSW<float, pecos::ann::FeatVecDenseL2Simd<float>> hnsw_drm_l2_t;

    #define C_ANN_HNSW_TRAIN(SUFFIX, PY_MAT, C_MAT, HNSW_T) \
    void* c_ann_hnsw_train ## SUFFIX( \
        const PY_MAT* pX, \
        uint32_t M, \
        uint32_t efC, \
        int threads, \
        int max_level_upper_bound) { \
        C_MAT feat_mat(pX); \
        HNSW_T *model_ptr = new HNSW_T(); \
        model_ptr->train(feat_mat, M, efC, threads, max_level_upper_bound); \
        return static_cast<void*>(model_ptr); \
    }
    C_ANN_HNSW_TRAIN(_csr_ip_f32, ScipyCsrF32, pecos::csr_t, hnsw_csr_ip_t)
    C_ANN_HNSW_TRAIN(_csr_l2_f32, ScipyCsrF32, pecos::csr_t, hnsw_csr_l2_t)
    C_ANN_HNSW_TRAIN(_drm_ip_f32, ScipyDrmF32, pecos::drm_t, hnsw_drm_ip_t)
    C_ANN_HNSW_TRAIN(_drm_l2_f32, ScipyDrmF32, pecos::drm_t, hnsw_drm_l2_t)

    #define C_ANN_HNSW_LOAD(SUFFIX, HNSW_T) \
    void* c_ann_hnsw_load ## SUFFIX(const char* model_dir, const bool lazy_load) { \
        HNSW_T *model_ptr = new HNSW_T(); \
        model_ptr->load(model_dir, lazy_load); \
        return static_cast<void*>(model_ptr); \
    }
    C_ANN_HNSW_LOAD(_drm_ip_f32, hnsw_drm_ip_t)
    C_ANN_HNSW_LOAD(_drm_l2_f32, hnsw_drm_l2_t)
    C_ANN_HNSW_LOAD(_csr_ip_f32, hnsw_csr_ip_t)
    C_ANN_HNSW_LOAD(_csr_l2_f32, hnsw_csr_l2_t)

    #define C_ANN_HNSW_SAVE(SUFFIX, HNSW_T) \
    void c_ann_hnsw_save ## SUFFIX(void* model_ptr, const char* model_dir) { \
        const auto &model = *static_cast<HNSW_T*>(model_ptr); \
        model.save(model_dir); \
    }
    C_ANN_HNSW_SAVE(_drm_ip_f32, hnsw_drm_ip_t)
    C_ANN_HNSW_SAVE(_drm_l2_f32, hnsw_drm_l2_t)
    C_ANN_HNSW_SAVE(_csr_ip_f32, hnsw_csr_ip_t)
    C_ANN_HNSW_SAVE(_csr_l2_f32, hnsw_csr_l2_t)

    #define C_ANN_HNSW_DESTRUCT(SUFFIX, HNSW_T) \
    void c_ann_hnsw_destruct ## SUFFIX(void* model_ptr) { \
        delete static_cast<HNSW_T*>(model_ptr); \
    }
    C_ANN_HNSW_DESTRUCT(_drm_ip_f32, hnsw_drm_ip_t)
    C_ANN_HNSW_DESTRUCT(_drm_l2_f32, hnsw_drm_l2_t)
    C_ANN_HNSW_DESTRUCT(_csr_ip_f32, hnsw_csr_ip_t)
    C_ANN_HNSW_DESTRUCT(_csr_l2_f32, hnsw_csr_l2_t)

    #define C_ANN_HNSW_SEARCHERS_CREATE(SUFFIX, HNSW_T) \
    void* c_ann_hnsw_searchers_create ## SUFFIX(void* model_ptr, uint32_t num_searcher) { \
	    typedef typename HNSW_T::Searcher searcher_t; \
        const auto &model = *static_cast<HNSW_T*>(model_ptr); \
        auto searchers_ptr = new std::vector<searcher_t>(); \
        for (uint32_t t = 0; t < num_searcher; t++) { \
            searchers_ptr->emplace_back(model.create_searcher()); \
        } \
        return static_cast<void*>(searchers_ptr); \
    }
    C_ANN_HNSW_SEARCHERS_CREATE(_drm_ip_f32, hnsw_drm_ip_t)
    C_ANN_HNSW_SEARCHERS_CREATE(_drm_l2_f32, hnsw_drm_l2_t)
    C_ANN_HNSW_SEARCHERS_CREATE(_csr_ip_f32, hnsw_csr_ip_t)
    C_ANN_HNSW_SEARCHERS_CREATE(_csr_l2_f32, hnsw_csr_l2_t)

    #define C_ANN_HNSW_SEARCHERS_DESTRUCT(SUFFIX, HNSW_T) \
    void c_ann_hnsw_searchers_destruct ## SUFFIX(void* searchers_ptr) { \
        typedef typename HNSW_T::Searcher searcher_t; \
        delete static_cast<std::vector<searcher_t>*>(searchers_ptr); \
    }
    C_ANN_HNSW_SEARCHERS_DESTRUCT(_drm_ip_f32, hnsw_drm_ip_t)
    C_ANN_HNSW_SEARCHERS_DESTRUCT(_drm_l2_f32, hnsw_drm_l2_t)
    C_ANN_HNSW_SEARCHERS_DESTRUCT(_csr_ip_f32, hnsw_csr_ip_t)
    C_ANN_HNSW_SEARCHERS_DESTRUCT(_csr_l2_f32, hnsw_csr_l2_t)

    #define OMP_PARA_FOR _Pragma("omp parallel for schedule(dynamic,1)")
    #define C_ANN_HNSW_PREDICT(SUFFIX, PY_MAT, C_MAT, HNSW_T) \
    void c_ann_hnsw_predict ## SUFFIX( \
        void* model_ptr, \
        const PY_MAT* pX, \
        uint32_t* ret_idx, \
        float* ret_val, \
        uint32_t efS, \
        uint32_t topk, \
        int32_t threads, \
        void* searchers_ptr) { \
        C_MAT feat_mat(pX); \
	    typedef typename HNSW_T::Searcher searcher_t; \
        const auto &model = *static_cast<HNSW_T*>(model_ptr); \
        std::vector<searcher_t> searchers_tmp; \
        if (searchers_ptr == NULL) { \
            threads = (threads <= 0) ? omp_get_num_procs() : threads; \
            for (int t=0; t < threads; t++) { \
                searchers_tmp.emplace_back(model.create_searcher()); \
            } \
        } \
        auto& searchers = (searchers_ptr == NULL) ? searchers_tmp : *static_cast<std::vector<searcher_t>*>(searchers_ptr); \
        threads = searchers.size(); \
        omp_set_num_threads(threads); \
    OMP_PARA_FOR \
        for (uint32_t qid=0; qid < feat_mat.rows; qid++) { \
            int thread_id = omp_get_thread_num(); \
            auto& ret_pairs = searchers[thread_id].predict_single(feat_mat.get_row(qid), efS, topk); \
            for (uint32_t k=0; k < ret_pairs.size(); k++) { \
                uint64_t offset = static_cast<uint64_t>(qid) * topk; \
                ret_val[offset + k] = ret_pairs[k].dist; \
                ret_idx[offset + k] = ret_pairs[k].node_id; \
            } \
        } \
    }
    C_ANN_HNSW_PREDICT(_drm_ip_f32, ScipyDrmF32, pecos::drm_t, hnsw_drm_ip_t)
    C_ANN_HNSW_PREDICT(_drm_l2_f32, ScipyDrmF32, pecos::drm_t, hnsw_drm_l2_t)
    C_ANN_HNSW_PREDICT(_csr_ip_f32, ScipyCsrF32, pecos::csr_t, hnsw_csr_ip_t)
    C_ANN_HNSW_PREDICT(_csr_l2_f32, ScipyCsrF32, pecos::csr_t, hnsw_csr_l2_t)


    // ==== C Interface of PairwiseANN ====

    typedef pecos::ann::PairwiseANN<pecos::ann::FeatVecSparseIPSimd<uint32_t, float>, pecos::csr_t> pairwise_ann_csr_ip_t;
    typedef pecos::ann::PairwiseANN<pecos::ann::FeatVecDenseIPSimd<float>, pecos::drm_t> pairwise_ann_drm_ip_t;

    #define C_PAIRWISE_ANN_TRAIN(SUFFIX, PY_MAT, C_MAT, PAIRWISE_ANN_T) \
    void* c_pairwise_ann_train ## SUFFIX(const PY_MAT* pX, const ScipyCscF32* pY) { \
        C_MAT X_trn(pX); \
        pecos::csc_t Y_csc(pY); \
        PAIRWISE_ANN_T *model_ptr = new PAIRWISE_ANN_T(); \
        model_ptr->train(X_trn, Y_csc); \
        return static_cast<void*>(model_ptr); \
    }
    C_PAIRWISE_ANN_TRAIN(_csr_ip_f32, ScipyCsrF32, pecos::csr_t, pairwise_ann_csr_ip_t)
    C_PAIRWISE_ANN_TRAIN(_drm_ip_f32, ScipyDrmF32, pecos::drm_t, pairwise_ann_drm_ip_t)

    #define C_PAIRWISE_ANN_LOAD(SUFFIX, PAIRWISE_ANN_T) \
    void* c_pairwise_ann_load ## SUFFIX(const char* model_dir, const bool lazy_load) { \
        PAIRWISE_ANN_T *model_ptr = new PAIRWISE_ANN_T(); \
        model_ptr->load(model_dir, lazy_load); \
        return static_cast<void*>(model_ptr); \
    }
    C_PAIRWISE_ANN_LOAD(_drm_ip_f32, pairwise_ann_drm_ip_t)
    C_PAIRWISE_ANN_LOAD(_csr_ip_f32, pairwise_ann_csr_ip_t)

    #define C_PAIRWISE_ANN_SAVE(SUFFIX, PAIRWISE_ANN_T) \
    void c_pairwise_ann_save ## SUFFIX(void* model_ptr, const char* model_dir) { \
        const auto &model = *static_cast<PAIRWISE_ANN_T*>(model_ptr); \
        model.save(model_dir); \
    }
    C_PAIRWISE_ANN_SAVE(_drm_ip_f32, pairwise_ann_drm_ip_t)
    C_PAIRWISE_ANN_SAVE(_csr_ip_f32, pairwise_ann_csr_ip_t)

    #define C_PAIRWISE_ANN_DESTRUCT(SUFFIX, PAIRWISE_ANN_T) \
    void c_pairwise_ann_destruct ## SUFFIX(void* model_ptr) { \
        delete static_cast<PAIRWISE_ANN_T*>(model_ptr); \
    }
    C_PAIRWISE_ANN_DESTRUCT(_drm_ip_f32, pairwise_ann_drm_ip_t)
    C_PAIRWISE_ANN_DESTRUCT(_csr_ip_f32, pairwise_ann_csr_ip_t)

    #define C_PAIRWISE_ANN_SEARCHERS_CREATE(SUFFIX, PAIRWISE_ANN_T) \
    void* c_pairwise_ann_searchers_create ## SUFFIX(void* model_ptr, uint32_t num_searcher) { \
	    typedef typename PAIRWISE_ANN_T::Searcher searcher_t; \
        const auto &model = *static_cast<PAIRWISE_ANN_T*>(model_ptr); \
        auto searchers_ptr = new std::vector<searcher_t>(); \
        for (uint32_t t = 0; t < num_searcher; t++) { \
            searchers_ptr->emplace_back(model.create_searcher()); \
        } \
        return static_cast<void*>(searchers_ptr); \
    }
    C_PAIRWISE_ANN_SEARCHERS_CREATE(_drm_ip_f32, pairwise_ann_drm_ip_t)
    C_PAIRWISE_ANN_SEARCHERS_CREATE(_csr_ip_f32, pairwise_ann_csr_ip_t)

    #define C_PAIRWISE_ANN_SEARCHERS_DESTRUCT(SUFFIX, PAIRWISE_ANN_T) \
    void c_pairwise_ann_searchers_destruct ## SUFFIX(void* searchers_ptr) { \
        typedef typename PAIRWISE_ANN_T::Searcher searcher_t; \
        delete static_cast<std::vector<searcher_t>*>(searchers_ptr); \
    }
    C_PAIRWISE_ANN_SEARCHERS_DESTRUCT(_drm_ip_f32, pairwise_ann_drm_ip_t)
    C_PAIRWISE_ANN_SEARCHERS_DESTRUCT(_csr_ip_f32, pairwise_ann_csr_ip_t)

    #define C_PAIRWISE_ANN_PREDICT(SUFFIX, PY_MAT, C_MAT, PAIRWISE_ANN_T) \
    void c_pairwise_ann_predict ## SUFFIX( \
        void* searchers_ptr, \
        uint32_t batch_size, \
        uint32_t only_topk, \
        const PY_MAT* pQ, \
        uint32_t* label_keys, \
        uint32_t* ret_Imat, \
        uint32_t* ret_Mmat, \
        float* ret_Dmat, \
        float* ret_Vmat, \
        const bool is_same_input) { \
        C_MAT Q_tst(pQ); \
        auto& searchers = *static_cast<std::vector<PAIRWISE_ANN_T::Searcher>*>(searchers_ptr); \
        omp_set_num_threads(searchers.size()); \
    OMP_PARA_FOR \
        for (uint32_t bidx=0; bidx < batch_size; bidx++) { \
            int tid = omp_get_thread_num(); \
            auto input_key = (is_same_input ? 0 : bidx); \
            auto label_key = label_keys[bidx]; \
            auto& ret_pairs = searchers[tid].predict_single(Q_tst.get_row(input_key), label_key, only_topk); \
            for (uint32_t k=0; k < ret_pairs.size(); k++) { \
                uint64_t offset = static_cast<uint64_t>(bidx) * only_topk; \
                ret_Imat[offset + k] = ret_pairs[k].input_key_idx; \
                ret_Dmat[offset + k] = ret_pairs[k].input_key_dist; \
                ret_Vmat[offset + k] = ret_pairs[k].input_label_val; \
                ret_Mmat[offset + k] = 1; \
            } \
        } \
    }
    C_PAIRWISE_ANN_PREDICT(_drm_ip_f32, ScipyDrmF32, pecos::drm_t, pairwise_ann_drm_ip_t)
    C_PAIRWISE_ANN_PREDICT(_csr_ip_f32, ScipyCsrF32, pecos::csr_t, pairwise_ann_csr_ip_t)


    // ==== C Interface of Memory-mappable Hashmap ====

    typedef pecos::mmap_hashmap::Str2IntMap mmap_hashmap_str2int;
    typedef pecos::mmap_hashmap::Int2IntMap mmap_hashmap_int2int;

    // New
    #define MMAP_MAP_NEW(SUFFIX) \
    void* mmap_hashmap_new_ ## SUFFIX () { \
    return static_cast<void*>(new mmap_hashmap_ ## SUFFIX()); }
    MMAP_MAP_NEW(str2int)
    MMAP_MAP_NEW(int2int)

    // Destruct
    #define MMAP_MAP_DESTRUCT(SUFFIX) \
    void mmap_hashmap_destruct_ ## SUFFIX (void* map_ptr) { \
    delete static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr); }
    MMAP_MAP_DESTRUCT(str2int)
    MMAP_MAP_DESTRUCT(int2int)

    // Save
    #define MMAP_MAP_SAVE(SUFFIX) \
    void mmap_hashmap_save_ ## SUFFIX (void* map_ptr, const char* map_dir) { \
    static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->save(map_dir); }
    MMAP_MAP_SAVE(str2int)
    MMAP_MAP_SAVE(int2int)

    // Load
    #define MMAP_MAP_LOAD(SUFFIX) \
    void* mmap_hashmap_load_ ## SUFFIX (const char* map_dir, const bool lazy_load) { \
    mmap_hashmap_ ## SUFFIX * map_ptr = new mmap_hashmap_ ## SUFFIX(); \
    map_ptr->load(map_dir, lazy_load); \
    return static_cast<void *>(map_ptr); }
    MMAP_MAP_LOAD(str2int)
    MMAP_MAP_LOAD(int2int)

    // Size
    #define MMAP_MAP_SIZE(SUFFIX) \
    size_t mmap_hashmap_size_ ## SUFFIX (void* map_ptr) { \
    return static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->size(); }
    MMAP_MAP_SIZE(str2int)
    MMAP_MAP_SIZE(int2int)

    // Insert
    #define KEY_SINGLE_ARG(A,B) A,B
    #define MMAP_MAP_INSERT(SUFFIX, KEY, FUNC_CALL_KEY) \
    void mmap_hashmap_insert_  ## SUFFIX (void* map_ptr, KEY, uint64_t val) { \
        static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->insert(FUNC_CALL_KEY, val); }
    MMAP_MAP_INSERT(str2int, KEY_SINGLE_ARG(const char* key, uint32_t key_len), KEY_SINGLE_ARG(key, key_len))
    MMAP_MAP_INSERT(int2int, uint64_t key, key)

    // Get
    #define MMAP_MAP_GET(SUFFIX, KEY, FUNC_CALL_KEY) \
    uint64_t mmap_hashmap_get_  ## SUFFIX (void* map_ptr, KEY) { \
        return static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->get(FUNC_CALL_KEY); }
    MMAP_MAP_GET(str2int, KEY_SINGLE_ARG(const char* key, uint32_t key_len), KEY_SINGLE_ARG(key, key_len))
    MMAP_MAP_GET(int2int, uint64_t key, key)

    #define MMAP_MAP_GET_W_DEFAULT(SUFFIX, KEY, FUNC_CALL_KEY) \
    uint64_t mmap_hashmap_get_w_default_  ## SUFFIX (void* map_ptr, KEY, uint64_t def_val) { \
        return static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->get_w_default(FUNC_CALL_KEY, def_val); }
    MMAP_MAP_GET_W_DEFAULT(str2int, KEY_SINGLE_ARG(const char* key, uint32_t key_len), KEY_SINGLE_ARG(key, key_len))
    MMAP_MAP_GET_W_DEFAULT(int2int, uint64_t key, key)

    #define MMAP_MAP_BATCH_GET_W_DEFAULT(SUFFIX, KEY, FUNC_CALL_KEY) \
    void mmap_hashmap_batch_get_w_default_  ## SUFFIX (void* map_ptr, const uint32_t n_key, KEY, uint64_t def_val, uint64_t* vals, const int threads) { \
        static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->batch_get_w_default(n_key, FUNC_CALL_KEY, def_val, vals, threads); }
    MMAP_MAP_BATCH_GET_W_DEFAULT(str2int, KEY_SINGLE_ARG(const char* const* keys, const uint32_t* keys_lens), KEY_SINGLE_ARG(keys, keys_lens))
    MMAP_MAP_BATCH_GET_W_DEFAULT(int2int, const uint64_t* key, key)

    // Contains
    #define MMAP_MAP_CONTAINS(SUFFIX, KEY, FUNC_CALL_KEY) \
    bool mmap_hashmap_contains_  ## SUFFIX (void* map_ptr, KEY) { \
        return static_cast<mmap_hashmap_ ## SUFFIX *>(map_ptr)->contains(FUNC_CALL_KEY); }
    MMAP_MAP_CONTAINS(str2int, KEY_SINGLE_ARG(const char* key, uint32_t key_len), KEY_SINGLE_ARG(key, key_len))
    MMAP_MAP_CONTAINS(int2int, uint64_t key, key)


    // ==== C Interface of Memory-mappable Value Store ====

    typedef pecos::mmap_valstore::Float32Store mmap_valstore_float32;
    typedef pecos::mmap_valstore::BytesStore mmap_valstore_bytes;
    typedef pecos::mmap_valstore::row_type row_type;
    typedef pecos::mmap_valstore::col_type col_type;

    // New
    #define MMAP_VALSTORE_NEW(SUFFIX) \
    void* mmap_valstore_new_ ## SUFFIX () { \
    return static_cast<void*>(new mmap_valstore_ ## SUFFIX()); }
    MMAP_VALSTORE_NEW(float32)
    MMAP_VALSTORE_NEW(bytes)

    // Destruct
    #define MMAP_VALSTORE_DESTRUCT(SUFFIX) \
    void mmap_valstore_destruct_ ## SUFFIX (void* map_ptr) { \
    delete static_cast<mmap_valstore_ ## SUFFIX *>(map_ptr); }
    MMAP_VALSTORE_DESTRUCT(float32)
    MMAP_VALSTORE_DESTRUCT(bytes)

    // Number of rows
    #define MMAP_MAP_N_ROW(SUFFIX) \
    row_type mmap_valstore_n_row_ ## SUFFIX (void* map_ptr) { \
    return static_cast<mmap_valstore_ ## SUFFIX *>(map_ptr)->n_row(); }
    MMAP_MAP_N_ROW(float32)
    MMAP_MAP_N_ROW(bytes)

    // Number of columns
    #define MMAP_MAP_N_COL(SUFFIX) \
    col_type mmap_valstore_n_col_ ## SUFFIX (void* map_ptr) { \
    return static_cast<mmap_valstore_ ## SUFFIX *>(map_ptr)->n_col(); }
    MMAP_MAP_N_COL(float32)
    MMAP_MAP_N_COL(bytes)

    // Save
    #define MMAP_VALSTORE_SAVE(SUFFIX) \
    void mmap_valstore_save_ ## SUFFIX (void* map_ptr, const char* map_dir) { \
    static_cast<mmap_valstore_ ## SUFFIX *>(map_ptr)->save(map_dir); }
    MMAP_VALSTORE_SAVE(float32)
    MMAP_VALSTORE_SAVE(bytes)

    // Load
    #define MMAP_VALSTORE_LOAD(SUFFIX) \
    void* mmap_valstore_load_ ## SUFFIX (const char* map_dir, const bool lazy_load) { \
    mmap_valstore_ ## SUFFIX * map_ptr = new mmap_valstore_ ## SUFFIX(); \
    map_ptr->load(map_dir, lazy_load); \
    return static_cast<void *>(map_ptr); }
    MMAP_VALSTORE_LOAD(float32)
    MMAP_VALSTORE_LOAD(bytes)

    // Create view from external values pointer
    void mmap_valstore_from_vals_float32 (
        void* map_ptr,
        const row_type n_row,
        const col_type n_col,
        const mmap_valstore_float32::value_type* vals
    ) {
        static_cast<mmap_valstore_float32 *>(map_ptr)->from_vals(n_row, n_col, vals);
    }
    // Allocate and Init
    void mmap_valstore_from_vals_bytes (
        void* map_ptr,
        const row_type n_row,
        const col_type n_col,
        const char* const* vals,
        const mmap_valstore_bytes::bytes_len_type* vals_lens
    ) {
        static_cast<mmap_valstore_bytes *>(map_ptr)->from_vals(n_row, n_col, vals, vals_lens);
    }

    // Get sub-matrix
    void mmap_valstore_batch_get_float32 (
        void* map_ptr,
        const uint64_t n_sub_row,
        const uint64_t n_sub_col,
        const row_type* sub_rows,
        const col_type* sub_cols,
        mmap_valstore_float32::value_type* ret,
        const int threads
    ) {
        static_cast<mmap_valstore_float32 *>(map_ptr)->batch_get(
            n_sub_row, n_sub_col, sub_rows, sub_cols, ret, threads);
    }
    void mmap_valstore_batch_get_bytes (
        void* map_ptr,
        const uint64_t n_sub_row,
        const uint64_t n_sub_col,
        const row_type* sub_rows,
        const col_type* sub_cols,
        const mmap_valstore_bytes::bytes_len_type trunc_val_len,
        char* ret,
        mmap_valstore_bytes::bytes_len_type* ret_lens,
        const int threads
    ) {
        static_cast<mmap_valstore_bytes *>(map_ptr)->batch_get(
            n_sub_row, n_sub_col, sub_rows, sub_cols, trunc_val_len, ret, ret_lens, threads);
    }

    // ==== C Interface of Score Calibrator ====

    #define C_FIT_PLATT_TRANSFORM(SUFFIX, VAL_TYPE) \
    uint32_t c_fit_platt_transform ## SUFFIX( \
        size_t num_samples, \
	const VAL_TYPE* logits, \
	const VAL_TYPE* tgt_probs, \
	double* AB, \
	size_t max_iter, \
	double eps \
    ) { \
        return pecos::fit_platt_transform(num_samples, logits, tgt_probs, AB[0], AB[1], max_iter, eps); \
    }
    C_FIT_PLATT_TRANSFORM(_f32, float32_t)
    C_FIT_PLATT_TRANSFORM(_f64, float64_t)
}
