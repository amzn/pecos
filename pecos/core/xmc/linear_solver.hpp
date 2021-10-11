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

#ifndef __LINEAR_SOLVER_H__
#define  __LINEAR_SOLVER_H__

#include <algorithm>
#include <vector>
#include "utils/matrix.hpp"
#include "utils/parallel.hpp"
#include "utils/random.hpp"
#include "utils/newton.hpp"

namespace pecos {

namespace linear_solver {

// For L2R_L2LOSS_SVC_DUAL and L2R_L1LOSS_SVC_DUAL, see Algorithm 3 of Hsieh et al., ICML 2008.
// For L2R_LR_DUAL, see Algorithm 5 of Yu et al., MLJ 2010.
//
// A Dual Coordinate Descent Method For Large-Scale Linear SVM (ICML 2008)
//     C.-J. Hsieh K.-W. Chang, C.-J. Lin, S. S. Keerthi, and S. Sundararajan
//     https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf
//
// Dual coordinate descent methods for logistic regression and maximum entropy models (MLJ 2010)
//     H.-F. Yu, F.-L. Huang, and C.-J. Lin
//     https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
enum SolverType {
    L2R_L2LOSS_SVC_DUAL=1,
    L2R_L1LOSS_SVC_DUAL=3,
    L2R_LR_DUAL=7,
    L2R_L2LOSS_SVC_PRIMAL = 2,
}; /* solver_type */


// ===== SVM Solvers =====
struct SVMParameter {
    SVMParameter(
        int solver_type=L2R_L1LOSS_SVC_DUAL,
        double Cp=1.0,
        double Cn=1.0,
        int max_iter=1000,
        double eps=0.1,
        double bias=1.0
    ): solver_type(solver_type), max_iter(max_iter), Cp(Cp), Cn(Cn), eps(eps), bias(bias) {}

    int solver_type;
    size_t max_iter;
    double Cp, Cn, eps, bias;
};

template <typename MAT, typename value_type, typename WORKER>
struct l2r_erm_fun : public objective_function<MAT, value_type, WORKER> {
   public:
    typedef dense_vec_t<value_type> dvec_wrapper_t;
    typedef std::vector<value_type> dvec_t;
    std::vector<int> I;
    int sizeI;
    const SVMParameter *param;
    const MAT &X;
    WORKER *worker;

   protected:
    double wTw;
    dvec_t tmp_wx, tmp_tmp;
    dvec_wrapper_t wx;
    dvec_wrapper_t tmp;

   public:
    l2r_erm_fun(const SVMParameter *param, const MAT &X, WORKER *worker) : param(param), X(X), worker(worker) {
        I.resize(worker->y_size, 0);
        tmp_wx.resize(worker->y_size, 0);
        tmp_tmp.resize(worker->y_size, 0);
        wx = dvec_wrapper_t(tmp_wx);
        tmp = dvec_wrapper_t(tmp_tmp);
    }
    ~l2r_erm_fun() {}

    uint64_t get_w_size() {
        return worker->w_size;
    }
    uint64_t get_y_size() {
        return worker->y_size;
    }
    double get_bias() {
        return param->bias;
    }

    double get_eps() {
        return param->eps;
    }

    double fun(dvec_wrapper_t w, value_type &b) {
        double f = 0;
        wTw = 0;
        Xv(w, wx, b);

        wTw = do_dot_product(w, w);
        if (param->bias > 0) {
            wTw += (double) b * b;
        }

        for (auto &i : worker->index) {
            f += this->C_times_loss(i, wx[i]);
        }

        f = f + 0.5 * wTw;
        return f;
    }

    double linesearch_and_update(double *f, double alpha, dvec_wrapper_t w, dvec_wrapper_t s, dvec_wrapper_t g, value_type &b, value_type &bs, value_type &bg) {
        double eta = 0.01;
        uint64_t n = get_w_size();
        int max_num_linesearch = 20;
        double fold = *f;

        dvec_t tmp_new_w;
        tmp_new_w.resize(n, 0);
        dvec_wrapper_t new_w(tmp_new_w);
        value_type new_b;

        double gTs = do_dot_product(s, g);
        if (param->bias > 0) {
            gTs += (double) bs * bg;
        }

        int num_linesearch = 0;
        for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
            for (uint64_t i = 0; i < n; i++) {
                new_w[i] = w[i] + alpha * s[i];
            }
            new_b = b + alpha * bs;
            *f = fun(new_w, new_b);
            if (*f - fold <= eta * alpha * gTs) {
                break;
            } else {
                alpha *= 0.5;
            }
        }
        if (num_linesearch >= max_num_linesearch) {
            *f = fold;
            return 0;
        } else {
            for (uint64_t i = 0; i < n; i++) {
                w[i] = new_w[i];
            }
            if (param->bias > 0) {
                b = new_b;
            }
        }
        return alpha;
    }

    void Xv(dvec_wrapper_t w, dvec_wrapper_t Xv, value_type &b) {
        const MAT &X = this->X;
        for (auto &i : this->worker->index) {
            const auto &xi = X.get_row(i);
            Xv[i] = do_dot_product(w, xi);
            if (param->bias > 0) {
                Xv[i] += param->bias * b;
            }
        }
    }
};

template <typename MAT, typename value_type, typename WORKER>
struct l2r_l2_svc_fun : public l2r_erm_fun<MAT, value_type, WORKER> {
   public:
    typedef dense_vec_t<value_type> dvec_wrapper_t;
    typedef std::vector<value_type> dvec_t;
    l2r_l2_svc_fun(const SVMParameter *param, const MAT &X, WORKER *worker) : l2r_erm_fun<MAT, value_type, WORKER>(param, X, worker) {}
    ~l2r_l2_svc_fun() {}

    value_type get_c(uint64_t i) {
        if (this->worker->inst_info[i].y > 0) {
            return this->worker->inst_info[i].cost * this->param->Cp;
        } else {
            return this->worker->inst_info[i].cost * this->param->Cn;
        }
    }

    void grad(dvec_wrapper_t w, dvec_wrapper_t G, value_type &b, value_type &bg) {
        this->sizeI = 0;
        for (auto &i : this->worker->index) {
            this->tmp[i] = this->wx[i] * this->worker->inst_info[i].y;
            if (this->tmp[i] < 1) {
                this->tmp[this->sizeI] = get_c(i) * this->worker->inst_info[i].y * (this->tmp[i] - 1);
                this->I[this->sizeI] = i;
                this->sizeI++;
            }
        }
        subXTv(this->tmp, G, bg);
        do_xp2y(w, G);
        if (this->param->bias > 0) {
            bg = b + 2 * bg;
        }
    }

    void get_diag_preconditioner(dvec_wrapper_t M, value_type &bM) {
        uint64_t w_size = this->worker->w_size;
        const MAT &X = this->X;
        for (uint64_t i = 0; i < w_size; i++) {
            M[i] = 1;
        }
        bM = 1;

        for (int i = 0; i < this->sizeI; i++) {
            const auto &xi = X.get_row(this->I[i]);
            do_ax2py(2*get_c(this->I[i]), xi, M);
            if (this->param->bias > 0) {
                bM += this->param->bias * this->param->bias * 2 * get_c(this->I[i]);
            }
        }
    }

    void Hv(dvec_wrapper_t s, dvec_wrapper_t Hs, value_type &bs, value_type &bHs) {
        uint64_t w_size = this->worker->w_size;
        const MAT &X = this->X;
        for (uint64_t i = 0; i < w_size; i++) {
            Hs[i] = 0;
        }
        bHs = 0;
        double xTs = 0;
        for (int i = 0; i < this->sizeI; i++) {
            const auto &xi = X.get_row(this->I[i]);
            xTs = do_dot_product(s, xi);
            if (this->param->bias > 0) {
                xTs += this->param->bias * bs;
            }
            xTs = get_c(this->I[i]) * xTs;
            do_axpy(xTs, xi, Hs);
            if (this->param->bias > 0) {
                bHs += xTs * this->param->bias;
            }
        }
        do_xp2y(s, Hs);
        bHs = bs + 2 * bHs;
    }

    double linesearch_and_update(double *f, double alpha, dvec_wrapper_t w, dvec_wrapper_t s, dvec_wrapper_t g, value_type &b, value_type &bs, value_type &bg) {
        double eta = 0.01;
        int max_num_linesearch = 20;
        double fold = *f;
        this->Xv(s, this->tmp, bs);

        double sTs = do_dot_product(s, s);
        if (this->param->bias > 0) {
            sTs += (double) bs * bs;
        }
        double wTs = do_dot_product(s, w);
        if (this->param->bias > 0) {
            wTs += (double) bs * b;
        }
        double gTs = do_dot_product(s, g);
        if (this->param->bias > 0) {
            gTs += (double) bs * bg;
        }

        int num_linesearch = 0;
        for (num_linesearch = 0; num_linesearch < max_num_linesearch; num_linesearch++) {
            double loss = 0;
            for (auto &i : this->worker->index) {
                double inner_product = this->tmp[i] * alpha + this->wx[i];
                loss += this->C_times_loss(i, inner_product);
            }
            *f = loss + (alpha * alpha * sTs + this->wTw) / 2.0 + alpha * wTs;
            if (*f - fold <= eta * alpha * gTs) {
                for (auto &i : this->worker->index) {
                    this->wx[i] += alpha * this->tmp[i];
                }
                break;
            } else {
                alpha *= 0.5;
            }
        }
        if (num_linesearch >= max_num_linesearch) {
            *f = fold;
            return 0;
        } else {
            do_axpy(alpha, s, w);
            b += alpha * bs;
        }
        this->wTw += alpha * alpha * sTs + 2 * alpha * wTs;
        return alpha;
    }

   protected:
    void subXTv(dvec_wrapper_t v, dvec_wrapper_t G, value_type &bg) {
        uint64_t w_size = this->worker->w_size;
        const MAT &X = this->X;
        for (size_t i = 0; i < w_size; i++) {
            G[i] = 0;
        }
        bg = 0;
        for (int i = 0; i < this->sizeI; i++) {
            const auto &xi = X.get_row(this->I[i]);
            do_axpy(v[i], xi, G);
            if (this->param->bias > 0) {
                bg += this->param->bias * v[i];
            }
        }
    }

   private:
    double C_times_loss(int i, double wx_i) {
        double d = 1 - this->worker->inst_info[i].y * wx_i;
        if (d > 0) {
            return get_c(i) * d * d;
        } else {
            return 0;
        }
    }
};

#define INF HUGE_VAL
template<class value_type>
struct SVMWorker {

    typedef std::vector<value_type> dvec_t;
    typedef dense_vec_t<value_type> dvec_wrapper_t;
    typedef random_number_generator<> rng_t;

    struct InstInfo {
        value_type y;
        value_type cost;
        InstInfo(value_type y=0, value_type cost=0):
            y(y), cost(cost) {}

        void clear() {
            y = 0;
            cost = 0;
        }
    };

    SVMParameter param;
    u64_dvec_t index; // used to determine the subset of rows of X are used in the training.
    u64_dvec_t feat_index; // used to get the ranking of features in terms of weights
    std::vector<InstInfo> inst_info;
    dvec_t w;
    value_type b; // bias parameter
    dvec_t QD;
    dvec_t alpha;
    uint64_t w_size, y_size;

    SVMWorker(): w_size(0), y_size(0) {}

    void init(uint64_t w_size, uint64_t y_size, const SVMParameter *param_ptr=NULL) {
        if(param_ptr != NULL) {
            param = *param_ptr;
        }
        this->w_size = w_size;
        this->y_size = y_size;
        w.resize(w_size, 0);
        inst_info.resize(y_size);
        b = 0;
        this->feat_index.reserve(this->w_size);
        this->index.reserve(this->y_size);


        if(param.solver_type == L2R_L2LOSS_SVC_DUAL) {
            alpha.resize(y_size, 0);
            QD.resize(y_size, 0);
        } else if(param.solver_type == L2R_L1LOSS_SVC_DUAL) {
            alpha.resize(y_size, 0);
            QD.resize(y_size, 0);
        } else if(param.solver_type == L2R_LR_DUAL) {
            alpha.resize(2 * y_size, 0); // store both alpha and upper_bound - alpha
            QD.resize(y_size, 0);
        }
    }

    void lazy_init(size_t w_size, size_t y_size, const SVMParameter *param_ptr=NULL) {
        if((w_size != this->w_size)
                || (y_size != this->y_size)
                || ((param_ptr != NULL) && (param_ptr->solver_type != param.solver_type))) {
            init(w_size, y_size, param_ptr);
        } else {
            param = *param_ptr;
        }
    }

    template<typename MAT>
    void solve(const MAT& X, int seed=0) {
        // the solution will be available in w and b
        if(param.solver_type == L2R_L1LOSS_SVC_DUAL) {
            solve_l2r_l1l2_svc(X, seed);
        } else if(param.solver_type == L2R_L2LOSS_SVC_DUAL) {
            solve_l2r_l1l2_svc(X, seed);
        } else if(param.solver_type == L2R_LR_DUAL) {
            solve_l2r_lr(X, seed);
        } else if (param.solver_type == L2R_L2LOSS_SVC_PRIMAL) {
            solve_l2r_l2_svc_primal(X, seed);
        }
    }

    template <typename MAT>
    void solve_l2r_l2_svc_primal(const MAT &X, int seed) {
        l2r_l2_svc_fun<MAT, value_type, SVMWorker> fun_obj(&param, X, this);
        NEWTON<MAT, value_type, SVMWorker> newton_obj(&fun_obj);
        dvec_wrapper_t curr_w(w);
        // re-initialize w and b
        for(size_t j = 0; j < w_size; j++) {
            curr_w[j] = 0;
        }
        b = 0;
        newton_obj.newton(curr_w, b);
    }

    template<typename MAT>
    void solve_l2r_l1l2_svc(const MAT& X, int seed) {
        dvec_wrapper_t curr_w(w);
        rng_t rng(seed);

        for(size_t j = 0; j < w_size; j++) {
            curr_w[j] = 0;
        }
        b = 0;

        auto get_diag = [&](size_t i) {
            auto class_cost = (inst_info[i].y > 0) ? param.Cp : param.Cn;
            return (param.solver_type == L2R_L2LOSS_SVC_DUAL)? (0.5 / (class_cost * inst_info[i].cost)) : 0.0;
        };
        auto get_upper_bound = [&](size_t i) {
            auto class_cost = (inst_info[i].y > 0) ? param.Cp : param.Cn;
            return (param.solver_type == L2R_L2LOSS_SVC_DUAL)? INF : class_cost * inst_info[i].cost;
        };

        for(auto& i : index) {
            alpha[i] = 0;
            QD[i] = get_diag(i);

            const auto& xi = X.get_row(i);
            QD[i] += do_dot_product(xi, xi) + (param.bias > 0 ? param.bias * param.bias : 0);
            double coef = (double) inst_info[i].y * alpha[i];
            do_axpy(coef, xi, curr_w);
            b += (param.bias > 0 ? coef * param.bias : 0);
        }

        // PG: projected gradient, for shrinking and stopping
        double PGmax_old = INF;
        double PGmin_old = -INF;
        double PGmax_new, PGmin_new;

        size_t active_size = index.size();
        size_t iter = 0;
        while(iter < param.max_iter) {
            PGmax_new = -INF;
            PGmin_new = INF;

            // shuffle
            rng.shuffle(index.begin(), index.begin() + active_size);

            size_t s = 0;
            for(s = 0; s < active_size; s++) {
                size_t i = index[s];
                const signed char yi = inst_info[i].y;
                const auto& xi = X.get_row(i);

                float64_t G = yi * (do_dot_product(curr_w, xi) + (param.bias > 0 ? b * param.bias : 0.0)) - 1;
                float64_t C = get_upper_bound(i);
                G += alpha[i] * get_diag(i);

                double PG = 0;
                if(alpha[i] == 0) {
                    if(G > PGmax_old) {
                        active_size--;
                        std::swap(index[s], index[active_size]);
                        s--;
                        continue;
                    } else if (G < 0) {
                        PG = G;
                    }
                } else if (alpha[i] == C) {
                    if (G < PGmin_old) {
                        active_size--;
                        std::swap(index[s], index[active_size]);
                        s--;
                        continue;
                    } else if (G > 0) {
                        PG = G;
                    }
                } else {
                    PG = G;
                }

                PGmax_new = std::max(PGmax_new, PG);
                PGmin_new = std::min(PGmin_new, PG);

                if(fabs(PG) > 1.0e-12) {
                    float64_t alpha_old = alpha[i];
                    alpha[i] = static_cast<float64_t>(std::min(std::max(alpha[i] - G / QD[i], 0.0), C));
                    float64_t d = (alpha[i] - alpha_old) * yi;
                    do_axpy(d, xi, curr_w);
                    b += (param.bias > 0 ? d * param.bias : 0);
                }
            }

            iter++;
            if(PGmax_new - PGmin_new <= param.eps) {
                if(active_size == index.size()) {
                    break;
                } else {
                    active_size = index.size();
                    PGmax_old = INF;
                    PGmin_old = -INF;
                    continue;
                }
            }
            PGmax_old = PGmax_new;
            PGmin_old = PGmin_new;
            if (PGmax_old <= 0) {
                PGmax_old = INF;
            }
            if (PGmin_old >= 0) {
                PGmin_old = -INF;
            }
        }
    }

    template<typename MAT>
    void solve_l2r_lr(const MAT& X, int seed) {
        dvec_wrapper_t curr_w(w);
        rng_t rng(seed);

        for(size_t j = 0; j < w_size; j++) {
            curr_w[j] = 0;
        }
        b = 0;

        auto get_upper_bound = [&](size_t i) {
            auto class_cost = (inst_info[i].y > 0) ? param.Cp : param.Cn;
            return class_cost * inst_info[i].cost;
        };

        dvec_t& xTx = QD;
        size_t max_inner_iter = 100; // for inner Newton
        double innereps = 1e-2;
        double innereps_min = std::min(1e-8, param.eps);

        // Initial alpha can be set here. Note that
        // 0 < alpha[2 * i] < upper_bound[GETI(i)]
        // alpha[2 * i] + alpha[2 * i + 1] = upper_bound[GETI(i)]
        for(auto& i : index) {
            alpha[2 * i] = std::min(0.001 * get_upper_bound(i), 1e-8);
            alpha[2 * i + 1] = get_upper_bound(i) - alpha[2 * i];

            const auto& xi = X.get_row(i);
            xTx[i] = do_dot_product(xi, xi) + (param.bias > 0 ? param.bias * param.bias : 0);
            double coef = (double) inst_info[i].y * alpha[2 * i];
            do_axpy(coef, xi, curr_w);
            b += (param.bias > 0 ? coef * param.bias : 0);
        }

        size_t iter = 0;
        while(iter < param.max_iter) {
            // shuffle
            rng.shuffle(index.begin(), index.end());

            size_t newton_iter = 0;
            float64_t Gmax = 0;
            for(auto& i : index) {
                const signed char yi = inst_info[i].y;
                const auto& xi = X.get_row(i);

                float64_t C = get_upper_bound(i);
                float64_t xisq = xTx[i];
                float64_t ywTx = yi * (do_dot_product(curr_w, xi) + (param.bias > 0 ? b * param.bias : 0.0));
                float64_t a = xisq, b = ywTx;

                // Decide to minimize g_1(z) or g_2(z)
                int ind1 = 2 * i, ind2 = 2 * i + 1, sign = 1;
                if(0.5 * a * (alpha[ind2] - alpha[ind1]) + b < 0) {
                    ind1 = 2 * i + 1;
                    ind2 = 2 * i;
                    sign = -1;
                }

                //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
                float64_t alpha_old = alpha[ind1];
                float64_t z = alpha_old;
                if(C - z < 0.5 * C) {
                    z = 0.1 * z;
                }
                float64_t gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
                Gmax = std::max(Gmax, fabs(gp));

                // Newton method on the sub-problem
                const float64_t eta = 0.1; // xi in the paper
                size_t inner_iter = 0;
                while(inner_iter <= max_inner_iter) {
                    if(fabs(gp) < innereps) {
                        break;
                    }
                    float64_t gpp = a + C / (C - z) / z;
                    float64_t tmpz = z - gp / gpp;
                    if(tmpz <= 0) {
                        z *= eta;
                    } else { // tmpz in (0, C)
                        z = tmpz;
                    }
                    gp = a * (z - alpha_old) + sign * b + log(z / (C - z));
                    newton_iter++;
                    inner_iter++;
                }
                if(inner_iter > 0) { // update curr_w
                    alpha[ind1] = z;
                    alpha[ind2] = C - z;
                    float64_t coef = sign * (z - alpha_old) * yi;
                    do_axpy(coef, xi, curr_w);
                    b += (param.bias > 0 ? coef * param.bias : 0);
                }
            }

            iter++;
            if(Gmax < param.eps) {
                break;
            }

            if(newton_iter <= index.size() / 10) {
                innereps = std::max(innereps_min, 0.1 * innereps);
            }
        }
    }
};

template<class MAT, class value_type=float64_t>
struct SVMJob {
    typedef SVMWorker<value_type> svm_worker_t;
    const MAT* feat_mat; // n \times d
    const csc_t* Y;      // n \times L
    const csc_t* C;      // L \times k: NULL to denote the pure Multi-label setting
    const csc_t* M;      // n \times k: NULL to denote the pure Multi-label setting
    const csc_t* R;      // n \times L: NULL to denote NOT-USING the relevance value for cost-sensitive learning
    size_t code;         // code idx in C (i.e., column index of C)
    size_t subcode;      // index of the label with code (i.e. column index of Y or row index of C)
    const SVMParameter *param_ptr;

    SVMJob(
        const MAT* feat_mat,
        const csc_t* Y,
        const csc_t* C,
        const csc_t* M,
        const csc_t* R,
        size_t code,
        size_t subcode,
        const SVMParameter *param_ptr=NULL
    ):
        feat_mat(feat_mat),
        Y(Y),
        C(C),
        M(M),
        R(R),
        code(code),
        subcode(subcode),
        param_ptr(param_ptr) { }

    void init_worker(svm_worker_t& worker) const {
        size_t w_size = feat_mat->cols;
        size_t y_size = feat_mat->rows;
        worker.lazy_init(w_size, y_size, param_ptr);

        for(auto &i : worker.index) {
            worker.inst_info[i].clear();
        }
        worker.index.clear();
        if(M != NULL) {
            // multi-label setting with codes for labels
            const auto& m_c = M->get_col(code);
            for(size_t idx = 0; idx < m_c.nnz; idx++) {
                size_t i = m_c.idx[idx];
                worker.index.push_back(i);
                worker.inst_info[i].y = -1;
                worker.inst_info[i].cost = 1.0;
            }
        } else {
            // pure multi-label setting without additional codes
            for(size_t i = 0; i < y_size; i++) {
                worker.index.push_back(i);
                worker.inst_info[i].y = -1;
                worker.inst_info[i].cost = 1.0;
            }
        }
        const auto& y_s = Y->get_col(subcode);
        for(size_t idx = 0; idx < y_s.nnz; idx++) {
            size_t i = y_s.idx[idx];
            worker.inst_info[i].y = +1;
            if(worker.inst_info[i].cost == 0) {
                // added positive instances which are not included by M.
                worker.index.push_back(i);
            }
            if(R != NULL) {
                // Cost-sensitive Learning with provided relevance matrix.
                // Assume Y and R has the same indices and indptr patterns,
                // which is verified in the (pecos.xmc.base) MLProblem constructor.
                const auto& r_s = R->get_col(subcode);
                worker.inst_info[i].cost = r_s.val[idx];
            }
            else {
                worker.inst_info[i].cost = 1.0;
            }
        }
    }

    /*
     * Solve the SVM Problem by the *worker*
     * Store *max_nonzeros* parameters with the absolute value >= *threshold* into  *coo_model*
     * */
    void solve(svm_worker_t& worker, coo_t& coo_model, double threshold=0.0, uint32_t max_nonzeros=0) const {
        worker.solve(*feat_mat);
        if(max_nonzeros == 0) {
            for(size_t i = 0; i < worker.w_size; i++) {
                coo_model.push_back(i, subcode, worker.w[i], threshold);
            }
            if(param_ptr->bias > 0) {
                coo_model.push_back(worker.w_size, subcode, worker.b, threshold);
            }
        } else { // max_nonzeros >= 1
            auto feat_index = worker.feat_index;
            feat_index.clear();
            for(size_t i = 0; i < worker.w_size; i++) {
                if(fabs(worker.w[i]) >= threshold) {
                    feat_index.push_back(i);
                }
            }

            if(feat_index.size() >= max_nonzeros) { // feat_index.size() >= 1
                struct comparator_by_absolute_value_t {
                    const float32_t *pred_val;
                    bool increasing;
                    comparator_by_absolute_value_t(const float32_t *val, bool increasing=true):
                        pred_val(val), increasing(increasing) {}
                    bool operator() (const size_t i, const size_t j) const {
                        if(increasing) {
                            return (fabs(pred_val[i]) < fabs(pred_val[j])) \
                                || (fabs(pred_val[i]) == fabs(pred_val[j]) && i < j);
                        } else {
                            return (fabs(pred_val[i]) > fabs(pred_val[j])) \
                                || (fabs(pred_val[i]) == fabs(pred_val[j]) && i < j);
                        }
                    }
                };

                // Keep max_nonzeros feature indices with largest absolute weight values
                // We are using nth_element to have an O(w_size) implementation, as a result
                // the index with the least absolute value from the remaining indices is put in the end.
                comparator_by_absolute_value_t comp(worker.w.data(), false);
                const auto first = feat_index.begin();
                const auto last = feat_index.end();
                const size_t actual_nonzeros = std::min<size_t>(max_nonzeros, feat_index.size());
                std::nth_element(first, first + actual_nonzeros - 1, last, comp);
                feat_index.resize(actual_nonzeros);
            }

            if(param_ptr->bias > 0) {
                if(max_nonzeros > feat_index.size()) {
                    coo_model.push_back(worker.w_size, subcode, worker.b, threshold);
                } else { // i.e., max_nonzeros == feat_index.size()
                    if(fabs(worker.b) > fabs(worker.w[feat_index.back()])) {
                        // we should consider to include bias term instead of the feature index with least absolute value
                        coo_model.push_back(worker.w_size, subcode, worker.b, threshold);
                        feat_index.pop_back();
                    }
                }
            }
            for(auto& i : feat_index) {
                coo_model.push_back(i, subcode, worker.w[i], threshold);
            }
        }
    }

    void reset_worker(svm_worker_t& worker) const {
        for(auto &i : worker.index) {
            worker.inst_info[i].clear();
        }
        worker.index.clear();
    }
};

// Training single-layer of multi-label problem with clustering codes
// Y: shape of N \times L, the instance-to-label matrix with binary classification signals
// C: shape of L \times K, the label-to-cluster matrix for selecting inst/labels within same cluster
// M: shape of N \times K, the instance-to-cluster matrix for negative sampling
// R: shape of N \times L, the relevance matrix for cost-sensitive learning
// Note that we assume Y and R has the same nonzero patterns (same indices and indptr),
// which is verified in the (pecos.xmc.base) MLProblem constructor.
// See more details in Eq. (10) of PECOS arxiv paper (Yu et. al., 2020)
template<class MAT>
void multilabel_train_with_codes(
    const MAT* feat_mat,
    const csc_t *Y,
    const csc_t *C,
    const csc_t *M,
    const csc_t *R,
    coo_t *model,
    double threshold,
    uint32_t max_nonzeros_per_label,
    SVMParameter *param,
    int threads
) {
    typedef typename MAT::value_type value_type;
    typedef SVMJob<MAT, value_type> svm_job_t;
    typedef typename svm_job_t::svm_worker_t svm_worker_t;

    size_t w_size = feat_mat->cols;
    size_t y_size = feat_mat->rows;
    size_t nr_labels = Y->cols;

    threads = set_threads(threads);
    std::vector<svm_worker_t> worker_set(threads);
    std::vector<coo_t> model_set(threads);

#pragma omp parallel for schedule(static, 1)
    for(int tid = 0; tid < threads; tid++) {
        worker_set[tid].init(w_size, y_size, param);
        model_set[tid].reshape(w_size + (param->bias > 0), nr_labels);
    }

    std::vector<svm_job_t> job_queue;
    if(C != NULL && M != NULL) {
        size_t code_size = C->cols;
        for(size_t code = 0; code < code_size; code++) {
            const auto& C_code = C->get_col(code);
            for(size_t idx = 0; idx < C_code.nnz; idx++) {
                size_t subcode = static_cast<size_t>(C_code.idx[idx]);
                job_queue.push_back(svm_job_t(feat_mat, Y, C, M, R, code, subcode, param));
            }
        }
    } else {
        // either C == NULL or M == NULL
        // pure multi-label setting
        for(size_t subcode = 0; subcode < nr_labels; subcode++) {
            job_queue.push_back(svm_job_t(feat_mat, Y, NULL, NULL, R, 0, subcode, param));
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for(size_t job_id = 0; job_id < job_queue.size(); job_id++) {
        int tid = omp_get_thread_num();
        auto& worker = worker_set[tid];
        auto& local_model = model_set[tid];
        const auto& job = job_queue[job_id];
        job.init_worker(worker);
        job.solve(worker, local_model, threshold, max_nonzeros_per_label);
        job.reset_worker(worker);
    }
    model->reshape(w_size + (param->bias > 0), nr_labels);
    model->swap(model_set[0]);
    for(int tid = 1; tid < threads; tid++) {
        model->extends(model_set[tid]);
    }
}

} // end of namespace linear_solver
} // end of namespace pecos

#endif // end of __LINEAR_SOLVER_H__
