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

// For L2R_L2LOSS_SVC_PRIMAL, see reference:
// Chang, K.-W., Hsieh, C.-J., & Lin, C.-J. (2008). Coordinate descent method for large-scale L2-loss linear
// SVM. Journal of Machine Learning Research, 9,
// 1369–1398.
// Implementation:
// https://github.com/cjlin1/liblinear/blob/master/newton.cpp
// https://github.com/cjlin1/liblinear/blob/master/linear.cpp

#ifndef _NEWTON_H
#define _NEWTON_H

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "parallel.hpp"
#include "random.hpp"
#ifndef min
template <class T>
static inline T min(T x, T y) { return (x < y) ? x : y; }
#endif
#ifndef max
template <class T>
static inline T max(T x, T y) {return (x > y) ? x : y; }
#endif
namespace pecos {
    // For L2R_L2LOSS_SVC_PRIMAL, see reference:
    //      Chang, K.-W., Hsieh, C.-J., & Lin, C.-J. (2008). Coordinate descent method for large-scale L2-loss linear
    //      SVM. Journal of Machine Learning Research, 9,
    //      1369–1398.
    //      Implementation:
    //          https://github.com/cjlin1/liblinear/blob/master/newton.cpp
    //          https://github.com/cjlin1/liblinear/blob/master/linear.cpp

    template <typename MAT, typename value_type, typename WORKER>
    struct objective_function {
    public:
        typedef dense_vec_t<value_type> dvec_wrapper_t;
        typedef std::vector<value_type> dvec_t;
        virtual double fun(dvec_wrapper_t w, value_type &b) = 0;
        virtual void grad(dvec_wrapper_t w, dvec_wrapper_t G, value_type &b, value_type &bg) = 0;
        virtual void Hv(dvec_wrapper_t s, dvec_wrapper_t Hs, value_type &bs, value_type &bHs) = 0;
        virtual void get_diag_preconditioner(dvec_wrapper_t M, value_type &bM) = 0;
        virtual uint64_t get_w_size() = 0;
        virtual uint64_t get_y_size() = 0;
        virtual double get_bias() = 0;
        virtual double get_eps() = 0;

        virtual double C_times_loss(int i, double wx_i) = 0;
        virtual ~objective_function(void) {}

        virtual double linesearch_and_update(double *f, double alpha, dvec_wrapper_t w, dvec_wrapper_t s, dvec_wrapper_t g, value_type &b, value_type &bs, value_type &bg) = 0;
    };

    template <typename MAT, typename value_type, typename WORKER>
    struct NEWTON {
        typedef std::vector<value_type> dvec_t;
        typedef dense_vec_t<value_type> dvec_wrapper_t;
        double eps;
        double eps_cg;
        int max_iter;
        objective_function<MAT, value_type, WORKER> *fun_obj;
        dvec_t tmp_w0, tmp_g, tmp_M, tmp_s, tmp_r;

        NEWTON(const objective_function<MAT, value_type, WORKER> *fun_obj, double eps_cg = 0.5, int max_iter = 1000) {
            this->fun_obj = const_cast<objective_function<MAT, value_type, WORKER> *>(fun_obj);
            this->eps = this->fun_obj->get_eps();
            this->eps_cg = eps_cg;
            this->max_iter = max_iter;

            uint64_t w_size = this->fun_obj->get_w_size();
            tmp_w0.resize(w_size, 0);
            tmp_g.resize(w_size, 0);
            tmp_M.resize(w_size, 0);
            tmp_s.resize(w_size, 0);
            tmp_r.resize(w_size, 0);
        };
        ~NEWTON(){};

        void newton(dvec_wrapper_t w, value_type &b) {
            uint64_t w_size = fun_obj->get_w_size();
            double step_size;
            double f, fold, actred;
            double init_step_size = 1;
            int search = 1, iter = 1;
            const double alpha_pcg = 0.01;
            dvec_wrapper_t w0(this->tmp_w0);
            dvec_wrapper_t g(this->tmp_g);
            dvec_wrapper_t M(this->tmp_M);
            dvec_wrapper_t s(this->tmp_s);
            dvec_wrapper_t r(this->tmp_r);
            value_type b0 = 0;
            value_type bg = 0;
            value_type bM = 0;
            value_type bs = 0;
            value_type br = 0;
            // calculate gradient norm at w=0 for stopping condition.
            f = fun_obj->fun(w0, b0);
            fun_obj->grad(w0, g, b0, bg);
            double gnorm0 = norm(g, w_size, bg);

            f = fun_obj->fun(w, b);
            fun_obj->grad(w, g, b, bg);
            double gnorm = norm(g, w_size, bg);

            if (gnorm <= eps * gnorm0) {
                search = 0;
            }

            while (iter <= max_iter && search) {
                fun_obj->get_diag_preconditioner(M, bM);

                for (uint64_t i = 0; i < w_size; i++) {
                    M[i] = (1 - alpha_pcg) + alpha_pcg * M[i];
                }
                bM = (1 - alpha_pcg) + alpha_pcg * bM;

                pcg(g, M, s, r, bg, bM, bs, br);
                fold = f;
                step_size = fun_obj->linesearch_and_update(&f, init_step_size, w, s, g, b, bs, bg);
                if (step_size == 0) {
                    printf("WARNING: line search fails\n");
                    break;
                }

                fun_obj->grad(w, g, b, bg);
                gnorm = norm(g, w_size, bg);
                if (gnorm <= eps * gnorm0) {
                    break;
                }
                if (f < -1.0e+32) {
                    printf("WARNING: f < -1.0e+32\n");
                    break;
                }
                actred = fold - f;
                if (fabs(actred) <= 1.0e-12 * fabs(f)) {
                    printf("WARNING: actred too small\n");
                    break;
                }
                iter++;
            }
            if (iter >= max_iter){
                printf("\nWARNING: reaching max number of Newton iterations\n");
                }
        }

        double norm(dvec_wrapper_t g, uint64_t size, value_type &bg) {
            double ans = 0;
            for (size_t i = 0; i < size; i++) {
                ans += (double) g[i] * g[i];
            }
            ans += (double) bg * bg;
            return sqrt(ans);
        }

        int pcg(dvec_wrapper_t g, dvec_wrapper_t M, dvec_wrapper_t s, dvec_wrapper_t r, value_type &bg, value_type &bM, value_type &bs, value_type &br) {
            uint64_t n = fun_obj->get_w_size();
            double one = 1;
            double zTr, znewTrnew, alpha, beta, cgtol, dHd;
            double Q = 0, newQ, Qdiff;

            dvec_t tmp_d, tmp_Hd, tmp_z;
            tmp_d.resize(n, 0);
            tmp_Hd.resize(n, 0);
            tmp_z.resize(n, 0);
            dvec_wrapper_t d(tmp_d);
            dvec_wrapper_t Hd(tmp_Hd);
            dvec_wrapper_t z(tmp_z);
            value_type bd=0, bHd=0, bz=0;
            bool bias_flag = fun_obj->get_bias() > 0;
            for (size_t i = 0; i < n; i++) {
                s[i] = 0;
                r[i] = -g[i];
                z[i] = r[i] / M[i];
                d[i] = z[i];
            }
            if (bias_flag) {
                bs = 0;
                br = -bg;
                bz = br / bM;
                bd = bz;
            }
            zTr = do_dot_product(z, r);
            if (bias_flag) {
                zTr += (double) bz * br;
            }
            double gMinv_norm = sqrt(zTr);
            cgtol = min(eps_cg, sqrt(gMinv_norm));
            int cg_iter = 0;
            int max_cg_iter = max((int)n + int(bias_flag), 5);
            while (cg_iter < max_cg_iter) {
                cg_iter++;

                fun_obj->Hv(d, Hd, bd, bHd);
                dHd = do_dot_product(d, Hd);
                if (bias_flag) {
                    dHd += (double) bd * bHd;
                }
                if (dHd <= 1.0e-16) {
                    break;
                }

                alpha = zTr / dHd;

                do_axpy(alpha, d, s);
                if (bias_flag) {
                    bs += alpha * bd;
                }

                alpha = -alpha;
                do_axpy(alpha, Hd, r);
                if (bias_flag) {
                    br += alpha * bHd;
                }

                newQ = -0.5 * (do_dot_product(s, r) - do_dot_product(s, g));
                if (bias_flag) {
                    newQ -= 0.5 * (bs * br - bs * bg);
                }
                Qdiff = newQ - Q;
                if (newQ <= 0 && Qdiff <= 0) {
                    if (cg_iter * Qdiff >= cgtol * newQ) {
                        break;
                    }
                }
                else {
                    printf("WARNING: quadratic approximation > 0 or increasing in CG\n");
                    break;
                }
                Q = newQ;

                for (size_t i = 0; i < n; i++)
                    z[i] = r[i] / M[i];
                if (bias_flag) {
                    bz = br / bM;
                }
                znewTrnew = do_dot_product(z, r);
                if (bias_flag) {
                    znewTrnew += (double) bz * br;
                }
                beta = znewTrnew / zTr;
                do_axpy((beta - 1.0), d, d);
                do_axpy(one, z, d);
                if (bias_flag) {
                    bd = bd * beta;
                    bd += one * bz;
                }
                zTr = znewTrnew;
            }

            if (cg_iter == max_cg_iter){
                printf("WARNING: reaching maximal number of CG steps\n");
            }
            return cg_iter;
        };
    };
} // namespace pecos
#endif
