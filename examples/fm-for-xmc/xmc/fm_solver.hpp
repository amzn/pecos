#ifndef __FM_SOLVER_H__
#define __FM_SOLVER_H__

#include <iomanip>
#include <algorithm>
#include <vector>
#include "utils/matrix.hpp"
#include "utils/parallel.hpp"
#include "utils/random.hpp"
#include "utils/newton.hpp"

#include "xmc/fm_utils.hpp"

#ifdef USEOMP
#include <omp.h>
#endif


namespace pecos {

namespace fm_solver {
    
enum SolverType {
    L2R_LOGLOSS_ADAGRAD=1
};

struct FMParameter {
    FMParameter(
        int solver_type=L2R_LOGLOSS_ADAGRAD,
        int max_iter=10,
        float eta=0.02,
        float reg=0.00002,
        bool auto_stop=false,
        bool identity_biased_init=false,
        bool factorized=false
    ): solver_type(solver_type), max_iter(max_iter), eta(eta), reg(reg), auto_stop(auto_stop), factorized(factorized) {}

    int solver_type;
    size_t max_iter;
    float eta, reg;
    bool auto_stop, factorized, identity_biased_init;
};

#define INF HUGE_VAL
template<typename index_type, typename value_type>
struct FMWorker {

    typedef std::vector<value_type> dvec_t;
    typedef sparse_vec_t<index_type, value_type> svec_wrapper_t;
    typedef dense_vec_t<value_type> dvec_wrapper_t;
    typedef random_number_generator<> rng_t;
    
    FMParameter param;
    // u64_dvec_t index; // used to determine the subset of rows of X are used in the training.
    dvec_t Wx, Wz;
    index_type wx_size, wz_size, k_size;

    FMWorker(): wx_size(0), wz_size(0), k_size(0) {}
    
    void init(index_type wx_size, index_type wz_size, index_type k_size, const FMParameter *param_ptr=NULL) {
        if(param_ptr != NULL) {
            param = *param_ptr;
        }
        this->wx_size = wx_size;
        this->wz_size = wz_size;
        this->k_size = k_size;
        // this->y_nnz = y_nnz;
        
        Wx.resize(wx_size * k_size, 0);
        Wz.resize(wz_size * k_size, 0);
        
        // this->index.reserve(this->y_nnz);
    }

    void lazy_init(size_t wx_size, size_t wz_size, size_t k_size, const FMParameter *param_ptr=NULL) {
        if((wx_size != this->wx_size)
                || (wz_size != this->wz_size)
                || (k_size != this->k_size)
                || ((param_ptr != NULL) && (param_ptr->solver_type != param.solver_type))) {
            init(wx_size, wz_size, k_size, param_ptr);
        } else {
            param = *param_ptr;
        }
    }
    
    template <typename XZ_MAT, typename Y_MAT>
    void solve(const XZ_MAT& X, const XZ_MAT& Z, const Y_MAT& Y,  
               const XZ_MAT& val_X, const XZ_MAT& val_Z, const Y_MAT& val_Y,
               int seed=0) {
        if(param.solver_type == L2R_LOGLOSS_ADAGRAD) {
            solve_l2r_logloss_adagrad(X, Z, Y, val_X, val_Z, val_Y, seed);
        }
    }
    
    template <typename VEC>
    float forward(const VEC& xi, const VEC& zi) {
        
        pecos::drm_t curr_Wx;
        curr_Wx.rows = this->wx_size;
        curr_Wx.cols = this->k_size;
        curr_Wx.val = this->Wx.data();
        
        pecos::drm_t curr_Wz;
        curr_Wz.rows = this->wz_size;
        curr_Wz.cols = this->k_size;
        curr_Wz.val = this->Wz.data();
        
        const size_t x_nnz = xi.get_nnz();
        const size_t z_nnz = zi.get_nnz();
        
        float v1, v2;
        dvec_wrapper_t w1, w2;
        
        float t = 0;
        // for loop over Xs and Zs
        for (size_t j1 = 0; j1 < x_nnz + z_nnz; ++j1) {

            if (j1 < x_nnz) {
                v1 = xi.val[j1];
                w1 = curr_Wx.get_row(get_ind(xi, j1));
            }
            else {
                v1 = zi.val[j1 - x_nnz];
                w1 = curr_Wz.get_row(get_ind(zi, j1 - x_nnz));
            }

            for (size_t j2 = j1 + 1; j2 < x_nnz + z_nnz; ++j2) {

                if (j2 < x_nnz) {
                    v2 = xi.val[j2];
                    w2 = curr_Wx.get_row(get_ind(xi, j2));
                }
                else {
                    v2 = zi.val[j2 - x_nnz];
                    w2 = curr_Wz.get_row(get_ind(zi, j2 - x_nnz));
                }
                t += pecos::do_dot_product(w1.val, w2.val, this->k_size) * v1 * v2;
            }
        }
        return t;
    }
    
    template <typename VEC>
    void backward(const VEC& xi, const VEC& zi, float kappa, drm_t& Gx, drm_t& Gz) {
    
        const size_t x_nnz = xi.get_nnz();
        const size_t z_nnz = zi.get_nnz();
    
        #ifdef VECGRAD
        // allocate space to place gradients.
        dvec_t g1, g2;
        g1.resize(this->k_size, 0);
        g2.resize(this->k_size, 0);

        dvec_wrapper_t curr_g1(g1);
        dvec_wrapper_t curr_g2(g2);
        #endif

        // wrap vectors in matrices
        pecos::drm_t curr_Wx;
        curr_Wx.rows = this->wx_size;
        curr_Wx.cols = this->k_size;
        curr_Wx.val = this->Wx.data();

        pecos::drm_t curr_Wz;
        curr_Wz.rows = this->wz_size;
        curr_Wz.cols = this->k_size;
        curr_Wz.val = this->Wz.data();
        
        for (size_t j1 = 0; j1 < x_nnz + z_nnz; ++j1) {
                    
            float v1, v2;
            dvec_wrapper_t w1, w2, G1, G2;

            if (j1 < x_nnz) {
                v1 = xi.val[j1];
                w1 = curr_Wx.get_row(get_ind(xi, j1));
                G1 = Gx.get_row(get_ind(xi, j1));
            }
            else {
                v1 = zi.val[j1 - x_nnz];
                w1 = curr_Wz.get_row(get_ind(zi, j1 - x_nnz));
                G1 = Gz.get_row(get_ind(zi, j1 - x_nnz));
            }

            for (size_t j2 = j1 + 1; j2 < x_nnz + z_nnz; ++j2) {

                if (j2 < x_nnz) {
                    v2 = xi.val[j2];
                    w2 = curr_Wx.get_row(get_ind(xi, j2));
                    G2 = Gx.get_row(get_ind(xi, j2));
                }
                else {
                    v2 = zi.val[j2 - x_nnz];
                    w2 = curr_Wz.get_row(get_ind(zi, j2 - x_nnz));
                    G2 = Gz.get_row(get_ind(zi, j2 - x_nnz));
                }

                float multiplier = kappa * v1 * v2;
                
                #ifdef VECGRAD
                // zero_grad
                std::fill(g1.begin(), g1.end(), 0);
                std::fill(g2.begin(), g2.end(), 0);
                
                // g1 = lambda * w1 + kappa * w2 * v1 * v2.
                // g2 = lambda * w2 + kappa * w1 * v1 * v2.
                
                pecos::do_axpy(this->param.reg, w1.val, curr_g1.val, this->k_size);
                pecos::do_axpy(multiplier, w2.val, curr_g1.val, this->k_size);

                pecos::do_axpy(this->param.reg, w2.val, curr_g2.val, this->k_size);
                pecos::do_axpy(multiplier, w1.val, curr_g2.val, this->k_size);
                #endif

                // accumulate gradient square.
                // pecos::do_ax2py(1.0, curr_g1, G1);
                // pecos::do_ax2py(1.0, curr_g2, G2);
                #pragma GCC ivdep
                for (size_t d = 0; d < this->k_size; ++d) {

                    #ifdef VECGRAD
                    float g1_ = g1[d];
                    float g2_ = g2[d];
                    #else
                    float g1_ = this->param.reg * w1.val[d] + multiplier * w2.val[d];
                    float g2_ = this->param.reg * w2.val[d] + multiplier * w1.val[d];
                    #endif

                    G1.val[d] += g1_ * g1_;
                    G2.val[d] += g2_ * g2_;

                    // descent.
                    w1.val[d] -= this->param.eta / std::sqrt(G1.val[d]) * g1_;
                    w2.val[d] -= this->param.eta / std::sqrt(G2.val[d]) * g2_;
                }
            }
        }
    }
    
    template <typename VEC>
    float forward_factorized(const VEC& x, const VEC& z, dvec_wrapper_t& emb_sum) {
        
        pecos::drm_t curr_Wx;
        curr_Wx.rows = this->wx_size;
        curr_Wx.cols = this->k_size;
        curr_Wx.val = this->Wx.data();
        
        pecos::drm_t curr_Wz;
        curr_Wz.rows = this->wz_size;
        curr_Wz.cols = this->k_size;
        curr_Wz.val = this->Wz.data();
        
        dvec_t ex, ez;
        ex.resize(this->k_size, 0);
        ez.resize(this->k_size, 0);
        
        dvec_wrapper_t curr_ex(ex);
        dvec_wrapper_t curr_ez(ez);
        
        // generate embeddings O(dk).
        mat_x_vec(curr_Wx, x, curr_ex);
        mat_x_vec(curr_Wz, z, curr_ez);
        
        // for (size_t i = 0; i < this->k_size; ++i) std::cout << curr_ex.val[i] << " ";
        // std::cout << std::endl;
        // for (size_t i = 0; i < this->k_size; ++i) std::cout << curr_ez.val[i] << " ";
        // std::cout << std::endl;
        
        float x_bias = get_bias(x, curr_Wx), z_bias = get_bias(z, curr_Wz);
        
        // std::cout << x_bias << " " << z_bias << std::endl;
        
        float t = pecos::do_dot_product(curr_ex.val, curr_ez.val, this->k_size) + x_bias + z_bias;
        
        pecos::do_axpy(1, curr_ex.val, emb_sum.val, this->k_size);
        pecos::do_axpy(1, curr_ez.val, emb_sum.val, this->k_size);
        
        return t;
    }
    
    template <typename VEC>
    void backward_factorized(const VEC& x, const VEC& z, dvec_wrapper_t& emb_sum, float kappa,
                             drm_t& Gx, drm_t& Gz) {
        
        pecos::drm_t curr_Wx;
        curr_Wx.rows = this->wx_size;
        curr_Wx.cols = this->k_size;
        curr_Wx.val = this->Wx.data();
        
        pecos::drm_t curr_Wz;
        curr_Wz.rows = this->wz_size;
        curr_Wz.cols = this->k_size;
        curr_Wz.val = this->Wz.data();
        
        const size_t x_nnz = x.get_nnz();
        const size_t z_nnz = z.get_nnz();
        
        /*
        dvec_t dummy;
        dummy.resize(this->k_size, 0);
        dvec_wrapper_t dummy_(dummy);
        
        dvec_t dummy2;
        dummy2.resize(this->k_size, 0);
        dvec_wrapper_t dummy2_(dummy2);
        */
        
        for (size_t d = 0; d < x_nnz + z_nnz; ++d) {
            float v;
            dvec_wrapper_t w, G;

            if (d < x_nnz) {
                w = curr_Wx.get_row(get_ind(x, d));
                G = Gx.get_row(get_ind(x, d));
                v = x.val[d];
            }
            else {
                w = curr_Wz.get_row(get_ind(z, d - x_nnz));
                G = Gz.get_row(get_ind(z, d - x_nnz));
                v = z.val[d - x_nnz];
            }
            #pragma GCC ivdep
            for (size_t j = 0; j < this->k_size; ++j) {
                float g = kappa * (emb_sum.val[j] - w.val[j] * v) * v + this->param.reg * w.val[j];
                G.val[j] += g * g;
                w.val[j] -= this->param.eta / std::sqrt(G.val[j]) * g;
            }
        }
    }
    
    template <typename XZ_MAT, typename Y_MAT>
    long double eval_loss(const XZ_MAT &X, const XZ_MAT& Z, const Y_MAT& Y) {
        
        long double loss = 0;
        
#ifdef USEOMP
#pragma omp parallel for schedule(static) reduction(+: loss)
#endif
        for(size_t i = 0; i < Y.rows; ++i) {
            
            dvec_t dummy;
            dummy.resize(this->k_size, 0);
            dvec_wrapper_t dummy_(dummy);
            
            const auto& xi = X.get_row(i);
            const auto& yi = Y.get_row(i);
                
            for (size_t j = 0; j < yi.get_nnz(); ++j) {
                
                const auto& zi = Z.get_row(yi.idx[j]);
                
                const float y = yi.val[j];
                
                const double t = this->param.factorized ? forward_factorized(xi, zi, dummy_) : forward(xi, zi);
                
                const double expnyt = std::exp(-y * t);
                loss += std::log1p(expnyt);
            }
        }
        
        loss /= Y.get_nnz();
        
        return loss;
    }
    
    void save(FILE *fp) const {
        pecos::file_util::fput_multiple<index_type>(&(this->wx_size), 1, fp);
        pecos::file_util::fput_multiple<index_type>(&(this->wz_size), 1, fp);
        pecos::file_util::fput_multiple<index_type>(&(this->k_size), 1, fp);
        
        pecos::file_util::fput_multiple<value_type>(&(this->Wx[0]), this->wx_size * this->k_size, fp);
        pecos::file_util::fput_multiple<value_type>(&(this->Wz[0]), this->wz_size * this->k_size, fp);
    }
    
    void load(FILE *fp) {
        pecos::file_util::fget_multiple<index_type>(&(this->wx_size), 1, fp);
        pecos::file_util::fget_multiple<index_type>(&(this->wz_size), 1, fp);
        pecos::file_util::fget_multiple<index_type>(&(this->k_size), 1, fp);
        
        this->Wx.resize(this->wx_size * this->k_size, 0);
        pecos::file_util::fget_multiple<value_type>(&(this->Wx[0]), this->wx_size * this->k_size, fp);
        
        this->Wz.resize(this->wz_size * this->k_size, 0);
        pecos::file_util::fget_multiple<value_type>(&(this->Wz[0]), this->wz_size * this->k_size, fp);
    }
       
    template <typename XZ_MAT, typename Y_MAT>
    void solve_l2r_logloss_adagrad(const XZ_MAT &X, const XZ_MAT& Z, const Y_MAT& Y,
                                   const XZ_MAT &val_X, const XZ_MAT& val_Z, const Y_MAT& val_Y, int seed) {
        rng_t rng(seed);
        // initialize index by copy pointers to Y
        u64_dvec_t y_rows, y_cols, index;
        dvec_t y_vals;
        
        int cnt = 0;
        for(size_t i = 0; i < Y.rows; ++i) {
            const auto& yi = Y.get_row(i);
            for (size_t j = 0; j < yi.get_nnz(); ++j) {
                y_rows.push_back(i);
                y_cols.push_back(yi.idx[j]);
                y_vals.push_back(yi.val[j]);
                index.push_back(cnt);
                ++cnt;
            }
        }
        
        // initialize weights according to paper.
        // Juan et al. (2016) Section 3.1
        const float sqrt_k = 1/std::sqrt(this->k_size);
        #ifdef DETERMINISTIC
        for(size_t i = 0; i < this->wx_size * this->k_size; i++) {
            this->Wx[i] = sqrt_k / 2;
        }
        for(size_t i = 0; i < this->wz_size * this->k_size; i++) {
            this->Wz[i] = sqrt_k / 2;
        }
        #else
        for(size_t i = 0; i < this->wx_size * this->k_size; i++) {
            this->Wx[i] = rng.uniform(0.0, sqrt_k);
        }
        if (param.identity_biased_init && (this->wx_size == this->wz_size)) {
            std::cout << "Initialize weight with identity bias." << std::endl;
            for(size_t i = 0; i < this->wz_size * this->k_size; i++) {
                this->Wz[i] = this->Wx[i];
            }
        }
        else {
            for(size_t i = 0; i < this->wz_size * this->k_size; i++) {
                this->Wz[i] = rng.uniform(0.0, sqrt_k);
            }
        }
        #endif
        
        // for implementing auto-stop
        dvec_t prev_Wx;
        dvec_t prev_Wz;
        
        // initialize G to 1 (gradient square sum)
        // Juan et al. (2016) Section 3.1
        dvec_t Gx;
        dvec_t Gz;
        Gx.resize(wx_size * k_size, 1);
        Gz.resize(wz_size * k_size, 1);
        
        // start training loop.
        pecos::StopW stopw = pecos::StopW();
        
        long double va_loss = std::numeric_limits<double>::max(),
                    best_va_loss = std::numeric_limits<double>::max();
        float momentum = 0.9;
        int log_freq = 10000;
        
        for (size_t iter = 0; iter < param.max_iter; ++iter) {
            
            stopw.reset();
            
            #ifndef DETERMINISTIC
            // shuffle
            rng.shuffle(index.begin(), index.end());
            #endif
            
            long double loss = 0;
            int progress = 0;
            double elapsed_time = 0, ema_ipt = 0;

#ifdef USEOMP
#pragma omp parallel for schedule(static) reduction(+: loss)
#endif
            for(auto i = index.begin(); i < index.end(); i++) { // cannot use C++11 syntactic sugar for OpenMP
                // retreive data for given index.
                const size_t ri = y_rows[*i];
                const size_t ci = y_cols[*i];
                const float yi = y_vals[*i];
                
                const auto& xi = X.get_row(ri);
                const auto& zi = Z.get_row(ci);
                
                pecos::drm_t curr_Gx;
                curr_Gx.rows = this->wx_size;
                curr_Gx.cols = this->k_size;
                curr_Gx.val = Gx.data();

                pecos::drm_t curr_Gz;
                curr_Gz.rows = this->wz_size;
                curr_Gz.cols = this->k_size;
                curr_Gz.val = Gz.data();
                
                // skip instance if feature is empty
                if (xi.get_nnz() + zi.get_nnz() <= 1) continue;
                
                if (this->param.factorized) {
                    // allocate space to place embeddings.
                    dvec_t emb_sum;
                    emb_sum.resize(this->k_size, 0);
                    dvec_wrapper_t curr_emb_sum(emb_sum);
                    
                    // forward get embeddings ex, ez, and t = phi(x, z)
                    double t = forward_factorized(xi, zi, curr_emb_sum);

                    double expnyt = std::exp(-yi * t);
                    loss += std::log1p(expnyt);
                    float kappa = -yi * expnyt / (1 + expnyt);
                    // std::cout << loss << std::endl;

                    // backward
                    backward_factorized(xi, zi, curr_emb_sum, kappa, curr_Gx, curr_Gz);
                }
                else {
                    double t = forward(xi, zi);
                    
                    double expnyt = std::exp(-yi * t);
                    loss += std::log1p(expnyt);
                    float kappa = -yi * expnyt / (1 + expnyt);

                    backward(xi, zi, kappa, curr_Gx, curr_Gz);
                }
                
                if (std::isnan(loss)) {
                    throw std::overflow_error("Overflow in loss result in NaN. Considering reducing learning rate or increasing weight regularization.");
                }
                
                #ifndef USEOMP
                progress++;
                
                // show progress bar.
                if (progress % log_freq == 0) {
                    double curr_elapsed_time = stopw.getElapsedTimeMicro() / 1000000.;
                    ema_dt = momentum * ema_dt + (1 - momentum) * (curr_elapsed_time - elapsed_time);
                    elapsed_time = curr_elapsed_time;
                    log_progress(progress, index.size(), elapsed_time, emd_dt);
                }
                #else
                if(omp_get_thread_num() == 0) {
                    progress++;
                
                    // show progress bar.
                    if (progress % log_freq == 0) {
                        double curr_elapsed_time = stopw.getElapsedTimeMicro() / 1000000.;
                        double ipt = omp_get_num_threads() * log_freq / (curr_elapsed_time - elapsed_time); 
                        ema_ipt = momentum * ema_ipt + (1 - momentum) * (ipt);
                        elapsed_time = curr_elapsed_time;
                        log_progress(progress * omp_get_num_threads(), index.size(), elapsed_time, ema_ipt);
                    }
                }
                #endif
            }
            
            float trn_time = stopw.getElapsedTimeMicro();
            
            loss /= index.size();
            
            
            va_loss = eval_loss(val_X, val_Z, val_Y);
            if (va_loss > best_va_loss) {
                if (this->param.auto_stop) {
                    std::cout << std::endl << "Auto-stop. Use model at " << iter << "th iteration." << std::endl;
                    break;
                }
            }
            else {
                prev_Wx = Wx;
                prev_Wz = Wz;
                best_va_loss = va_loss;
            }
            
            double sum_G = 0, cnt_G = 0;
            for(auto& i : Gx) {
                if (i == 1) continue;
                sum_G += i;
                cnt_G += 1;
            }
            for(auto& i : Gz) {
                if (i != 1) continue;
                sum_G += i;
                cnt_G += 1;
            }
            
            // flush out log_progress.
            std::cerr << "\t\t\t\t\t\t\t\t\t\t\r";
            std::cerr.flush();
            
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "iter: " << iter + 1
                      << " logloss: " << loss 
                      << " va_logloss: " << va_loss 
                      << " trn_time: " << trn_time / 1000000.
                      << " avg_G: " << sum_G / cnt_G
                      << std::endl;
        }
        // restore best model
        Wx = prev_Wx;
        Wz = prev_Wz;
    }
    
    void log_progress(int step, int total_steps, double elapsed_time, double ema_ipt) {
        float progress_ratio = (float)step / total_steps;
        
        std::cout << "[";
        int barWidth = 70;
        int pos = barWidth * progress_ratio;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        // double elapsed_time = stopw.getElapsedTimeMicro() / 1000000.;
        int elapsed_mins = (int)elapsed_time / 60, elapsed_secs = (int)elapsed_time % 60;
        
        double remaining_time = (total_steps - step) / ema_ipt;
        int remaining_mins = (int)remaining_time / 60, remaining_secs = (int)remaining_time % 60;
        
        std::cerr << std::fixed << std::setprecision(3)
                  << "] " << (progress_ratio * 100.0) << "% "
                  << elapsed_mins << "m" << elapsed_secs << "s"
                  << " < "
                  << remaining_mins << "m" << remaining_secs << "s"
                  << " (" << ema_ipt << "instance/s)"
                  << "\t\t\t\t\r";
        std::cerr.flush();
    }
    
};
    
} // end of namespace fm_solver
} // end of namespace pecos

#endif // end of __FM_SOLVER_H__
