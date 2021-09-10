import numpy as np
# import pylab as p
import sparse
# from scikits.umfpack import spsolve
import pickle
import scipy.sparse.linalg as sla
# from scipy.sparse import csr_matrix
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.special import xlogy
from scipy.special import logsumexp
# import os
import sys
import time
import numba as nb
# from numba.experimental import jitclass
# from numba import int32, float32    # import the types
# sys.path.append('../')
plt.rcParams.update({'font.size': 16})


# def get_reward():
#     P0 = sparse.COO(pickle.load(open("P2019_4week_1920.pkl", "rb")))
#     P0 = P0.tocsr()
#     cP0 = P0.tocoo()
#     P1 = sparse.COO(pickle.load(open("P2020_4week_1920.pkl", "rb")))
#     P1 = P1.tocsr()
#     cP1 = P1.tocoo()
#     cr0 = sparse.COO(pickle.load(open("R2019_4week_1920.pkl", "rb")))
#     cr1 = sparse.COO(pickle.load(open("R2020_4week_1920.pkl", "rb")))
#     val = []
#     for s, t, v in zip(cP0.row, cP0.col, cP0.data):
#         val.append(v * cr0[s, t])
#     r_tmp0 = scipy.sparse.coo_matrix((val, (cP0.row, cP0.col)), shape=cP0.shape)
#     r0 = np.sum(r_tmp0, axis=1)
#     del cr0, r_tmp0
#     val = []
#     for s, t, v in zip(cP1.row, cP1.col, cP1.data):
#         val.append(v * cr1[s, t])
#     r_tmp1 = scipy.sparse.coo_matrix((val, (cP1.row, cP1.col)), shape=cP1.shape)
#     r1 = np.sum(r_tmp1, axis=1)
#     del cr1, r_tmp1
#     r = np.asarray(np.concatenate([r0, r1], axis=1))
#     np.savetxt('r.txt', r)


# get_reward()


def solve_sparse(A, b, tol=1e-15):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1
    # call the solver on your data
    x, status = scipy.sparse.linalg.bicgstab(A, b, tol=tol, callback=callback)
    return x, status, num_iters


class PolicyGradient:
    def __init__(self, tp=0, his=False, gam=0.99, eps=10**-12):
        self.P0 = sparse.COO(pickle.load(open("P2019_4week_1920.pkl", "rb")))
        self.P1 = sparse.COO(pickle.load(open("P2020_4week_1920.pkl", "rb")))
        self.P = sparse.stack([self.P0, self.P1], axis=0)
        self.P0 = self.P0.tocsr()
        self.P1 = self.P1.tocsr()
        self.Na = 2
        self.Ns = self.P0.shape[0]
        self.r = np.loadtxt('r.txt')

        etp = ('KL', 'rKL', 'alpha')
        self.tau = 0.001
        self.gam = gam
        self.eps = eps
        self.alpha = None
        self.result = None
        self.ref_result = None

        self.resultZ = np.zeros([self.Ns, self.Na])
        self.train_from_history = his
        self.err = [[], [], {}]
        self.err_exact = []
        self.V_exact = None
        self.pi_exact = None

        self.etp = etp[tp]
        if self.etp == 'alpha':
            self.alpha = 0

    def train(self, eta=1, T=20, count_spsolver=True):
        if self.etp == 'KL':
            self.err[0] = []
        if self.etp == 'rKL':
            self.err[1] = []
        if self.etp == 'alpha':
            self.err[2][self.alpha] = []

        pi = np.ones([self.Ns, self.Na]) / self.Na if not self.train_from_history else np.copy(self.result)
        pi_new = np.zeros([self.Ns, self.Na])

        # Idt = np.identity(self.Ns)
        crds = [list(range(self.Ns)), list(range(self.Ns))]
        Idt = sparse.COO(crds, [1] * self.Ns, shape=(self.Ns, self.Ns))
        Idt = Idt.tocsr()

        if self.ref_result is not None:
            self.err_exact = []

        for itr in range(T):
            if self.ref_result is not None:
                self.err_exact.append(np.linalg.norm(self.ref_result-pi, 'fro')/np.linalg.norm(self.ref_result, 'fro'))

            # Construct P_pi
            cP0 = self.P0.tocoo()
            val = []
            for s, v in zip(cP0.row, cP0.data):
                val.append(v * pi[s, 0])
            P_pi_tmp0 = scipy.sparse.coo_matrix((val, (cP0.row, cP0.col)), shape=cP0.shape)

            cP1 = self.P1.tocoo()
            val = []
            for s, v in zip(cP1.row, cP1.data):
                val.append(v * pi[s, 1])
            P_pi_tmp1 = scipy.sparse.coo_matrix((val, (cP1.row, cP1.col)), shape=cP1.shape)

            P_pi = P_pi_tmp0 + P_pi_tmp1

            # Construct r_pi
            r_pi = np.zeros(self.Ns)
            for s in range(self.Ns):
                r_pi[s] = pi[s, 0] * self.r[s, 0] + pi[s, 1] * self.r[s, 1]
            H_pi = 0
            if self.etp == 'KL':
                H_pi = np.sum(xlogy(pi, pi), axis=1)
            if self.etp == 'rKL':
                H_pi = np.sum(-np.log(pi), axis=1) / self.Na
            if self.etp == 'alpha':
                H_pi = 4/(1-self.alpha**2) - 4/(1-self.alpha**2)*self.Na**((self.alpha-1)/2) * \
                       np.sum(pi**((self.alpha+1)/2), axis=1)
            if count_spsolver:
                v_pi, _, num_iters = solve_sparse(Idt - self.gam * P_pi, r_pi - self.tau * H_pi, tol=1e-12)
                print(num_iters)
            else:
                v_pi, _ = sla.bicgstab(Idt - self.gam * P_pi, r_pi - self.tau * H_pi, tol=1e-12)
            # print(np.mean(v_pi))
            # print(np.linalg.norm((Idt-gam * P_pi) @ v_pi - r_pi))
            if self.etp == 'KL':
                pi_new = pi ** (1 - eta)
                for a in range(self.Na):
                    pi_new[:, a] = pi_new[:, a] * np.exp(eta*(self.r[:, a]-(Idt-self.gam*self.P[a].tocsr())@v_pi)/self.tau)

            if self.etp == 'rKL':
                for a in range(self.Na):
                    tmp = self.P[a].tocsr() @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) / pi[s, a]
                        pi_new[s, a] -= eta * self.Na * (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau

                for s in range(self.Ns):
                    def equation(x):
                        return 1 / (x + pi_new[s, 0] + 1e-300) + 1 / (x + pi_new[s, 1] + 1e-300) - 1
                    low = max(self.Na-np.max(pi_new[s, :]), -np.min(pi_new[s, :]))
                    high = self.Na - np.min(pi_new[s, :])
                    cs = low
                    # if equation(low) * equation(high) >= 0:
                    #     print(pi_new[s, :])
                    #     print(equation(low))
                    #     print(equation(high))
                    # if np.abs(equation(low)) < 1e-15:
                    #     cs = low
                    if np.abs(equation(high)) < 1e-15:
                        cs = high
                    if np.abs(equation(low)) >= 1e-15 and np.abs(equation(high)) >= 1e-15:
                        cs = scipy.optimize.bisect(equation, low, high)
                    # cs = scipy.optimize.newton_krylov(equation, np.array([1]), f_tol=1e-12)
                    # print(np.abs(equation(cs)) < 1*10**-10)
                    for a in range(self.Na):
                        pi_new[s, a] = 1 / (pi_new[s, a] + cs)
                    # print(s)
                    # print(np.abs(np.sum(pi_new[s, :])-1) < 1*10**-10)
                    # print(np.min(pi_new[s, :]))

            if self.etp == 'alpha':
                for a in range(self.Na):
                    tmp = self.P[a].tocsr() @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) * (pi[s, a] ** ((self.alpha - 1) / 2))
                        pi_new[s, a] -= (1 - self.alpha) / 2 * eta * (self.Na ** ((1 - self.alpha) / 2)) \
                            * (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau

                for s in range(self.Ns):
                    alpha = self.alpha

                    def equation(x):
                        return (x + pi_new[s, 0] + 1e-300) ** (2 / (alpha - 1)) + (x + pi_new[s, 1] + 1e-300) ** (2 / (alpha - 1)) - 1
                    low = max(self.Na ** ((1 - self.alpha) / 2) - np.max(pi_new[s, :]), -np.min(pi_new[s, :]))
                    high = self.Na ** ((1 - self.alpha) / 2) - np.min(pi_new[s, :])
                    cs = low
                    # if np.abs(equation(low)) < 1e-15:
                    #     cs = low
                    if np.abs(equation(high)) < 1e-15:
                        cs = high
                    if np.abs(equation(low)) >= 1e-15 and np.abs(equation(high)) >= 1e-15:
                        cs = scipy.optimize.bisect(equation, low, high)
                    for a in range(self.Na):
                        pi_new[s, a] = (pi_new[s, a] + cs) ** (2 / (self.alpha-1))

            for s in range(self.Ns):
                pi_new[s, :] /= sum(pi_new[s, :])
            err = np.linalg.norm(pi_new - pi, 'fro') / np.linalg.norm(pi, 'fro')
            print(err)
            if self.etp == 'KL':
                self.err[0].append(err)
            if self.etp == 'rKL':
                self.err[1].append(err)
            if self.etp == 'alpha':
                self.err[2][self.alpha].append(err)
            if err < self.eps:
                print(itr)
                break
            pi[:] = pi_new[:]
        self.result = np.copy(pi)
        self.ref_result = np.copy(pi)

    def set_tau(self, tau):
        self.tau = tau
        self.ref_result = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_history(self, his):
        self.train_from_history = his
        if his and self.result is None:
            self.result = np.ones([self.Ns, self.Na]) / self.Na

    def clear_history(self):
        self.result = self.result = np.ones([self.Ns, self.Na]) / self.Na

    def set_etp(self, tp):
        etp = ('KL', 'rKL', 'alpha')
        self.etp = etp[tp]
        if self.etp == 'alpha':
            self.alpha = 0
        self.ref_result = None

    def get_value(self, pi=None, reg=False):
        if self.result is None:
            return
        if pi is None:
            pi = self.result
        Idt = np.identity(self.Ns)
        P_pi = np.zeros([self.Ns, self.Ns])
        for s in range(self.Ns):
            for t in range(self.Ns):
                P_pi[s, t] = np.dot(pi[s, :], self.P[:, s, t])
        r_pi = np.zeros(self.Ns)
        for s in range(self.Ns):
            r_pi[s] = np.dot(pi[s, :], self.r[s, :])
        H_pi = 0
        if reg:
            if self.etp == 'KL':
                H_pi = np.sum(xlogy(pi, pi), axis=1)
            if self.etp == 'rKL':
                H_pi = np.sum(-np.log(pi), axis=1) / self.Na
            if self.etp == 'alpha':
                H_pi = 4/(1-self.alpha**2) - 4/(1-self.alpha**2)*self.Na**((self.alpha-1)/2) * \
                       np.sum(pi**((self.alpha+1)/2), axis=1)
        v_pi = np.linalg.solve(Idt - self.gam * P_pi, r_pi - self.tau * H_pi)
        return v_pi

    def get_expect_value(self, pi=None):
        if self.result is None:
            return
        if pi is None:
            pi = self.result
        P_pi = np.zeros([self.Ns, self.Ns])
        for s in range(self.Ns):
            for t in range(self.Ns):
                P_pi[s, t] = np.dot(pi[s, :], self.P[:, s, t])
        r_pi = np.zeros(self.Ns)
        for s in range(self.Ns):
            r_pi[s] = np.dot(pi[s, :], self.r[s, :])
        _, evec = sla.eigs(P_pi.T, k=1, which='LM')
        d_pi = (evec / evec.sum()).real
        d_pi = d_pi.reshape(len(d_pi))
        return np.dot(d_pi, r_pi)

    def make_fig(self, alpha1, alpha2, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = self.err[0]
        err1 = self.err[1]
        err2 = self.err[2][alpha1]
        err3 = self.err[2][alpha2]
        ax.plot(err0, label='KL', linewidth=linewidth)
        ax.plot(err1, label='reversed KL', linewidth=linewidth)
        ax.plot(err2, label='Hellinger', linewidth=linewidth)
        ax.plot(err3, label='alpha divergence (alpha=' + str(alpha2) + '-3', linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('relative change of policy', fontsize=22)
        ax.legend(loc='upper right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig2(self, alpha1, alpha2, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = np.log(np.abs(np.log(self.err[0])))
        err1 = np.log(np.abs(np.log(self.err[1])))
        err2 = np.log(np.abs(np.log(self.err[2][alpha1])))
        err3 = np.log(np.abs(np.log(self.err[2][alpha2])))
        ax.plot(err0, label='KL', linewidth=linewidth, marker='+')
        ax.plot(err1, label='reversed KL', linewidth=linewidth, marker='+')
        ax.plot(err2, label='alpha divergence with alpha = ' + str(alpha1), linewidth=linewidth, marker='+')
        ax.plot(err3, label='alpha divergence with alpha = ' + str(alpha2), linewidth=linewidth, marker='+')
        # ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('relative change of policy', fontsize=22)
        ax.legend(loc='lower right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig_exact(self, label='KL', linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = np.log(np.abs(np.log(self.err_exact[:-1]))) if self.err_exact[-1] == 0 \
            else np.log(np.abs(np.log(self.err_exact)))
        ax.plot(err0, label=label, linewidth=linewidth, marker='+')
        ax.plot([0, len(err0) - 1], [0, np.log(2) * (len(err0) - 1)], label='slope=log2', color='g', linestyle='dashed',
                linewidth=0.5)
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('loglog error', fontsize=22)
        ax.legend(loc='upper left')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def train_all(self, alpha1=0, alpha2=-3):
        self.set_etp(0)
        self.train()
        self.set_etp(1)
        self.train()
        self.set_etp(2)
        self.set_alpha(alpha1)
        self.train()
        self.set_alpha(alpha2)
        self.train()

    def value_iteration(self):
        V_pre = np.ones(self.Ns)
        V = np.zeros(self.Ns)
        Q = np.zeros([self.Ns, self.Na])
        cnt = 0
        while np.linalg.norm(V - V_pre) / np.linalg.norm(V_pre) > 10 ** -15:
            cnt += 1
            V_pre = V
            for a in range(self.Na):
                Q[:, a] = self.r[:, a] + self.gam * (self.P[a, :, :] @ V)

            if self.etp == 'KL':
                V = logsumexp(Q/self.tau, axis=1) * self.tau
            if self.etp == 'rKL':
                return
            if self.etp == 'alpha':
                return
            if cnt % 100 == 0:
                print(np.linalg.norm(V - V_pre) / np.linalg.norm(V_pre))

        for a in range(self.Na):
            Q[:, a] = self.r[:, a] + self.gam * (self.P[a, :, :] @ V) - V
        self.V_exact = V
        self.pi_exact = softmax(Q/self.tau, axis=1)
        print(cnt)

    def train_primal_dual_old(self, CV=0.1, CW=0.1, tol=10**-6, step=0.01):
        V = np.zeros(self.Ns)
        W = np.ones(self.Ns)
        Z0 = np.zeros([self.Ns, self.Na])
        Z = np.ones([self.Ns, self.Na]) / self.Na
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0

        while np.max(np.abs(Z-Z0)) > tol:
            cnt += 1
            PZ = np.zeros([self.Ns, self.Ns])
            for s in range(self.Ns):
                for t in range(self.Ns):
                    PZ[s, t] = np.dot(Z[s, :], self.P[:, s, t])
            RZ = np.zeros(self.Ns)
            for s in range(self.Ns):
                RZ[s] = np.dot(Z[s, :], self.r[s, :])
            HZ = np.sum(xlogy(Z, Z), axis=1)

            EV = V - (Idt - self.gam * PZ.T) @ W / CV
            EW = (RZ - (Idt - self.gam * PZ) @ V - self.tau * HZ) / CW - W
            EZ = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EZ[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a, :, :]) @ V) / self.tau - np.log(Z[:, a])
            Z0 = Z
            V = V - dt * EV
            W = W + dt * EW
            Z = Z * np.exp(dt * EZ)
            # print(np.min(W))
            for s in range(self.Ns):
                Z[s, :] /= sum(Z[s, :])
            if cnt % 10 == 0:
                print(cnt)
                print(np.max(np.abs(Z-Z0)))
        self.resultZ[:] = Z[:]
        print(cnt)
        print(np.max(np.abs(Z - Z0)))

    def train_primal_dual(self, CV=0.01, CM=0.01, tol=10**-8, step=0.01):
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        # Idt = np.identity(self.Ns)
        crds = [list(range(self.Ns)), list(range(self.Ns))]
        Idt = sparse.COO(crds, [1] * self.Ns, shape=(self.Ns, self.Ns))
        Idt = Idt.tocsr()
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])

        while np.linalg.norm(M-M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            EV /= CV
            V = (V + dt * EV) / (1 + dt)

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a]) + 1
            H = (H + dt * EM) / (1 + dt)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)

            if cnt % 50 == 0:
                print(cnt)
                print(np.linalg.norm(M-M0) / np.linalg.norm(M))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / sum(M[s, :])
        self.resultZ[:] = Z[:]
        print(M)
        print(cnt)
        print(np.linalg.norm(M-M0) / np.linalg.norm(M))


# spec = [
#     ('value', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
# ]


# @jitclass(spec)
class PolicyGradientNew:
    def __init__(self, tp=0, his=False, gam=0.99, eps=10**-12):
        self.P0 = sparse.COO(pickle.load(open("P2019_4week_1920.pkl", "rb")))
        self.P1 = sparse.COO(pickle.load(open("P2020_4week_1920.pkl", "rb")))
        self.Na = 2
        self.Ns = self.P0.shape[0]
        self.P0 = self.P0.tocsr()
        self.P1 = self.P1.tocsr()
        self.P0 = self.P0.tolil()
        self.P1 = self.P1.tolil()
        for s in range(self.Ns):
            if self.P0[s].nnz == 0:
                self.P0[s, -1] = 1
            if self.P1[s].nnz == 0:
                self.P1[s, -1] = 1
        self.P0 = self.P0.tocsr()
        self.P1 = self.P1.tocsr()
        self.P = sparse.stack([sparse.COO(self.P0), sparse.COO(self.P1.tocoo())], axis=0)

        self.r = np.loadtxt('r.txt')
        self.mu = np.ones(self.Na) / self.Na
        self.init_type = 'prior'
        self.init_dis = None

        etp = ('KL', 'rKL', 'alpha')
        self.tau = 0.001
        self.gam = gam
        self.eps = eps
        self.alpha = None
        self.result = None
        self.ref_result = None
        self.resultZ = None
        self.resultM = None
        self.resultV = None

        self.resultZ = np.zeros([self.Ns, self.Na])
        self.train_from_history = his
        self.err = [[], [], {}]
        self.err_exact = []
        self.err_pd = []
        self.err_pd_exact = []
        self.err_pd_val_pi = []
        self.err_pd_val_V = []
        self.lyap = []
        self.V_exact = None
        self.pi_exact = None
        self.mu_exact = None

        self.etp = etp[tp]
        if self.etp == 'alpha':
            self.alpha = 0

    def set_prior(self, mu):
        for a in range(self.Na):
            self.mu[a] = mu[a]

    def set_init_dis(self, tmp):
        self.init_dis = np.copy(tmp)

    def set_init_type(self, tp):
        self.init_type = tp

    def set_tau(self, tau):
        self.tau = tau
        self.ref_result = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_history(self, his):
        self.train_from_history = his
        if his and self.result is None:
            self.result = np.ones([self.Ns, self.Na]) / self.Na

    def clear_history(self):
        self.result = self.result = np.ones([self.Ns, self.Na]) / self.Na

    def set_etp(self, tp):
        etp = ('KL', 'rKL', 'alpha')
        self.etp = etp[tp]
        if self.etp == 'alpha':
            self.alpha = 0
        self.ref_result = None

    def train(self, eta=1, T=20, count_spsolver=True):
        if self.etp == 'KL':
            self.err[0] = []
        if self.etp == 'rKL':
            self.err[1] = []
        if self.etp == 'alpha':
            self.err[2][self.alpha] = []

        pi = None
        if self.init_type == 'prior':
            pi = np.ones([self.Ns, 1]) * self.mu if not self.train_from_history else np.copy(self.result)
        if self.init_type == 'uniform':
            pi = np.ones([self.Ns, self.Na]) / self.Na if not self.train_from_history else np.copy(self.result)
        if self.init_type == 'random':
            pi = np.random.exponential(size=[self.Ns, self.Na])
            pi = pi / np.sum(pi, axis=1, keepdims=True)
        if self.init_type == 'assigned' and self.init_dis is not None:
            pi = np.copy(self.init_dis)

        pi_new = np.zeros([self.Ns, self.Na])

        # Idt = np.identity(self.Ns)
        crds = [list(range(self.Ns)), list(range(self.Ns))]
        Idt = sparse.COO(crds, [1] * self.Ns, shape=(self.Ns, self.Ns))
        Idt = Idt.tocsr()

        if self.ref_result is not None:
            self.err_exact = []

        for itr in range(T):
            if self.ref_result is not None:
                self.err_exact.append(np.linalg.norm(self.ref_result-pi, 'fro')/np.linalg.norm(self.ref_result, 'fro'))

            # Construct P_pi
            cP0 = self.P0.tocoo()
            val = []
            for s, v in zip(cP0.row, cP0.data):
                val.append(v * pi[s, 0])
            P_pi_tmp0 = scipy.sparse.coo_matrix((val, (cP0.row, cP0.col)), shape=cP0.shape)

            cP1 = self.P1.tocoo()
            val = []
            for s, v in zip(cP1.row, cP1.data):
                val.append(v * pi[s, 1])
            P_pi_tmp1 = scipy.sparse.coo_matrix((val, (cP1.row, cP1.col)), shape=cP1.shape)

            P_pi = P_pi_tmp0 + P_pi_tmp1

            # Construct r_pi
            r_pi = np.zeros(self.Ns)
            for s in range(self.Ns):
                r_pi[s] = pi[s, 0] * self.r[s, 0] + pi[s, 1] * self.r[s, 1]
            H_pi = 0
            if self.etp == 'KL':
                H_pi = np.sum(xlogy(pi, pi / self.mu), axis=1)
            if self.etp == 'rKL':
                H_pi = -np.sum(xlogy(self.mu, pi / self.mu), axis=1)
            if self.etp == 'alpha':
                H_pi = 4/(1-self.alpha**2) - 4/(1-self.alpha**2) * \
                       np.sum(self.mu*(pi/self.mu)**((self.alpha+1)/2), axis=1)
            if count_spsolver:
                v_pi, _, num_iters = solve_sparse(Idt - self.gam * P_pi, r_pi - self.tau * H_pi, tol=1e-12)
                print(num_iters)
            else:
                v_pi, _ = sla.bicgstab(Idt - self.gam * P_pi, r_pi - self.tau * H_pi, tol=1e-12)
                # print((Idt - self.gam * P_pi).shape)
                # print((r_pi - self.tau * H_pi).shape)
                # print(sla.bicgstab(Idt - self.gam * P_pi, r_pi - self.tau * H_pi, tol=1e-12))
            # print(np.mean(v_pi))
            # print(np.linalg.norm((Idt-gam * P_pi) @ v_pi - r_pi))
            if self.etp == 'KL':
                pi_new = pi ** (1 - eta)
                for a in range(self.Na):
                    pi_new[:, a] *= self.mu[a] ** eta
                    pi_new[:, a] = pi_new[:, a] * np.exp(eta*(self.r[:, a]-(Idt-self.gam*self.P[a].tocsr())@v_pi)/self.tau)

            if self.etp == 'rKL':
                for a in range(self.Na):
                    tmp = self.P[a].tocsr() @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) / pi[s, a]
                        pi_new[s, a] -= eta / self.mu[a] * (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau

                for s in range(self.Ns):
                    def equation(x):
                        if x == -np.min(pi_new[s, :] * self.mu):
                            return 1e+300
                        return 1 / (x/self.mu[0] + pi_new[s, 0] + 1e-300) + 1 / (x/self.mu[1] + pi_new[s, 1] + 1e-300) - 1
                    low = max(np.min((self.Na-pi_new[s, :]) * self.mu), -np.min(pi_new[s, :] * self.mu))
                    high = np.max((self.Na-pi_new[s, :]) * self.mu)
                    cs = low
                    # if equation(low) * equation(high) >= 0:
                    #     print(pi_new[s, :])
                    #     print(low)
                    #     print(high)
                    #     print(equation(low))
                    #     print(equation(high))
                    # if np.abs(equation(low)) < 1e-15:
                    #     cs = low
                    if np.abs(equation(high)) < 1e-15:
                        cs = high
                    if np.abs(equation(low)) >= 1e-15 and np.abs(equation(high)) >= 1e-15:
                        cs = scipy.optimize.bisect(equation, low, high)
                    # cs = scipy.optimize.newton_krylov(equation, np.array([1]), f_tol=1e-12)
                    # print(np.abs(equation(cs)) < 1*10**-10)
                    for a in range(self.Na):
                        pi_new[s, a] = 1 / (pi_new[s, a] + cs / self.mu[a])
                    # print(s)
                    # print(np.abs(np.sum(pi_new[s, :])-1) < 1*10**-10)
                    # print(np.min(pi_new[s, :]))

            if self.etp == 'alpha':
                for a in range(self.Na):
                    tmp = self.P[a].tocsr() @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) * (pi[s, a] ** ((self.alpha - 1) / 2))
                        pi_new[s, a] -= (1 - self.alpha) / 2 * eta * (self.mu[a] ** ((self.alpha - 1) / 2)) \
                            * (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau

                for s in range(self.Ns):
                    alpha = self.alpha

                    def equation(x):
                        if x == -np.min(pi_new[s, :] * self.mu):
                            return 1e+30
                        return (x*self.mu[0]**((self.alpha-1)/2) + pi_new[s, 0] + 1e-30) ** (2/(alpha-1)) \
                            + (x*self.mu[1]**((self.alpha-1)/2) + pi_new[s, 1] + 1e-30) ** (2/(alpha-1)) - 1
                    low = max(np.min((self.Na**((1-self.alpha)/2)-pi_new[s, :])*self.mu**((1-self.alpha)/2)), -np.min(pi_new[s, :] * self.mu**((1-self.alpha)/2)))
                    high = np.max((self.Na**((1-self.alpha)/2)-pi_new[s, :])*self.mu**((1-self.alpha)/2))
                    # low = max(self.Na ** ((1 - self.alpha) / 2) - np.max(pi_new[s, :]), -np.min(pi_new[s, :]))
                    # high = self.Na ** ((1 - self.alpha) / 2) - np.min(pi_new[s, :])
                    cs = low
                    if np.abs(equation(high)) < 1e-15:
                        cs = high
                    if np.abs(equation(low)) >= 1e-15 and np.abs(equation(high)) >= 1e-15:
                        cs = scipy.optimize.bisect(equation, low, high)
                    for a in range(self.Na):
                        pi_new[s, a] = (pi_new[s, a] + cs * self.mu[a] ** ((self.alpha-1) / 2)) ** (2 / (self.alpha-1))

            for s in range(self.Ns):
                pi_new[s, :] /= sum(pi_new[s, :])
            err = np.linalg.norm(pi_new - pi, 'fro') / np.linalg.norm(pi, 'fro')
            print(err)
            if self.etp == 'KL':
                self.err[0].append(err)
            if self.etp == 'rKL':
                self.err[1].append(err)
            if self.etp == 'alpha':
                self.err[2][self.alpha].append(err)
            if err < self.eps:
                print(itr)
                break
            pi[:] = pi_new[:]
        self.result = np.copy(pi)
        self.ref_result = np.copy(pi)

    def get_value(self, pi=None, reg=False):
        if self.result is None:
            return
        if pi is None:
            pi = self.result
        Idt = np.identity(self.Ns)
        P_pi = np.zeros([self.Ns, self.Ns])
        for s in range(self.Ns):
            for t in range(self.Ns):
                P_pi[s, t] = np.dot(pi[s, :], self.P[:, s, t])
        r_pi = np.zeros(self.Ns)
        for s in range(self.Ns):
            r_pi[s] = np.dot(pi[s, :], self.r[s, :])
        H_pi = 0
        if reg:
            if self.etp == 'KL':
                H_pi = np.sum(xlogy(pi, pi), axis=1)
            if self.etp == 'rKL':
                H_pi = np.sum(-np.log(pi), axis=1) / self.Na
            if self.etp == 'alpha':
                H_pi = 4/(1-self.alpha**2) - 4/(1-self.alpha**2)*self.Na**((self.alpha-1)/2) * \
                       np.sum(pi**((self.alpha+1)/2), axis=1)
        v_pi = np.linalg.solve(Idt - self.gam * P_pi, r_pi - self.tau * H_pi)
        return v_pi

    def get_expect_value(self, pi=None):
        if self.result is None:
            return
        if pi is None:
            pi = self.result
        P_pi = np.zeros([self.Ns, self.Ns])
        for s in range(self.Ns):
            for t in range(self.Ns):
                P_pi[s, t] = np.dot(pi[s, :], self.P[:, s, t])
        r_pi = np.zeros(self.Ns)
        for s in range(self.Ns):
            r_pi[s] = np.dot(pi[s, :], self.r[s, :])
        _, evec = sla.eigs(P_pi.T, k=1, which='LM')
        d_pi = (evec / evec.sum()).real
        d_pi = d_pi.reshape(len(d_pi))
        return np.dot(d_pi, r_pi)

    def make_fig(self, alpha1, alpha2, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = self.err[0]
        err1 = self.err[1]
        err2 = self.err[2][alpha1]
        err3 = self.err[2][alpha2]
        ax.plot(err0, label='KL', linewidth=linewidth)
        ax.plot(err1, label='reversed KL', linewidth=linewidth)
        ax.plot(err2, label='Hellinger', linewidth=linewidth)
        ax.plot(err3, label='alpha divergence (alpha=' + str(alpha2) + '-3', linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('relative change of policy', fontsize=22)
        ax.legend(loc='upper right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig_init(self, linewidth=0.5):
        err0, err1, err2, err3 = None, None, None, None
        if self.etp == 'KL':
            self.set_init_type('uniform')
            self.train()
            err0 = np.copy(self.err[0])
            self.set_init_type('prior')
            self.train()
            err1 = np.copy(self.err[0])
            self.set_init_type('random')
            self.train()
            err2 = np.copy(self.err[0])
            self.set_init_type('random')
            self.train()
            err3 = np.copy(self.err[0])
        if self.etp == 'rKL':
            self.set_init_type('uniform')
            self.train()
            err0 = np.copy(self.err[1])
            self.set_init_type('prior')
            self.train()
            err1 = np.copy(self.err[1])
            self.set_init_type('random')
            self.train()
            err2 = np.copy(self.err[1])
            self.set_init_type('random')
            self.train()
            err3 = np.copy(self.err[1])
        if self.etp == 'alpha':
            self.set_init_type('uniform')
            self.train()
            err0 = np.copy(self.err[2][self.alpha])
            self.set_init_type('prior')
            self.train()
            err1 = np.copy(self.err[2][self.alpha])
            self.set_init_type('random')
            self.train()
            err2 = np.copy(self.err[2][self.alpha])
            self.set_init_type('random')
            self.train()
            err3 = np.copy(self.err[2][self.alpha])
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        ax.plot(err0, label='uniform', linewidth=linewidth)
        ax.plot(err1, label='prior', linewidth=linewidth)
        ax.plot(err2, label='random 1', linewidth=linewidth)
        ax.plot(err3, label='random 2', linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('relative change of policy', fontsize=22)
        ax.legend(loc='upper right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig2(self, alpha1, alpha2, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = np.log(np.abs(np.log(self.err[0])))
        err1 = np.log(np.abs(np.log(self.err[1])))
        err2 = np.log(np.abs(np.log(self.err[2][alpha1])))
        err3 = np.log(np.abs(np.log(self.err[2][alpha2])))
        ax.plot(err0, label='KL', linewidth=linewidth, marker='+')
        ax.plot(err1, label='reversed KL', linewidth=linewidth, marker='+')
        ax.plot(err2, label='alpha divergence with alpha = ' + str(alpha1), linewidth=linewidth, marker='+')
        ax.plot(err3, label='alpha divergence with alpha = ' + str(alpha2), linewidth=linewidth, marker='+')
        # ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('relative change of policy', fontsize=22)
        ax.legend(loc='lower right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig_exact(self, label='KL', linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = np.log(np.abs(np.log(self.err_exact[:-1]))) if self.err_exact[-1] == 0 \
            else np.log(np.abs(np.log(self.err_exact)))
        ax.plot(err0, label=label, linewidth=linewidth, marker='+')
        ax.plot([0, len(err0) - 1], [0, np.log(2) * (len(err0) - 1)], label='slope=log2', color='g', linestyle='dashed',
                linewidth=0.5)
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('loglog error', fontsize=22)
        ax.legend(loc='upper left')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def train_all(self, alpha1=0, alpha2=-3):
        self.set_etp(0)
        self.train()
        self.set_etp(1)
        self.train()
        self.set_etp(2)
        self.set_alpha(alpha1)
        self.train()
        self.set_alpha(alpha2)
        self.train()

    def value_iteration(self):
        V_pre = np.ones(self.Ns)
        V = np.zeros(self.Ns)
        Q = np.zeros([self.Ns, self.Na])
        cnt = 0
        while np.linalg.norm(V - V_pre) / np.linalg.norm(V_pre) > 10 ** -15:
            cnt += 1
            V_pre = V
            for a in range(self.Na):
                Q[:, a] = self.r[:, a] + self.gam * (self.P[a, :, :] @ V)

            if self.etp == 'KL':
                V = logsumexp(Q/self.tau, axis=1) * self.tau
            if self.etp == 'rKL':
                return
            if self.etp == 'alpha':
                return
            if cnt % 100 == 0:
                print(np.linalg.norm(V - V_pre) / np.linalg.norm(V_pre))

        for a in range(self.Na):
            Q[:, a] = self.r[:, a] + self.gam * (self.P[a, :, :] @ V) - V
        self.V_exact = V
        self.pi_exact = softmax(Q/self.tau, axis=1)
        print(cnt)

    def train_primal_dual_old(self, CV=0.1, CW=0.1, tol=10**-6, step=0.01):
        V = np.zeros(self.Ns)
        W = np.ones(self.Ns)
        Z0 = np.zeros([self.Ns, self.Na])
        Z = np.ones([self.Ns, self.Na]) / self.Na
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0

        while np.max(np.abs(Z-Z0)) > tol:
            cnt += 1
            PZ = np.zeros([self.Ns, self.Ns])
            for s in range(self.Ns):
                for t in range(self.Ns):
                    PZ[s, t] = np.dot(Z[s, :], self.P[:, s, t])
            RZ = np.zeros(self.Ns)
            for s in range(self.Ns):
                RZ[s] = np.dot(Z[s, :], self.r[s, :])
            HZ = np.sum(xlogy(Z, Z), axis=1)

            EV = V - (Idt - self.gam * PZ.T) @ W / CV
            EW = (RZ - (Idt - self.gam * PZ) @ V - self.tau * HZ) / CW - W
            EZ = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EZ[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a, :, :]) @ V) / self.tau - np.log(Z[:, a])
            Z0 = Z
            V = V - dt * EV
            W = W + dt * EW
            Z = Z * np.exp(dt * EZ)
            for s in range(self.Ns):
                Z[s, :] /= sum(Z[s, :])
            if cnt % 10 == 0:
                print(cnt)
                print(np.max(np.abs(Z-Z0)))
        self.resultZ[:] = Z[:]
        print(cnt)
        print(np.max(np.abs(Z - Z0)))

    def train_primal_dual(self, CV=0.01, CM=0.01, tol=10**-8, step=0.01):
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        # Idt = np.identity(self.Ns)
        crds = [list(range(self.Ns)), list(range(self.Ns))]
        Idt = sparse.COO(crds, [1] * self.Ns, shape=(self.Ns, self.Ns))
        Idt = Idt.tocsr()
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])

        while np.linalg.norm(M-M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].tocsr().T) @ M[:, a]
            EV /= CV
            V = (V + dt * EV) / (1 + dt)

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a].tocsr()) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a]) + 1
            H = (H + dt * EM) / (1 + dt)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)

            if cnt % 50 == 0:
                print(cnt)
                print(np.linalg.norm(M-M0) / np.linalg.norm(M))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / sum(M[s, :])
        self.resultZ[:] = Z[:]
        print(M)
        print(cnt)
        print(np.linalg.norm(M-M0) / np.linalg.norm(M))

    def _lyap(self, CV, V, M):
        return CV / 2 * (np.linalg.norm(V-self.resultV)) ** 2 + self.tau * np.sum(xlogy(self.resultM, self.resultM)) \
               - self.tau * np.sum(xlogy(self.resultM, M)) + self.tau * np.sum(M - self.resultM)

    def _lyap_mixed(self, CV, V, M, c):
        tmp_b = np.sum(self.resultM, axis=1)
        tmp_M = np.sum(M, axis=1)
        return CV / 2 * (np.linalg.norm(V-self.resultV)) ** 2 + self.tau * np.sum(xlogy(self.resultM, self.resultM)) \
            - self.tau * np.sum(xlogy(self.resultM, M)) + self.tau * c / (1-c) * np.sum(xlogy(tmp_b, tmp_b)) \
            - self.tau * c / (1-c) * np.sum(xlogy(tmp_b, tmp_M)) + self.tau / (1-c) * np.sum(M - self.resultM)

    # @nb.jit(nopython=True)
    def train_primal_dual_pseudo_mix(self, CV=0.01, tol=1e-5, step=0.01, c=0.99):  # with \mu\log\hat{\mu} regularization and a different approximate Hessian
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) * CV
        # M = np.ones([self.Ns, self.Na]) * 199629
        M0 = np.zeros([self.Ns, self.Na])
        V0 = np.ones(self.Ns)
        # Idt = np.identity(self.Ns)
        crds = [list(range(self.Ns)), list(range(self.Ns))]
        Idt = sparse.COO(crds, [1] * self.Ns, shape=(self.Ns, self.Ns))
        Idt = Idt.tocsr()
        cnt = 0
        dt = step
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        self.err_pd_val_V = []
        self.err_pd_val_pi = []
        self.lyap = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        if self.resultM is None or self.resultV is None:
            self.get_mu(CV)
            self.resultM = np.copy(self.mu_exact)
            self.resultV = np.copy(self.V_exact)
            print("Exact mu obtained")

        tmp_err = 1
        while tmp_err > tol:
            M0[:] = M[:]
            V0[:] = V[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            # for a in range(self.Na):
            #     EV += (Idt - self.gam * self.P[a].tocsr().T) @ M[:, a]
            EV += (Idt - self.gam * self.P[0].tocsr().T) @ M[:, 0]
            EV += (Idt - self.gam * self.P[1].tocsr().T) @ M[:, 1]
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            # for a in range(self.Na):
            #     EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a].tocsr()) @ V) / CM
            EM[:, 0] = (self.r[:, 0] - (Idt - self.gam * self.P[0].tocsr()) @ V) / CM
            EM[:, 1] = (self.r[:, 1] - (Idt - self.gam * self.P[1].tocsr()) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                H[s, 0] = np.log(M[s, 0] + 1e-300)
                H[s, 1] = np.log(M[s, 1] + 1e-300)
                # for a in range(self.Na):
                #     H[s, a] = np.log(M[s, a] + 1e-300)

            # print(1)

            for s in range(self.Ns):
                # tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                tmp_w = np.log(M[s, 0] + M[s, 1] + 1e-300)
                # tmp_pi = softmax(H[s, :])
                tmp_pi0 = np.exp(H[s, 0])
                tmp_pi1 = np.exp(H[s, 1])
                tmp_s = tmp_pi0 + tmp_pi1
                tmp_pi0 /= tmp_s
                tmp_pi1 /= tmp_s
                # for a in range(self.Na):
                #     # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                #     H[s, a] = (1-dt) * H[s, a] + dt * (EM[s, a] + tmp_w)
                # tmp_add = np.dot(tmp_pi, EM[s, :] + tmp_w - H[s, :])
                tmp_add = tmp_pi0 * (EM[s, 0] + tmp_w - H[s, 0]) + tmp_pi1 * (EM[s, 1] + tmp_w - H[s, 1])
                tmp_add *= c * dt
                H[s, 0] = (1 - dt) * H[s, 0] + dt * (EM[s, 0] + tmp_w) - tmp_add
                H[s, 1] = (1 - dt) * H[s, 1] + dt * (EM[s, 1] + tmp_w) - tmp_add

                # H[s, :] = (1 - dt) * H[s, :] + dt * (EM[s, :] + tmp_w) \
                #     - c * dt * np.ones(self.Na) * np.dot(tmp_pi, EM[s, :] + tmp_w - H[s, :])

            # print(2)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, 'fro') == 0 or np.linalg.norm(M, 'fro') == np.inf:
                # self.M = M
                print(M)
                return

            # err = np.linalg.norm(M - M0, 'fro') / np.linalg.norm(M, 'fro')
            # tmp_err = err
            # self.err_pd.append(err)
            err1 = np.linalg.norm(M - M0, 'fro') / np.linalg.norm(M, 'fro')
            err2 = np.linalg.norm(V - V0) / np.linalg.norm(V)
            tmp_err = max(err1, err2)
            self.err_pd.append(err1)
            for s in range(self.Ns):
                # if np.sum(M[s, :]) == 0:
                #     self.M = M
                #     print(M)
                #     return
                Z[s, :] = M[s, :] / np.sum(M[s, :])
            if self.resultZ is not None:
                self.err_pd_exact.append(np.linalg.norm(self.resultZ - Z, 'fro') / np.linalg.norm(self.resultZ, 'fro'))
            if self.V_exact is not None:
                self.err_pd_val_V.append(np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact))
            if self.pi_exact is not None:
                self.err_pd_val_pi.append(
                    np.linalg.norm(self.pi_exact - Z, 'fro') / np.linalg.norm(self.pi_exact, 'fro'))

            if cnt % 5 == 0:
                print(cnt)
                # print(CM)
                print(np.sum(M))
                print(err1)
                print(err2)
                # print(np.sum(Z > 0.05))
                print(np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact))
                print(np.max(np.abs(Z - self.pi_exact)))
                print(np.mean(np.abs(Z - self.pi_exact)))
                print(np.sum(np.abs(Z - self.pi_exact) > 0.1))

            if self.resultM is not None and self.resultV is not None:
                self.lyap.append(self._lyap_mixed(CV, V, M, c))

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultM = np.copy(M)
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))

    def get_mu(self, CV):
        if self.V_exact is None:
            self.value_iteration()
        pi = np.copy(self.pi_exact)
        # Construct P_pi
        cP0 = self.P0.tocoo()
        val = []
        for s, v in zip(cP0.row, cP0.data):
            val.append(v * pi[s, 0])
        P_pi_tmp0 = scipy.sparse.coo_matrix((val, (cP0.row, cP0.col)), shape=cP0.shape)

        cP1 = self.P1.tocoo()
        val = []
        for s, v in zip(cP1.row, cP1.data):
            val.append(v * pi[s, 1])
        P_pi_tmp1 = scipy.sparse.coo_matrix((val, (cP1.row, cP1.col)), shape=cP1.shape)

        P_pi = P_pi_tmp0 + P_pi_tmp1

        crds = [list(range(self.Ns)), list(range(self.Ns))]
        Idt = sparse.COO(crds, [1] * self.Ns, shape=(self.Ns, self.Ns))
        Idt = Idt.tocsr()

        w, _ = sla.bicgstab(Idt - self.gam * P_pi.T, self.V_exact, tol=1e-12)
        w *= CV
        # print((Idt - self.gam * P_pi.T).shape)
        # print(self.V_exact.shape)
        # print(w)

        mu = np.copy(self.pi_exact)
        for s in range(self.Ns):
            mu[s, :] *= w[s]
        self.mu_exact = np.copy(mu)

    def make_fig_pd(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.err_pd
        ax.tick_params(axis='both', labelsize=14)
        ax.plot(err0, label='primal dual', linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('relative change of mu', fontsize=22)
        ax.legend(loc='upper right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig_pd_ref(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.err_pd_val_V
        err1 = self.err_pd_val_pi
        ax.tick_params(axis='both', labelsize=14)
        ax.plot(err0, label='error of value', linewidth=linewidth)
        ax.plot(err1, label='error of policy', linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        # ax.set_ylabel('relative change of mu', fontsize=22)
        ax.legend(loc='upper right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig_pd_lyap(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.lyap
        ax.tick_params(axis='both', labelsize=14)
        ax.plot(err0, label='primal dual', linewidth=linewidth)
        ax.set_yscale('log')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('Lyapunov', fontsize=22)
        ax.legend(loc='upper right')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

    def make_fig_exact_pd(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis='both', labelsize=14)
        err0 = np.abs(np.log(self.err_pd_exact[:-1])) if self.err_pd_exact[-1] == 0 \
            else np.abs(np.log(self.err_pd_exact))
        ax.plot(err0, label='primal dual', linewidth=linewidth, marker='+')
        ax.set_xlabel('iterations', fontsize=22)
        ax.set_ylabel('log error', fontsize=22)
        ax.legend(loc='upper left')
        ax.set_title('Tau = ' + str(self.tau))
        fig.tight_layout()

# def load_var(var_name):
#     fid = open(os.path.join(var_name, '.pkl'))
#     data = pickle.load(fid)
#     fid.close()
#     return data


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


p = PolicyGradientNew()
# p.set_prior([1e-3, 1-1e-3])
p.set_tau(0.1)
p.gam = 0.9
p.value_iteration()
p.get_mu(CV=50000)
p.train_primal_dual_pseudo_mix(CV=50000, step=1e-3, c=0.9)
start_time = time.time()
p.train()
elapsed_time = time.time() - start_time
print(elapsed_time)
