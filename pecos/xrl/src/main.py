import numpy as np
import random
import scipy.sparse.linalg as sla
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.special import xlogy
from scipy.special import logsumexp
from collections import defaultdict

plt.rcParams.update({"font.size": 16})


class PolicyGradientNew:
    def __init__(
        self, tp=0, his=False, k=None, kernel_type="toy1", Na=50, Ns=200, gam=0.99, eps=1e-12
    ):
        if kernel_type == "toy3":
            Na = 2
        etp = ("KL", "rKL", "alpha")
        self.kernel_type = kernel_type
        self.tau = 1
        self.Na = Na
        self.Ns = Ns
        self.gam = gam
        self.eps = eps
        self.k = k if k else Ns // 10 + 1
        self.P = np.zeros([self.Na, self.Ns, self.Ns])
        self.alpha = None
        self.result = None
        self.ref_result = None
        self.resultZ = None
        self.resultM = None
        self.resultV = None
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
        if self.kernel_type == "toy1":
            idr = list(range(self.Ns))
            for i in range(self.Na):
                for j in range(self.Ns):
                    idx = random.sample(idr, self.k)
                    self.P[i, j, idx] = 1 / self.k
            Us = np.random.rand(Ns)
            Usa = np.random.rand(Ns, Na)
            self.r = np.zeros([Ns, Na])
            for i in range(Ns):
                for j in range(Na):
                    self.r[i, j] = Us[i] * Usa[i, j]

        if self.kernel_type == "toy2":
            P = np.random.uniform(0, 1, (self.Na, self.Ns, self.Ns))
            sparse = min(self.Ns - 1, int(self.Ns * 0.9))
            for i in range(self.Na):
                for j in range(self.Ns):
                    indices = np.random.choice(np.arange(self.Ns), replace=False, size=sparse)
                    P[i, j, indices] = 0
            P = P / P.sum(axis=2, keepdims=True)
            self.P[:] = P[:]
            Us = np.random.rand(Ns)
            Usa = np.random.rand(Ns, Na)
            self.r = np.zeros([Ns, Na])
            for i in range(Ns):
                for j in range(Na):
                    self.r[i, j] = Us[i] * Usa[i, j]

        if self.kernel_type == "toy3":
            self.P = np.zeros([self.Na, self.Ns, self.Ns])
            self.r = np.zeros([Ns, Na])
            for s in range(self.Ns):
                self.P[0, s, s] = 1 / 2
                self.P[0, s, (s + 1) % self.Ns] = 1 / 2
                self.P[1, s, s] = 1
                self.r[s, 0] = np.sin(4 * np.pi * s / self.Ns) + 1
                self.r[s, 1] = np.cos(4 * np.pi * s / self.Ns) + 1

        self.etp = etp[tp]
        if self.etp == "alpha":
            self.alpha = 0

    def train(self, eta=1, T=20):
        if self.etp == "KL":
            self.err[0] = []
        if self.etp == "rKL":
            self.err[1] = []
        if self.etp == "alpha":
            self.err[2][self.alpha] = []
        pi = (
            np.ones([self.Ns, self.Na]) / self.Na
            if not self.train_from_history
            else np.copy(self.result)
        )
        pi_new = np.zeros([self.Ns, self.Na])
        Idt = np.identity(self.Ns)
        if self.ref_result is not None:
            self.err_exact = []
        for itr in range(T):
            if self.ref_result is not None:
                self.err_exact.append(np.linalg.norm(self.ref_result - pi, "fro"))
                self.err_exact[-1] /= np.linalg.norm(self.ref_result, "fro")
            P_pi = np.zeros([self.Ns, self.Ns])
            for s in range(self.Ns):
                for t in range(self.Ns):
                    P_pi[s, t] = np.dot(pi[s, :], self.P[:, s, t])
            r_pi = np.zeros(self.Ns)
            for s in range(self.Ns):
                r_pi[s] = np.dot(pi[s, :], self.r[s, :])
            H_pi = 0
            if self.etp == "KL":
                H_pi = np.sum(xlogy(pi, pi), axis=1)
            if self.etp == "rKL":
                H_pi = np.sum(-np.log(pi), axis=1) / self.Na
            if self.etp == "alpha":
                H_pi = 4 / (1 - self.alpha ** 2) - 4 / (1 - self.alpha ** 2) * self.Na ** (
                    (self.alpha - 1) / 2
                ) * np.sum(pi ** ((self.alpha + 1) / 2), axis=1)
            v_pi = np.linalg.solve(Idt - self.gam * P_pi, r_pi - self.tau * H_pi)
            if self.etp == "KL":
                pi_new = pi ** (1 - eta)
                for a in range(self.Na):
                    pi_new[:, a] = pi_new[:, a] * np.exp(
                        eta * (self.r[:, a] - (Idt - self.gam * self.P[a]) @ v_pi) / self.tau
                    )

            if self.etp == "rKL":
                for a in range(self.Na):
                    tmp = self.P[a] @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) / pi[s, a]
                        pi_new[s, a] -= (
                            eta * self.Na * (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau
                        )

                for s in range(self.Ns):

                    def equation(x):
                        return np.sum(1 / (x + pi_new[s, :] + 1e-300)) - 1

                    low = max(self.Na - np.max(pi_new[s, :]), -np.min(pi_new[s, :]))
                    high = self.Na - np.min(pi_new[s, :])
                    cs = scipy.optimize.bisect(equation, low, high)
                    # cs = scipy.optimize.newton_krylov(equation, np.array([1]), f_tol=1e-12)
                    # print(np.abs(equation(cs)) < 1*10**-10)
                    for a in range(self.Na):
                        pi_new[s, a] = 1 / (pi_new[s, a] + cs)
                    # print(s)
                    # print(np.abs(np.sum(pi_new[s, :])-1) < 1*10**-10)
                    # print(np.min(pi_new[s, :]))

            if self.etp == "alpha":
                for a in range(self.Na):
                    tmp = self.P[a] @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) * (pi[s, a] ** ((self.alpha - 1) / 2))
                        tmp_sa = (1 - self.alpha) / 2 * eta * (self.Na ** ((1 - self.alpha) / 2))
                        pi_new[s, a] -= (
                            tmp_sa * (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau
                        )
                for s in range(self.Ns):
                    alpha = self.alpha

                    def equation(x):
                        return np.sum((x + pi_new[s, :] + 1e-300) ** (2 / (alpha - 1))) - 1

                    low = max(
                        self.Na ** ((1 - self.alpha) / 2) - np.max(pi_new[s, :]),
                        -np.min(pi_new[s, :]),
                    )
                    high = self.Na ** ((1 - self.alpha) / 2) - np.min(pi_new[s, :])
                    cs = scipy.optimize.bisect(equation, low, high)
                    for a in range(self.Na):
                        pi_new[s, a] = (pi_new[s, a] + cs) ** (2 / (self.alpha - 1))

            for s in range(self.Ns):
                pi_new[s, :] /= sum(pi_new[s, :])
            err = np.linalg.norm(pi_new - pi, "fro") / np.linalg.norm(pi, "fro")
            print(err)
            if self.etp == "KL":
                self.err[0].append(err)
            if self.etp == "rKL":
                self.err[1].append(err)
            if self.etp == "alpha":
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
        etp = ("KL", "rKL", "alpha")
        self.etp = etp[tp]
        if self.etp == "alpha":
            self.alpha = 0
        self.ref_result = None

    def get_value(self, pi=None, reg=False):
        # if self.result is None:
        #     return
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
            if self.etp == "KL":
                H_pi = np.sum(xlogy(pi, pi), axis=1)
            if self.etp == "rKL":
                H_pi = np.sum(-np.log(pi), axis=1) / self.Na
            if self.etp == "alpha":
                H_pi = 4 / (1 - self.alpha ** 2) - 4 / (1 - self.alpha ** 2) * self.Na ** (
                    (self.alpha - 1) / 2
                ) * np.sum(pi ** ((self.alpha + 1) / 2), axis=1)
        v_pi = np.linalg.solve(Idt - self.gam * P_pi, r_pi - self.tau * H_pi)
        return v_pi

    def get_expect_value(self, pi=None):
        # if self.result is None:
        #     return
        if pi is None:
            pi = self.result
        P_pi = np.zeros([self.Ns, self.Ns])
        for s in range(self.Ns):
            for t in range(self.Ns):
                P_pi[s, t] = np.dot(pi[s, :], self.P[:, s, t])
        r_pi = np.zeros(self.Ns)
        for s in range(self.Ns):
            r_pi[s] = np.dot(pi[s, :], self.r[s, :])
        _, evec = sla.eigs(P_pi.T, k=1, which="LM")
        d_pi = (evec / evec.sum()).real
        d_pi = d_pi.reshape(len(d_pi))
        return np.dot(d_pi, r_pi)

    def make_fig(self, alpha1, alpha2, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.err[0]
        err1 = self.err[1]
        err2 = self.err[2][alpha1]
        err3 = self.err[2][alpha2]
        ax.tick_params(axis="both", labelsize=14)
        ax.plot(err0, label="KL", linewidth=linewidth)
        ax.plot(err1, label="reversed KL", linewidth=linewidth)
        ax.plot(err2, label="Hellinger", linewidth=linewidth)
        ax.plot(err3, label="alpha divergence (alpha=" + str(alpha2) + ")", linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlabel("iterations", fontsize=22)
        ax.set_ylabel("relative change of policy", fontsize=22)
        ax.legend(loc="upper right")
        ax.set_title("Tau = " + str(self.tau))
        fig.tight_layout()

    def make_fig_exact(self, label="KL", linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis="both", labelsize=14)
        err0 = (
            np.log(np.abs(np.log(self.err_exact[:-1])))
            if self.err_exact[-1] == 0
            else np.log(np.abs(np.log(self.err_exact)))
        )
        ax.plot(err0, label=label, linewidth=linewidth, marker="+")
        ax.plot(
            [0, len(err0) - 1],
            [0, np.log(2) * (len(err0) - 1)],
            label="slope=log2",
            color="g",
            linestyle="dashed",
            linewidth=0.5,
        )
        ax.set_xlabel("iterations", fontsize=22)
        ax.set_ylabel("loglog error", fontsize=22)
        ax.legend(loc="upper left")
        ax.set_title("Tau = " + str(self.tau))
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

            if self.etp == "KL":
                V = logsumexp(Q / self.tau, axis=1) * self.tau
            if self.etp == "rKL":
                return
            if self.etp == "alpha":
                return
            if cnt % 100 == 0:
                print(np.linalg.norm(V - V_pre) / np.linalg.norm(V_pre))

        for a in range(self.Na):
            Q[:, a] = self.r[:, a] + self.gam * (self.P[a, :, :] @ V) - V
        self.V_exact = V
        self.pi_exact = softmax(Q / self.tau, axis=1)
        print(cnt)

    def train_primal_dual_old(self, CV=0.1, CW=0.1, tol=1e-6, step=0.01):
        V = np.zeros(self.Ns)
        W = np.ones(self.Ns)
        Z0 = np.zeros([self.Ns, self.Na])
        Z = np.ones([self.Ns, self.Na]) / self.Na
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0

        while np.max(np.abs(Z - Z0)) > tol:
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
                EZ[:, a] = (
                    self.r[:, a] - (Idt - self.gam * self.P[a, :, :]) @ V
                ) / self.tau - np.log(Z[:, a])
            Z0 = Z
            V = V - dt * EV
            W = W + dt * EW
            Z = Z * np.exp(dt * EZ)
            # print(np.min(W))
            for s in range(self.Ns):
                Z[s, :] /= sum(Z[s, :])
            if cnt % 10 == 0:
                print(cnt)
                print(np.max(np.abs(Z - Z0)))
        self.resultZ = np.copy(Z)
        print(cnt)
        print(np.max(np.abs(Z - Z0)))

    def train_primal_dual(self, CV=0.01, tol=1e-6, step=0.01):
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        while np.linalg.norm(M - M0) / np.linalg.norm(M) > tol:
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
                    H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a] + 1e-20) + 1
            H = (H + dt * EM) / (1 + dt)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)
            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def sample_gen(self, s, a, nb):
        samples = np.random.choice(self.Ns, size=nb, replace=True, p=self.P[a, s])
        return samples

    def train_primal_dual_batch(
        self, CV=0.01, tol=1e-5, step=1e-4, nb1=1, nb2=1
    ):  # primal dual method from samples
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        # Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        while np.linalg.norm(M - M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            s_idx = np.random.choice(self.Ns, replace=False, size=nb1)
            EV = np.sum(M, axis=1)
            for s in s_idx:
                for a in range(self.Na):
                    samples = self.sample_gen(s, a, nb2)
                    for sample in samples:
                        EV[sample] -= self.gam * M[s, a] / nb2
            EV /= CV
            V = (V + dt * EV) / (1 + dt)

            s_idx = np.random.choice(self.Ns, replace=False, size=nb1)
            # EM = np.zeros([self.Ns, self.Na])
            # for a in range(self.Na):
            #     EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in s_idx:
                for a in range(self.Na):
                    H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a] + 1e-300) + 1
                    samples = self.sample_gen(s, a, nb2)
                    tmp_v = 0
                    for sample in samples:
                        tmp_v += V[sample]
                    tmp_v /= nb2
                    H[s, a] = (H[s, a] + dt * ((self.r[s, a] - V[s] + self.gam * tmp_v) / CM)) / (
                        1 + dt
                    )
            # H = (H + dt * EM) / (1 + dt)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)
            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def train_primal_dual_batch_acc(
        self, CV=0.01, tol=1e-5, step=1e-4, nb1=1, nb2=1, lam=0.01, m=5
    ):  # "acc" = anderson acceleration
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        V_h = []
        GV_h = []
        GH_h = []
        H_h = []
        EV = 1

        while np.linalg.norm(M - M0) / np.linalg.norm(M) > tol or np.linalg.norm(EV) > tol:
            M0[:] = M[:]
            cnt += 1
            s_idx = None
            if cnt == 1:
                s_idx = np.random.choice(self.Ns, replace=False, size=nb1)
                EV = np.sum(M, axis=1)
                for s in s_idx:
                    for a in range(self.Na):
                        samples = self.sample_gen(s, a, nb2)
                        for sample in samples:
                            EV[sample] -= self.gam * M[s, a] / nb2
                EV /= CV
                # print(np.linalg.norm(EV))
                GV_h.append((V + dt * EV) / (1 + dt) - V)
                V = (V + dt * EV) / (1 + dt)
                V_h.append(V)

            if 1 < cnt:
                s_idx = np.random.choice(self.Ns, replace=False, size=nb1)
                EV = np.sum(M, axis=1)
                for s in s_idx:
                    for a in range(self.Na):
                        samples = self.sample_gen(s, a, nb2)
                        for sample in samples:
                            EV[sample] -= self.gam * M[s, a] / nb2
                EV /= CV
                # print(np.linalg.norm(EV))
                GV_h.append((V + dt * EV) / (1 + dt) - V)
                V = (V + dt * EV) / (1 + dt)
                V_h.append(V)
                U = np.array(GV_h).T
                c = np.linalg.solve(
                    U.T @ U + lam * np.identity(np.size(U, 1)), np.ones(np.size(U, 1))
                )
                c = c / np.sum(c)
                V = c @ V_h
                if cnt > m:
                    GV_h = GV_h[1:]
                    V_h = V_h[1:]

            # s_idx = np.random.choice(self.Ns, replace=False, size=nb1)
            H = None
            gH = None
            if cnt == 1:
                H = np.zeros([self.Ns, self.Na])
                for s in range(self.Ns):
                    for a in range(self.Na):
                        H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a] + 1e-300) + 1

                gH = np.zeros([self.Ns, self.Na])
                for s in s_idx:
                    for a in range(self.Na):
                        samples = self.sample_gen(s, a, nb2)
                        tmp_v = 0
                        for sample in samples:
                            tmp_v += V[sample]
                        tmp_v /= nb2
                        gH[s, a] = (
                            H[s, a] + dt * ((self.r[s, a] - V[s] + self.gam * tmp_v) / CM)
                        ) / (1 + dt) - H[s, a]
                GH_h.append(gH.reshape(-1))
                H = H + gH
                H_h.append(H.reshape(-1))

            # H = (H + dt * EM) / (1 + dt)

            if 1 < cnt:
                H = np.zeros([self.Ns, self.Na])
                for s in range(self.Ns):
                    for a in range(self.Na):
                        H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a] + 1e-300) + 1

                gH = np.zeros([self.Ns, self.Na])
                for s in s_idx:
                    for a in range(self.Na):
                        samples = self.sample_gen(s, a, nb2)
                        tmp_v = 0
                        for sample in samples:
                            tmp_v += V[sample]
                        tmp_v /= nb2
                        gH[s, a] = (
                            H[s, a] + dt * ((self.r[s, a] - V[s] + self.gam * tmp_v) / CM)
                        ) / (1 + dt) - H[s, a]
                GH_h.append(gH.reshape(-1))
                H = H + gH
                H_h.append(H.reshape(-1))

                U = np.array(GH_h).T
                c = np.linalg.solve(
                    U.T @ U + lam * np.identity(np.size(U, 1)), np.ones(np.size(U, 1))
                )
                c = c / np.sum(c)
                H = c @ H_h
                if cnt > m:
                    GH_h = GH_h[1:]
                    H_h = H_h[1:]

            H = H.reshape([self.Ns, -1])
            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)
            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
                print(np.linalg.norm(EV))
                print(np.linalg.norm(gH))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def train_primal_dual_new(
        self, CV=0.01, tol=1e-6, step=0.01
    ):  # with \mu\log\hat{\mu} regularization and a different approximate Hessian
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        while np.linalg.norm(M - M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                for a in range(self.Na):
                    # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                    H[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
                print(np.sum(M > 0.05))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        # self.M = M
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def train_primal_dual_new_continue(self, CV=0.01, tol=1e-6, step=3e-4):
        CM = self.tau if self.tau >= 0.01 else 2 ** (int(np.log2(0.01 / self.tau)) + 1) * self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        # err_tmp = np.linalg.norm(M-M0) / np.linalg.norm(M)

        while CM > self.tau or np.linalg.norm(M - M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                for a in range(self.Na):
                    # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                    H[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
                print(np.sum(M > 0.05))

            if np.linalg.norm(M - M0) / np.linalg.norm(M) <= tol:
                CM /= 2
                dt /= 2

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        # self.M = M
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def train_primal_dual_new_continue_acc(self, CV=0.01, tol=1e-6, step=3e-4, lam=0.001, m=5):
        CM = self.tau if self.tau >= 0.01 else 2 ** (int(np.log2(0.01 / self.tau)) + 1) * self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        V_h = []
        GV_h = []
        H_h = []
        GH_h = []

        # err_tmp = np.linalg.norm(M-M0) / np.linalg.norm(M)
        cc = 100

        while CM > self.tau or np.linalg.norm(M - M0) / np.linalg.norm(M) > tol * cc:
            if CM == self.tau:
                cc = 1
            M0[:] = M[:]
            cnt += 1

            if cnt == 1:
                EV = np.zeros(self.Ns)
                for a in range(self.Na):
                    EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
                EV /= CV
                GV_h.append((1 - dt) * V + dt * EV - V)
                V = (1 - dt) * V + dt * EV
                V_h.append(V)

            if cnt > 1:
                EV = np.zeros(self.Ns)
                for a in range(self.Na):
                    EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
                EV /= CV
                GV_h.append((1 - dt) * V + dt * EV - V)
                V = (1 - dt) * V + dt * EV
                V_h.append(V)
                U = np.array(GV_h).T
                c = np.linalg.solve(
                    U.T @ U + lam * np.identity(np.size(U, 1)), np.ones(np.size(U, 1))
                )
                c = c / np.sum(c)
                V = c @ V_h
                # if np.min(c) < 0:
                #     self.U = U
                #     print(cnt)
                #     print("!")
                #     print(c)
                #     return
                if cnt > m:
                    GV_h = GV_h[1:]
                    V_h = V_h[1:]

            H = None
            if cnt == 1:
                H = np.zeros([self.Ns, self.Na])
                for s in range(self.Ns):
                    for a in range(self.Na):
                        H[s, a] = np.log(M[s, a] + 1e-300)

                EM = np.zeros([self.Ns, self.Na])
                for a in range(self.Na):
                    EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

                gH = np.zeros([self.Ns, self.Na])

                for s in range(self.Ns):
                    tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                    for a in range(self.Na):
                        # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                        gH[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w) - H[s, a]
                        H[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

                GH_h.append(gH.reshape(-1))
                H_h.append(H.reshape(-1))

            if cnt > 1:
                H = np.zeros([self.Ns, self.Na])
                for s in range(self.Ns):
                    for a in range(self.Na):
                        H[s, a] = np.log(M[s, a] + 1e-300)

                EM = np.zeros([self.Ns, self.Na])
                for a in range(self.Na):
                    EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

                gH = np.zeros([self.Ns, self.Na])

                for s in range(self.Ns):
                    tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                    for a in range(self.Na):
                        # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                        gH[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w) - H[s, a]
                        H[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

                GH_h.append(gH.reshape(-1))
                H_h.append(H.reshape(-1))
                U = np.array(GH_h).T
                c = np.linalg.solve(
                    U.T @ U + lam * np.identity(np.size(U, 1)), np.ones(np.size(U, 1))
                )
                c = c / np.sum(c)

                H = c @ H_h
                if cnt > m:
                    GH_h = GH_h[1:]
                    H_h = H_h[1:]

            H = H.reshape([self.Ns, -1])

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(cnt)
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
                print(np.sum(M > 0.05))

            if np.linalg.norm(M - M0) / np.linalg.norm(M) <= tol:
                CM /= 2
                dt /= 2

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        # self.M = M
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def train_primal_dual_epart(
        self, tol=1e-6, step=0.01
    ):  # gradient descent for v, natural gradient descent for mu
        CM = self.tau
        V = np.ones(self.Ns)
        vec_e = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        while np.linalg.norm(M - M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            V = V - dt * (vec_e - EV)

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                for a in range(self.Na):
                    # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                    H[s, a] = (1 - dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
                print(np.sum(M > 0.05))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        # self.M = M
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def train_primal_dual_e(self, tol=1e-6, step=0.01):  # vanilla gradient descent
        CM = self.tau
        V = np.ones(self.Ns)
        vec_e = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        while np.linalg.norm(M - M0) / np.linalg.norm(M) > tol:
            M0[:] = M[:]
            cnt += 1

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            V = V - dt * (vec_e - EV)

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = self.r[:, a] - (Idt - self.gam * self.P[a]) @ V

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                for a in range(self.Na):
                    # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                    M[s, a] = M[s, a] + dt * (EM[s, a] + CM * tmp_w - CM * np.log(M[s, a]))

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf or np.min(M) < 0:
                # self.M = M
                print(M)
                print(np.min(M))
                return

            err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            self.err_pd.append(err)
            if self.resultZ is not None:
                for s in range(self.Ns):
                    # if np.sum(M[s, :]) == 0:
                    #     self.M = M
                    #     print(M)
                    #     return
                    Z[s, :] = M[s, :] / np.sum(M[s, :])
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(err)
                print(np.sum(M > 0.05))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        # self.M = M
        # print("M")
        # print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))  # step needs to be tuned

    def _lyap(self, CV, V, M):
        L1 = CV / 2 * (np.linalg.norm(V - self.resultV)) ** 2
        L21 = self.tau * np.sum(xlogy(self.resultM, self.resultM))
        L22 = self.tau * np.sum(xlogy(self.resultM, M)) + self.tau * np.sum(M - self.resultM)
        return L1 + L21 - L22

    def _lyap_mixed(self, CV, V, M, c):
        tmp_b = np.sum(self.resultM, axis=1)
        tmp_M = np.sum(M, axis=1)
        L1 = CV / 2 * (np.linalg.norm(V - self.resultV)) ** 2
        L21 = self.tau * np.sum(xlogy(self.resultM, self.resultM))
        L22 = self.tau * np.sum(xlogy(self.resultM, M))
        L22 += self.tau * c / (1 - c) * np.sum(xlogy(tmp_b, tmp_b))
        L23 = self.tau * c / (1 - c) * np.sum(xlogy(tmp_b, tmp_M))
        L23 += self.tau / (1 - c) * np.sum(M - self.resultM)
        return L1 + L21 - L22 - L23

    def train_primal_dual_pseudo_mix(
        self, CV=0.01, tol=1e-6, step=0.01, c=0.99
    ):  # with \mu\log\hat{\mu} regularization and a different approximate Hessian
        CM = self.tau
        V = np.ones(self.Ns)
        V0 = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        Idt = np.identity(self.Ns)
        cnt = 0
        dt = step
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        self.err_pd_val_V = []
        self.err_pd_val_pi = []
        self.lyap = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        tmp_err = 1
        while tmp_err > tol:
            M0[:] = M[:]
            V0[:] = V[:]
            cnt += 1
            # if tmp_err < 2 * tol:
            #     c = 0.98

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                tmp_pi = softmax(H[s, :])
                # for a in range(self.Na):
                #     # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                #     H[s, a] = (1-dt) * H[s, a] + dt * (EM[s, a] + tmp_w)
                tmp_h1 = (1 - dt) * H[s, :] + dt * (EM[s, :] + tmp_w)
                tmp_h2 = c * dt * np.ones(self.Na) * np.dot(tmp_pi, EM[s, :] + tmp_w - H[s, :])

                H[s, :] = tmp_h1 - tmp_h2

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            # err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            # tmp_err = err
            err1 = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
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
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                # print(CM)
                print(np.sum(M))
                print(err1)
                print(err2)
                print(np.sum(Z > 0.05))
                print(np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact))
                print(np.max(np.abs(Z - self.pi_exact)))
                print(np.mean(np.abs(Z - self.pi_exact)))

            if self.resultM is not None and self.resultV is not None:
                self.lyap.append(self._lyap_mixed(CV, V, M, c))
            if self.V_exact is not None:
                self.err_pd_val_V.append(
                    np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact)
                )
            if self.pi_exact is not None:
                self.err_pd_val_pi.append(
                    np.linalg.norm(self.pi_exact - Z, "fro") / np.linalg.norm(self.pi_exact, "fro")
                )

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultM = np.copy(M)
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))

    def train_primal_dual_pseudo_mix_random_grad(
        self, CV=0.01, tol=1e-4, step=0.01, c=0.99, sigma=0.01
    ):  # with \mu\log\hat{\mu} regularization and a different approximate Hessian
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        V0 = np.ones(self.Ns)
        Idt = np.identity(self.Ns)
        cnt = 0
        dt = step
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        self.lyap = []
        self.err_pd_val_V = []
        self.err_pd_val_pi = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        tmp_err = 1
        while tmp_err > tol:
            M0[:] = M[:]
            V0[:] = V[:]
            cnt += 1
            # if tmp_err < 2 * tol:
            #     c = 0.98

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a] + np.random.normal(
                    loc=0.0, scale=sigma, size=self.Ns
                )
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (
                    self.r[:, a] - (Idt - self.gam * self.P[a]) @ V
                ) / CM + np.random.normal(loc=0.0, scale=sigma, size=self.Ns)

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                tmp_pi = softmax(H[s, :])
                # for a in range(self.Na):
                #     # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                #     H[s, a] = (1-dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

                tmp_h1 = (1 - dt) * H[s, :] + dt * (EM[s, :] + tmp_w)
                tmp_h2 = c * dt * np.ones(self.Na) * np.dot(tmp_pi, EM[s, :] + tmp_w - H[s, :])
                H[s, :] = tmp_h1 - tmp_h2

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err1 = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
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
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if self.V_exact is not None:
                self.err_pd_val_V.append(
                    np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact)
                )
            if self.pi_exact is not None:
                self.err_pd_val_pi.append(
                    np.linalg.norm(self.pi_exact - Z, "fro") / np.linalg.norm(self.pi_exact, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                # print(CM)
                print(np.sum(M))
                print(err1)
                print(err2)
                print(np.sum(Z > 0.05))
                print(np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact))
                print(np.max(np.abs(Z - self.pi_exact)))
                print(np.mean(np.abs(Z - self.pi_exact)))

            if self.resultM is not None and self.resultV is not None:
                self.lyap.append(self._lyap_mixed(CV, V, M, c))

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultM = np.copy(M)
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))

    def train_primal_dual_pseudo_mix_sample(
        self, CV=0.01, tol=1e-4, step=0.01, c=0.99, n_sam=100, n_batch=1000
    ):  # with \mu\log\hat{\mu} regularization and a different approximate Hessian
        CM = self.tau
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        V0 = np.ones(self.Ns)
        # Idt = np.identity(self.Ns)
        cnt = 0
        dt = step
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        self.lyap = []
        if self.resultZ is not None:
            self.err_pd_exact = []
        if self.V_exact is not None:
            self.err_pd_val_V = []
        if self.pi_exact is not None:
            self.err_pd_val_pi = []

        samples_buffer = []
        for s in range(self.Ns):
            for a in range(self.Na):
                # samples_buffer[(s, a)] = []
                samples = self.sample_gen(s, a, n_sam)
                for sample in samples:
                    samples_buffer.append((s, a, sample))
        print("sampling finished")
        print("buffer size: " + str(len(samples_buffer)))

        tmp_err = 1
        while tmp_err > tol:
            M0[:] = M[:]
            V0[:] = V[:]
            cnt += 1
            # if tmp_err < 2 * tol:
            #     c = 0.98
            samples = random.sample(samples_buffer, n_batch)

            ctr = defaultdict(lambda: 0)
            for (s, a, _) in samples:
                ctr[(s, a)] += 1

            # s_idx = np.random.choice(len(samples_buffer), replace=False, size=n_batch)
            EV = np.sum(M, axis=1)
            for s, a, sample in samples:
                EV[sample] -= self.gam * M[s, a] / ctr[(s, a)]
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                # EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM
                EM[:, a] = (self.r[:, a] - V) / CM

            for s, a, sample in samples:
                EM[s, a] += self.gam / CM * V[sample] / ctr[(s, a)]

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                tmp_pi = softmax(H[s, :])
                # for a in range(self.Na):
                #     # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                #     H[s, a] = (1-dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

                tmp_h1 = (1 - dt) * H[s, :] + dt * (EM[s, :] + tmp_w)
                tmp_h2 = c * dt * np.ones(self.Na) * np.dot(tmp_pi, EM[s, :] + tmp_w - H[s, :])

                H[s, :] = tmp_h1 - tmp_h2

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            # err = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            err1 = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
            err2 = np.linalg.norm(V - V0) / np.linalg.norm(V)
            tmp_err = max(err1, err2)
            # tmp_err = err
            self.err_pd.append(err1)
            for s in range(self.Ns):
                # if np.sum(M[s, :]) == 0:
                #     self.M = M
                #     print(M)
                #     return
                Z[s, :] = M[s, :] / np.sum(M[s, :])
            if self.resultZ is not None:
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )
            if self.V_exact is not None:
                self.err_pd_val_V.append(
                    np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact)
                )
            if self.pi_exact is not None:
                self.err_pd_val_pi.append(
                    np.linalg.norm(self.pi_exact - Z, "fro") / np.linalg.norm(self.pi_exact, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                # print(CM)
                print(np.sum(M))
                print(err1)
                print(err2)
                print(np.sum(Z > 0.05))
                print(np.linalg.norm(V - self.V_exact) / np.linalg.norm(self.V_exact))
                print(np.max(np.abs(Z - self.pi_exact)))
                print(np.mean(np.abs(Z - self.pi_exact)))

            if self.resultM is not None and self.resultV is not None:
                self.lyap.append(self._lyap_mixed(CV, V, M, c))

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultM = np.copy(M)
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))

    def train_primal_dual_pseudo_mix_continue(
        self, CV=0.1, tol=1e-6, step=0.01, c=0.99
    ):  # with \mu\log\hat{\mu} regularization and a different approximate Hessian
        # CM = self.tau
        CM_start = 0.01
        CM = (
            self.tau
            if self.tau >= CM_start
            else 2 ** (int(np.log2(CM_start / self.tau)) + 1) * self.tau
        )
        V = np.ones(self.Ns)
        V0 = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        Idt = np.identity(self.Ns)
        cnt = 0
        dt = step
        Z = np.zeros([self.Ns, self.Na])
        self.err_pd = []
        self.lyap = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        tmp_err = 1
        # mode_cnt = 0
        while tmp_err > tol or CM > self.tau:
            M0[:] = M[:]
            V0[:] = V[:]
            cnt += 1
            # if tmp_err < 2 * tol:
            #     c = 0.98

            EV = np.zeros(self.Ns)
            for a in range(self.Na):
                EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
            EV /= CV
            V = (1 - dt) * V + dt * EV

            EM = np.zeros([self.Ns, self.Na])
            for a in range(self.Na):
                EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

            H = np.zeros([self.Ns, self.Na])
            for s in range(self.Ns):
                for a in range(self.Na):
                    H[s, a] = np.log(M[s, a] + 1e-300)

            for s in range(self.Ns):
                tmp_w = np.log(np.sum(M[s, :]) + 1e-300)
                tmp_pi = softmax(H[s, :])
                # for a in range(self.Na):
                #     # H[s, a] = (H[s, a] + dt * EM[s, a] + dt * tmp_w) / (1 + dt)
                #     H[s, a] = (1-dt) * H[s, a] + dt * (EM[s, a] + tmp_w)

                tmp_h1 = (1 - dt) * H[s, :] + dt * (EM[s, :] + tmp_w)
                tmp_h2 = c * dt * np.ones(self.Na) * np.dot(tmp_pi, EM[s, :] + tmp_w - H[s, :])

                H[s, :] = tmp_h1 - tmp_h2

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = np.exp(H[s, a])

            if np.linalg.norm(M, "fro") == 0 or np.linalg.norm(M, "fro") == np.inf:
                # self.M = M
                print(M)
                return

            err1 = np.linalg.norm(M - M0, "fro") / np.linalg.norm(M, "fro")
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
                self.err_pd_exact.append(
                    np.linalg.norm(self.resultZ - Z, "fro") / np.linalg.norm(self.resultZ, "fro")
                )

            if cnt % 50 == 0:
                print(cnt)
                print(CM)
                print(np.sum(M))
                print(err1)
                print(err2)
                print(np.sum(Z > 0.05))

            if self.resultM is not None and self.resultV is not None:
                self.lyap.append(self._lyap_mixed(CV, V, M, c))

            if CM > self.tau and tmp_err <= 400 * tol:
                CM /= 2
                dt /= 2

            # mode_cnt = np.sum(Z > 0.05)

        for s in range(self.Ns):
            Z[s, :] = M[s, :] / np.sum(M[s, :])
        self.resultM = np.copy(M)
        self.resultZ = np.copy(Z)
        self.resultV = np.copy(V)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))

    def _f(self, CV, CM, V, H):
        Idt = np.identity(self.Ns)
        M = np.zeros([self.Ns, self.Na])
        for s in range(self.Ns):
            for a in range(self.Na):
                M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)

        V_r = np.zeros(self.Ns)
        for a in range(self.Na):
            V_r += (Idt - self.gam * self.P[a].T) @ M[:, a]
        V_r /= CV

        H_r = np.zeros([self.Ns, self.Na])
        for a in range(self.Na):
            H_r[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

        return V_r, H_r, V_r - V, H_r - H

    def _f_step(self, CV, CM, V, H, step):
        Idt = np.identity(self.Ns)
        M = np.zeros([self.Ns, self.Na])
        for s in range(self.Ns):
            for a in range(self.Na):
                M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)

        dt = step
        self.err_pd = []
        if self.resultZ is not None:
            self.err_pd_exact = []

        EV = np.zeros(self.Ns)
        for a in range(self.Na):
            EV += (Idt - self.gam * self.P[a].T) @ M[:, a]
        EV /= CV
        V_r = (V + dt * EV) / (1 + dt)

        EM = np.zeros([self.Ns, self.Na])
        for a in range(self.Na):
            EM[:, a] = (self.r[:, a] - (Idt - self.gam * self.P[a]) @ V) / CM

        H_r = (H + dt * EM) / (1 + dt)

        return V_r, H_r, V_r - V, H_r - H

    def make_fig_pd(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.err_pd
        ax.tick_params(axis="both", labelsize=14)
        ax.plot(err0, label="primal dual", linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlabel("iterations", fontsize=22)
        ax.set_ylabel("relative change of mu", fontsize=22)
        ax.legend(loc="upper right")
        ax.set_title("Tau = " + str(self.tau))
        fig.tight_layout()

    def make_fig_pd_ref(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.err_pd_val_V
        err1 = self.err_pd_val_pi
        ax.tick_params(axis="both", labelsize=14)
        ax.plot(err0, label="error of value", linewidth=linewidth)
        ax.plot(err1, label="error of policy", linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlabel("iterations", fontsize=22)
        # ax.set_ylabel("relative change of mu", fontsize=22)
        ax.legend(loc="upper right")
        ax.set_title("Tau = " + str(self.tau))
        fig.tight_layout()

    def make_fig_pd_lyap(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        err0 = self.lyap
        ax.tick_params(axis="both", labelsize=14)
        ax.plot(err0, label="primal dual", linewidth=linewidth)
        ax.set_yscale("log")
        ax.set_xlabel("iterations", fontsize=22)
        ax.set_ylabel("Lyapunov", fontsize=22)
        ax.legend(loc="upper right")
        ax.set_title("Tau = " + str(self.tau))
        fig.tight_layout()

    def make_fig_exact_pd(self, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.tick_params(axis="both", labelsize=14)
        err0 = (
            np.abs(np.log(self.err_pd_exact[:-1]))
            if self.err_pd_exact[-1] == 0
            else np.abs(np.log(self.err_pd_exact))
        )
        ax.plot(err0, label="primal dual", linewidth=linewidth, marker="+")
        ax.set_xlabel("iterations", fontsize=22)
        ax.set_ylabel("log error", fontsize=22)
        ax.legend(loc="upper left")
        ax.set_title("Tau = " + str(self.tau))
        fig.tight_layout()


p = PolicyGradientNew(Ns=200, Na=50, k=20, gam=0.9)
p.set_tau(0.01)
p.train_primal_dual(CV=0.01, step=2e-3)
# p.train_primal_dual_pseudo(CV=0.01, step=2e-2)
p.train_primal_dual_pseudo_mix(CV=0.1, step=1e-2, c=0.9)
p.train_primal_dual_pseudo_mix_random_grad(CV=0.1, step=1e-2, c=0.9, sigma=0.01)
p.train_primal_dual_new(CV=0.01, step=4e-4)

p = PolicyGradientNew(Ns=20, Na=5, k=3, gam=0.9)
p.set_tau(0.01)
# p.train_primal_dual_pseudo_mix_sample(CV=0.1, step=1e-3, tol=1e-3, c=0.9, n_sam=10000, n_batch=2000)
p.train_primal_dual_pseudo_mix_sample(CV=0.1, step=1e-3, tol=1e-3, c=0.9, n_sam=4000, n_batch=4000)

# p = PolicyGradientNew(Ns=50, Na=5, k=3, gam=0.9)
# p.set_tau(0.01)
# p.train_primal_dual_pseudo_mix_sample(CV=0.1, step=1e-3, tol=1e-3, c=0.9, n_sam=10000, n_batch=2000)

p = PolicyGradientNew(Ns=100, Na=20, k=8, gam=0.9)
p.set_tau(0.1)
# p.train_primal_dual_pseudo_mix_sample(CV=0.1, step=1e-3, tol=1e-3, c=0.9, n_sam=10000, n_batch=20000)
p.train_primal_dual_pseudo_mix_sample(CV=0.1, step=1e-3, tol=1e-3, c=0.9, n_sam=10000, n_batch=8000)

p = PolicyGradientNew(Ns=200, Na=50, k=20, gam=0.9)
p.set_tau(0.1)
p.value_iteration()
p.train_primal_dual_pseudo_mix(CV=0.1, step=8e-3, tol=1e-5, c=0.9)
p.train_primal_dual_pseudo_mix_sample(
    CV=0.1, step=1e-3, tol=1e-3, c=0.9, n_sam=10000, n_batch=100000
)

p = PolicyGradientNew(Ns=200, Na=50, k=20, gam=0.9)
p.set_tau(0.1)
p.value_iteration()
p.train_primal_dual_pseudo_mix(CV=0.1, step=8e-3, tol=1e-5, c=0.9)
p.train_primal_dual_pseudo_mix_random_grad(CV=0.1, tol=1e-3, step=0.01, c=0.9, sigma=0.05)

p.set_tau(0.001)
# p.train_primal_dual_pseudo(CV=0.01, step=2e-3)
p.train_primal_dual_pseudo_mix(CV=0.1, step=2e-3, c=0.993)
p.train_primal_dual_new_continue()
p.train_primal_dual_new_continue_acc(lam=0)
V_old = np.copy(p.resultV)
# Pi_old = np.copy(p.resultZ)
p.train_primal_dual_new(CV=0.01, step=3e-4)
p.train_primal_dual_e(step=3e-4)
# V = np.copy(p.resultV)
# Pi = np.copy(p.resultZ)
p.make_fig_pd()
# np.savetxt("Pi_old.txt", Pi_old, fmt="%s")
# np.savetxt("V_old.txt", V_old, fmt="%s")
plt.savefig("conv.png")
Pi = np.loadtxt("Pi_new_compare.txt")
Pi_old = np.loadtxt("Pi_old_compare.txt")
# with open("model.pkl", "wb") as outp:
#     pickle.dump(p, outp, pickle.HIGHEST_PROTOCOL)
#
# p = pickle.load(open("model1.pkl", "rb"))
