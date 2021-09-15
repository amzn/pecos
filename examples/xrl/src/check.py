import numpy as np
import random
import scipy.sparse.linalg as sla
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.special import xlogy
from scipy.special import logsumexp

# import pickle

plt.rcParams.update({"font.size": 16})


class PolicyGradientCheck:
    def __init__(
        self, tp=0, his=False, k=None, kernel_type="toy1", Na=50, Ns=200, gam=0.99, eps=1e-12
    ):
        if kernel_type == "toy3":
            Na = 2
        if kernel_type == "toy4":
            Na = 2
        if kernel_type == "toy5":
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
        self.resultZ = np.zeros([self.Ns, self.Na])
        self.train_from_history = his
        self.err = [[], [], {}]
        self.err_exact = []
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

        if self.kernel_type == "toy4":
            self.P = np.zeros([self.Na, self.Ns, self.Ns])
            self.r = np.zeros([Ns, Na])
            self.P[0, 0, 0] = 1
            self.P[1, 0, 1] = 1
            for s in range(1, self.Ns - 1):
                self.P[0, s, s - 1] = 1  # go left
                self.P[1, s, s + 1] = 1  # go right
            self.P[0, self.Ns - 1, self.Ns - 1] = 1
            self.P[1, self.Ns - 1, self.Ns - 1] = 1
            self.r[self.Ns - 2, 1] = 1

        if self.kernel_type == "toy5":
            self.P = np.zeros([self.Na, self.Ns, self.Ns])
            self.r = np.zeros([Ns, Na])
            self.P[0, 0, 0] = 1
            self.P[1, 0, 1] = 1
            for s in range(1, self.Ns - 1):
                self.P[0, s, s - 1] = 1  # go left
                self.P[1, s, s + 1] = 1  # go right
            self.P[0, self.Ns - 1, self.Ns - 2] = 1
            self.P[1, self.Ns - 1, self.Ns - 1] = 0
            self.r[self.Ns - 1, 1] = 1

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
                tmp_h1 = 4 / (1 - self.alpha ** 2)
                tmp_h2 = 4 / (1 - self.alpha ** 2) * self.Na ** ((self.alpha - 1) / 2)
                H_pi = tmp_h1 - tmp_h2 * np.sum(pi ** ((self.alpha + 1) / 2), axis=1)
            v_pi = np.linalg.solve(Idt - self.gam * P_pi, r_pi - self.tau * H_pi)
            # print(np.mean(v_pi))
            # print(np.linalg.norm((Idt-gam * P_pi) @ v_pi - r_pi))
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
                    cs = low
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

            if self.etp == "alpha":
                for a in range(self.Na):
                    tmp = self.P[a] @ v_pi
                    for s in range(self.Ns):
                        pi_new[s, a] = (1 - eta) * (pi[s, a] ** ((self.alpha - 1) / 2))
                        tmp_pi1 = (1 - self.alpha) / 2 * eta * (self.Na ** ((1 - self.alpha) / 2))
                        tmp_pi2 = (self.r[s, a] - v_pi[s] + self.gam * tmp[s]) / self.tau
                        pi_new[s, a] -= tmp_pi1 * tmp_pi2

                for s in range(self.Ns):
                    alpha = self.alpha

                    def equation(x):
                        return np.sum((x + pi_new[s, :] + 1e-300) ** (2 / (alpha - 1))) - 1

                    low = max(
                        self.Na ** ((1 - self.alpha) / 2) - np.max(pi_new[s, :]),
                        -np.min(pi_new[s, :]),
                    )
                    high = self.Na ** ((1 - self.alpha) / 2) - np.min(pi_new[s, :])
                    cs = low
                    if np.abs(equation(high)) < 1e-15:
                        cs = high
                    if np.abs(equation(low)) >= 1e-15 and np.abs(equation(high)) >= 1e-15:
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
            if self.etp == "KL":
                H_pi = np.sum(xlogy(pi, pi), axis=1)
            if self.etp == "rKL":
                H_pi = np.sum(-np.log(pi), axis=1) / self.Na
            if self.etp == "alpha":
                tmp_h1 = 4 / (1 - self.alpha ** 2)
                tmp_h2 = 4 / (1 - self.alpha ** 2) * self.Na ** ((self.alpha - 1) / 2)
                H_pi = tmp_h1 - tmp_h2 * np.sum(pi ** ((self.alpha + 1) / 2), axis=1)
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
        _, evec = sla.eigs(P_pi.T, k=1, which="LM")
        d_pi = (evec / evec.sum()).real
        d_pi = d_pi.reshape(len(d_pi))
        return np.dot(d_pi, r_pi)

    def make_fig(self, alpha1, alpha2, linewidth=0.5):
        fig, ax = plt.subplots(figsize=(12, 8))
        # err0 = self.err[0]
        err1 = self.err[1]
        err2 = self.err[2][alpha1]
        err3 = self.err[2][alpha2]
        ax.tick_params(axis="both", labelsize=14)
        # ax.plot(err0, label="KL", linewidth=linewidth)
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
        self.resultZ[:] = Z[:]
        print(cnt)
        print(np.max(np.abs(Z - Z0)))

    def train_primal_dual(self, CV=0.01, CM=0.01, tol=1e-8, step=0.01):
        V = np.ones(self.Ns)
        M = np.ones([self.Ns, self.Na]) / self.Na
        M0 = np.zeros([self.Ns, self.Na])
        dt = step
        Idt = np.identity(self.Ns)
        cnt = 0
        Z = np.zeros([self.Ns, self.Na])

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
                    H[s, a] = M[s, a] if M[s, a] >= 1 else np.log(M[s, a]) + 1
            H = (H + dt * EM) / (1 + dt)

            for s in range(self.Ns):
                for a in range(self.Na):
                    M[s, a] = H[s, a] if H[s, a] >= 1 else np.exp(H[s, a] - 1)

            if cnt % 50 == 0:
                print(cnt)
                print(np.linalg.norm(M - M0) / np.linalg.norm(M))
        for s in range(self.Ns):
            Z[s, :] = M[s, :] / sum(M[s, :])
        self.resultZ[:] = Z[:]
        print(M)
        print(cnt)
        print(np.linalg.norm(M - M0) / np.linalg.norm(M))


p = PolicyGradientCheck(Ns=1000, kernel_type="toy5")
p.set_tau(0.001)

# with open("model.pkl", "wb") as outp:
#     pickle.dump(p, outp, pickle.HIGHEST_PROTOCOL)
#
# p = pickle.load(open("model1.pkl", "rb"))
