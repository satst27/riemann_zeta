from mpmath import *
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import pandas as pd


# consult arxiv paper for h0, h1 and h2 series and integrals

class RiemannZeta:
    def __init__(self, s):
        self.s = s
        self.i = mpc('0', '1')
        self.alpha = mpf('1.0')  # sandeep
        self.residue_factor = mpf(
            '1.0')  # if set to 1 it takes residue contributions into account. Set to zero it ignores them
        self.eps = exp(self.i * pi / 4)  # direction of integration for h1 and h2. For h0 it is 1/eps
        self.num_of_residues = 1  # sandeep
        self.debug = False

        self.n0 = self.build_n0()

        # self.r0 = self.build_root_h0()
        self.r0 = self.n0 + 0.5

        if self.debug:
            print('n0', self.n0)
            print('r0', nstr(self.r0, 5))
            # print('r01', nstr(self.r01, 5))

        self.q = self.set_q_value()
        self.m, self.h = self.set_m_and_h_values()
        self.lin_grid = linspace(-self.q, self.q, self.m)

    def build_n0(self):
        w0 = self.build_root_h0()
        n0 = floor(w0)
        return n0

    def build_root_h0(self):
        w0 = sqrt(im(self.s) / (2 * pi))
        return w0

    def double_exp_residue_pos_h0(self, k):
        # t = (k - 0.5) / (self.alpha * self.eps)
        t = (self.n0 + k - self.r0) / (self.alpha * self.eps)
        xp = asinh(t)
        return xp

    def phi_hat_h0(self, m):
        xp = self.double_exp_residue_pos_h0(m)
        return self.phi_hat(xp)

    def phi_hat(self, x):
        if (im(x)) > 0:
            y = +1 / (1 - exp(-2 * pi * self.i * x / self.h))
        else:
            y = -1 / (1 - exp(+2 * pi * self.i * x / self.h))
        return y

    def series_h0(self, s):
        sum0 = nsum(lambda k: power(k, -s), [1, self.n0])
        if self.debug:
            print('series_h0 : ', sum0)
        return sum0

    def series_residues_h0(self, s):
        val = mpf('0.0')
        for k in range(- self.num_of_residues + 1, self.num_of_residues + 1):
            if self.n0 + k > 0:
                val += self.phi_hat_h0(k) * power(self.n0 + k, -s)

        # val = nsum(lambda k: (self.n0 + k > 0) * self.phi_hat_h0(k) * power(self.n0 + k, -s),
        #            [- self.num_of_residues + 1, self.num_of_residues])
        if self.debug:
            print('n0 : ', self.n0)
            print('series_residues_h0 : ', val)
        return val

    def integrand_h0(self, s, x):
        zn = self.r0 + self.alpha * self.eps * sinh(x)
        y = -(self.alpha * self.eps * cosh(x)) * exp(self.i * pi * zn * zn) * power(zn, -s) \
            / (exp(self.i * pi * zn) - exp(-self.i * pi * zn))
        return y

    def integral_h0(self, s):
        sum0 = mpf('0.0')
        for jx in self.lin_grid:
            sum0 += self.integrand_h0(s, jx)
        y = self.h * sum0
        if self.debug:
            print('integral_h0 : ', y)
        return y

    def set_q_value(self):
        mp_org_accuracy = mp.dps
        mp.dps = 20
        # h0 :

        q_est = asinh(sqrt(mp_org_accuracy / pi * ln(10.)) / self.alpha)
        if self.debug:
            print('q_est', q_est)
            print('lower', log10(abs(self.integrand_h0(self.s, 0.8 * q_est))) + mp_org_accuracy)
            print('upper', log10(abs(self.integrand_h0(self.s, 1.5 * q_est))) + mp_org_accuracy)
        q_right = findroot(lambda q: log10(abs(self.integrand_h0(self.s, +q))) + mp_org_accuracy,
                           (0.8 * q_est, 1.5 * q_est),
                           solver='bisect', tol=1.e-8)
        q_left = findroot(lambda q: log10(abs(self.integrand_h0(self.s, -q))) + mp_org_accuracy,
                          (0.8 * q_est, 1.5 * q_est),
                          solver='bisect', tol=1.e-8)
        q = max(q_right, q_left)
        if self.debug:
            print('q', q)
        mp.dps = mp_org_accuracy
        return q

    def set_m_and_h_values(self):
        h = 0.25 * power(pi, 2) / (ln(10.) * mp.dps)  # sandeep
        m = 2 * int(self.q / h) + 1
        h = 2.0 * self.q / (m - 1)
        return m, h

    def set_q_m_and_h(self, q, h):
        self.q = q
        self.m = 2 * int(self.q / h) + 1
        self.h = 2.0 * self.q / (self.m - 1)
        self.lin_grid = linspace(-self.q, self.q, self.m)
        return

    def total_value(self, s):
        series_and_residues_h0 = self.series_h0(s) - self.residue_factor * self.series_residues_h0(s)
        val_our = self.integral_h0(s) + series_and_residues_h0

        if self.debug:
            print('series_plus_residue_h0 ', series_and_residues_h0)
            print('val_our ', val_our)

        return val_our

    def riemann_ours(self):
        print('m = ', self.m, ' q = ', nstr(self.q, 5), ' h = ', nstr(self.h, 5))
        sum1 = self.total_value(self.s)
        sum2 = self.total_value(1 - conj(self.s))
        pre_fac = power(pi, self.s - 0.5) * gamma(0.5 * (1 - self.s)) / gamma(0.5 * self.s)
        zeta_ours = sum1 + pre_fac * conj(sum2)

        if self.debug:
            print('sum1 ', sum1)
            print('sum2 ', sum2)
            print('zeta_ours ', zeta_ours)

        return zeta_ours


def riemann_ours(s):
    val = mpf('0.0')
    if im(s) > 0:
        rz_obj = RiemannZeta(s)
        val = rz_obj.riemann_ours()
    else:
        pre_fac = power(pi, -(1-s) / 2) * gamma((1-s) / 2) / (power(pi, -s / 2) * gamma(s / 2))
        rz_obj = RiemannZeta(1-s)
        val = pre_fac * rz_obj.riemann_ours()
    return val


def riemann_ours_h(s, q, h):
    rz_obj = RiemannZeta(s)
    rz_obj.set_q_m_and_h(q, h)
    return rz_obj.riemann_ours()


def check_for_one_setting(s, accuracy, accuracy_mpmath):
    mp_org_accuracy = mp.dps
    print('Calculating RiemannZeta function for ..')
    print('s = ', nstr(s, 8))
    print('------------------------------------------------------------')

    print('Calculating RiemannZeta function by DE method ...')
    start = time.time()
    val_our = riemann_ours(s)
    end = time.time()
    time_ours = (end - start)
    # print('val_ours  	:', nstr(val_our, 8))
    print('Done')

    print('Calculating using mpmath RiemannZetaPhi function ...')
    start = time.time()
    mp.dps = accuracy_mpmath
    i = mpc('0.0', '1.0')
    val_ref = zeta(s)
    end = time.time()

    time_mpmath = (end - start)

    err = abs(val_our - val_ref)
    time_ours = round(time_ours, 3)
    time_mpmath = round(time_mpmath, 3)

    print('mpmath :', nstr(val_ref, accuracy))
    print('ours   :', nstr(val_our, accuracy))
    #
    # print('time mpmath:', time_mpmath)
    # print('time ours  :', time_ours)
    print('s :', nstr(s, 10))
    error = round(log10(err), 2)
    print('log error  :', error)
    assert (error < -accuracy)

    mp.dps = mp_org_accuracy

    print('val_mpmath	:', nstr(val_ref, 25))
    print('val_ours  	:', nstr(val_our, 25))
    print('------------------------------------------------------------')

    # print('Desired accuracy achieved?	:', flag, )
    # print('Log10 errors in real and imaginary parts :', diff_real, ',', diff_img)
    # print('time taken mpmath :', round((end - start), 3))
    # print('time taken ours   :', round((end1 - start1), 3))
    return


def check_for_multiple_settings():
    mp_org_accuracy = mp.dps

    mp.dps = 1000  # temporary
    i = mpc('0.0', '1.0')
    accuracy_vec = [100]  # ensure acc_mpmath is larger than this accuracy
    s_vec = [mpf('0.0'), mpf('0.1'), mpf('0.2'), mpf('0.3'), mpf('0.4'), mpf('0.5'), mpf('0.6'), mpf('0.7'), mpf('0.8'),
             mpf('0.9'), mpf('1.0')]
    t_vec = [mpf('1.0'), mpf('10.0'), mpf('100.0')]
    ls = [(acc, s, t) for acc in accuracy_vec for s in s_vec for t in t_vec]

    acc_mpmath = 150
    for (acc, sig, t) in ls:
        mp.dps = acc + 20
        s = sig + i * t
        check_for_one_setting(s, acc, acc_mpmath)

    mp.dps = mp_org_accuracy
    print('checking for multiple settings finished successfully')
    return


def plot_conformal_mapping(results_dir):
    mp_org_accuracy = mp.dps

    i = mpc('0.0', '1.0')
    xp = range(-6, 7)

    s = mpc('0.6', '100.0')

    riemann_obj = RiemannZeta(s)

    xi = list(map(lambda x: im(riemann_obj.double_exp_residue_pos_h0(x)), xp))

    # plt.plot(xp, xi, 'ro')

    fig, ax = plt.subplots()
    ax.plot(xp, xi, 'ro')

    ax.set_title(r'$\alpha = 1/4$')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    # remove the ticks from the top and right edges
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.axhline(y=0)
    # ax.axvline(x=0, color='k')

    # n = [kVal(x1, s) for x1 in qr]
    for i, txt in enumerate(xp):
        ax.annotate(txt, (xp[i], xi[i]-0.07))

    plt.show()


    # file_name = "riemann_conformal_mapping.png"
    #
    # if not os.path.isdir(results_dir):
    #     os.makedirs(results_dir)
    # plt.savefig(results_dir + file_name)
    #
    # plt.show()
    #
    # mp.dps = mp_org_accuracy
    # print('Conformal mapping experiment finished successfully')
    return


def check_plots_h0(results_dir):
    mp_org_accuracy = mp.dps
    # mp.dps = 10
    num_cols, num_rows = 2, 2
    q = mpf('3.5')
    xp = np.linspace(float(-q), float(q), num=101)

    s_list = [mpc('0.6', '100.0'), mpc('0.6', '1000.0'), mpc('0.6', '10000.0'), mpc('0.6', '100000.0')]

    m = 0
    for k in range(1, num_cols + 1):
        for l in range(1, num_rows + 1):
            s = s_list[m]
            riemann_obj = RiemannZeta(s)
            m += 1
            xr = list(map(lambda x: re(riemann_obj.integrand_h0(s, x)), xp))
            xi = list(map(lambda x: im(riemann_obj.integrand_h0(s, x)), xp))
            plt.subplot(num_rows, num_cols, m)
            plt.tight_layout()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(xp, xr, 'r', xp, xi, 'g')
            plt.title(s)

    file_name = "riemann_plots_h0.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + file_name)

    mp.dps = mp_org_accuracy

    plt.show()
    print('Experiment_plot_h0 finished successfully')
    return


def check_for_h_dependence():
    accuracy = 600
    extra_accuracy = 20
    mp.dps = accuracy + extra_accuracy

    s = mpc('0.6', '1000.0')

    start = time.time()
    riemann_ref = zeta(s)
    end = time.time()
    time_mpmath = (end - start)
    q = mpf('6.0')
    h = mpf('0.05')

    file_name = 'riemann_results_with_residue.csv'
    col_values = []
    col_names = ['h', 'error']
    for k in range(1, 6):
        print(k)
        start = time.time()
        riemann_ours_val = riemann_ours_h(s, q, h)
        end = time.time()

        err = abs(riemann_ours_val - riemann_ref)
        # mp.dps = accuracy
        error = log10(err)
        # print 'log error  :', error
        col_values.append([nstr(h, 5), nstr(error, 5)])
        print('theirs ', riemann_ref)
        print('ours   ', riemann_ours_val)
        print('error', nstr(error, 5))
        h = 0.5 * h
        res = pd.DataFrame.from_records(col_values, columns=col_names)
        res.to_csv(file_name)

    res = pd.DataFrame.from_records(col_values, columns=col_names)
    res.to_csv(file_name)
    return


if __name__ == "__main__":
    mp.pretty = True
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Experiment/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    accuracy = 50
    extra_accuracy = 20
    accuracy_mpmath = 100
    mp.dps = accuracy + extra_accuracy

    s = mpc('0.5', '100000.0')

    # check_for_one_setting(s, accuracy, accuracy_mpmath)
    # check_for_multiple_settings()
    # check_for_h_dependence()
    # check_plots_h0(results_dir)
    plot_conformal_mapping(results_dir)
