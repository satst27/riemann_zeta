"""
Lerch class provides functionality to calculate Hurwitz Lerch function for L(s, lambda, a). The requirements are
as follows:
im(s) < 0, 0 <= lambda <=1 and 0 < a < 1.
Consult the Arxiv paper for details on the functions implemented here.

"""
from mpmath import *
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import pandas as pd


# check if alpha needs a different direction


class Lerch:
    def __init__(self, s, lam, a, alpha=mpf('1.0'), num_of_poles=1):
        if lam == 1:
            lam = 0
        assert (0 <= lam < 1)
        # assert (0 < a < 1)
        self.s = s
        self.lam = lam
        self.a = a
        self.i = mpc('0', '1')
        self.alpha = alpha
        self.eps = exp(self.i * pi / 4)  # direction of integration for h1 and h2. For h0 it is 1/eps
        self.num_of_poles = num_of_poles
        self.debug = False

        self.n0 = self.build_n0()
        self.n1 = self.build_n1()
        self.n2 = self.build_n2()

        # the following choice of r0, r1 and r2 (the centre points for various integrals) leads to exact location
        # of the saddle points. However, this means that some singularities due to the sin function in the denominator
        # may lie very close to the saddle points

        self.r0 = self.build_root_h0()
        self.r1 = self.build_root_h1()
        self.r2 = self.build_root_h2()

        # the following choices ensure that r0, r1 and r2 fall at the centre of the two nearby singularities

        # self.r0 = (self.n0 + self.a) + 0.5
        # self.r1 = min(-self.lam / 2.0, -(self.n1 + self.lam) + 0.5)
        # self.r2 = max((1.0 - self.lam) / 2.0, +(self.n2 - self.lam) + 0.5)

        if self.debug:
            print('n0', self.n0, 'n1', self.n1, 'n2', self.n2)
            print('r0', nstr(self.r0, 5), 'r1', nstr(self.r1, 5), 'r2', nstr(self.r2, 5))

        self.q = self.set_q_value()
        self.m, self.h = self.set_m_and_h_values()
        self.lin_grid = linspace(-self.q, self.q, self.m)

    def build_n0(self):
        w0 = self.build_root_h0()
        n0 = floor(re(w0) + im(w0) - self.a)
        return n0

    def build_n1(self):
        w_minus = self.build_root_h1()
        n1 = -floor(re(w_minus) - im(w_minus) + self.lam)
        return n1

    def build_n2(self):
        w_plus = self.build_root_h2()
        n2 = floor(re(w_plus) - im(w_plus) + self.lam)
        return n2

    def build_root_h0(self):
        c = (self.lam + self.a) / 2
        w0 = c + sqrt(c * c - self.s / (2 * pi * self.i))
        return w0

    def build_root_h1(self):
        c = (self.lam + self.a) / 2
        w_minus = -c - sqrt(c * c + (1 - self.s) / (2 * pi * self.i))
        return w_minus

    def build_root_h2(self):
        c = (self.lam + self.a) / 2
        w_plus = -c + sqrt(c * c +(1 - self.s) / (2 * pi * self.i))
        return w_plus

    def double_exp_residue_pos_h0(self, k):
        # t = (k - 0.5) / (self.alpha / self.eps)
        t = (self.n0 + self.a + k - self.r0) / (self.alpha / self.eps)
        xp = asinh(t)
        return xp

    def double_exp_residue_pos_h1(self, k):
        # t = (k - 0.5) / (self.alpha * self.eps)
        t = (-self.n1 - self.lam - k - self.r1) / (self.alpha * self.eps)
        xp = asinh(t)
        return xp

    def double_exp_residue_pos_h2(self, k):
        # t = (k - 0.5) / (self.alpha * self.eps)
        t = (self.n2 - self.lam + k - self.r2) / (self.alpha * self.eps)
        xp = asinh(t)
        return xp

    def phi_hat_h0(self, m):
        xp = self.double_exp_residue_pos_h0(m)
        return self.phi_hat(xp)

    def phi_hat_h1(self, m):
        xp = self.double_exp_residue_pos_h1(m)
        return self.phi_hat(xp)

    def phi_hat_h2(self, m):
        xp = self.double_exp_residue_pos_h2(m)
        return self.phi_hat(xp)

    def phi_hat(self, x):
        if (im(x)) > 0:
            y = +1 / (1 - exp(-2 * pi * self.i * x / self.h))
        else:
            y = -1 / (1 - exp(+2 * pi * self.i * x / self.h))
        return y

    def series_h0(self):
        # start the sum from 1 for Riemann i.e. when a = 0. This ensures Riemann sub case gives meaningful results

        sum0 = nsum(lambda k: exp(2 * pi * self.i * self.lam * k) * power(k + self.a, -self.s)
        if (re(k + self.a) > 0) else 0.0, [0, self.n0])
        if self.debug:
            print('series_h0 : ', sum0)
        return sum0

    def series_residues_h0(self):
        val = nsum(
            lambda k: self.phi_hat_h0(k) * exp(2 * pi * self.i * self.lam * (self.n0 + k))
                      * power(self.n0 + k + self.a, -self.s) if re(self.n0 + k + self.a) > 0 else 0.0,
            [- self.num_of_poles + 1, self.num_of_poles])  # note the minus sign

        if self.debug:
            print('n0 : ', self.n0)
            print('series_residues_h0 : ', val)
        return val

    def series_h1(self):
        sum_h1 = nsum(lambda k: exp(-2 * pi * self.i * self.a * k) * power(k + self.lam, self.s - 1)
        if re(k + self.lam) > 0 else 0.0, [0, self.n1 - 1])  # check the start of the sum

        pre_factor = exp(-2 * pi * self.i * self.a * self.lam) * gamma(1 - self.s) \
                     / power((2 * pi), 1 - self.s) * exp(pi / 2 * self.i * (1 - self.s))
        res = pre_factor * sum_h1  # important sign missing in the Balanzario paper
        if self.debug:
            print('n1 : ', self.n1)
            print('series_h1', res)
        return res

    def series_residues_h1(self):
        pre_factor = exp(-2 * pi * self.i * self.a * self.lam) * gamma(1 - self.s) \
                     / power((2 * pi), 1 - self.s) * exp(pi / 2 * self.i * (1 - self.s))

        val = nsum(
            lambda k: self.phi_hat_h1(k)
                      * exp(-2 * pi * self.i * self.a * (self.n1 + k))
                      * power(self.n1 + k + self.lam, self.s - 1) if (self.n1 + k + self.lam > 0) else 0.0,
            [- self.num_of_poles, self.num_of_poles-1])
        val *= pre_factor
        if self.debug:
            print('series_residues_h1', val)
        return val

    def series_h2(self):
        pre_factor = exp(-2 * pi * self.i * self.a * self.lam) * gamma(1 - self.s) / power((2 * pi), 1 - self.s)  \
                     * exp(-pi / 2 * self.i * (1 - self.s))
        sum_h2 = nsum(lambda k: exp(+2 * pi * self.i * self.a * k) * power(k - self.lam, self.s - 1), [1, self.n2])
        res = pre_factor * sum_h2   # important signs missing in paper #sandeep
        if self.debug:
            print('series_h2 : ', res)
        return res

    def series_residues_h2(self):
        pre_factor = exp(-2 * pi * self.i * self.a * self.lam) * gamma(1 - self.s) / power((2 * pi), 1 - self.s) \
                     * exp(-pi / 2 * self.i * (1 - self.s))

        val = nsum(
            lambda k: - self.phi_hat_h2(k) * exp(+2 * pi * self.i * self.a * (self.n2 + k)) * power(
                self.n2 + k - self.lam, self.s - 1) if (self.n2 + k - self.lam > 0) else 0.0,
            [- self.num_of_poles + 1, self.num_of_poles])

        val *= pre_factor
        if self.debug:
            print('series_residues_h2 : ', val)
        return val

    def integrand_h0(self, x):
        zn = self.r0 + self.alpha / self.eps * sinh(x)
        y = (self.alpha / self.eps * cosh(x)) * exp(-self.i * pi * power(zn, 2) + 2
                                                    * pi * self.i * (self.a + self.lam) * zn) * power(zn, -self.s) / (
                    exp(-2 * pi * self.i * self.a + self.i * pi * zn) - exp(-self.i * pi * zn))
        return y

    def integral_h0(self):
        sum0 = mpf('0.0')
        for jx in self.lin_grid:
            sum0 += self.integrand_h0(jx)

        y = (self.h * sum0) * exp(-self.i * pi * self.a * (1 + self.a + 2 * self.lam))
        if self.debug:
            print('integral_h0 : ', y)
        return y

    def integrand_h1(self, x):
        zn = self.r1 + self.alpha * self.eps * sinh(x)
        y = (self.alpha * self.eps * cosh(x)) * exp(self.i * pi * power(zn, 2) + 2 * pi
                                                    * self.i * (self.a + self.lam) * zn) * power(zn, self.s - 1) / (
                    exp(2 * pi * self.i * self.lam + self.i * pi * zn) - exp(-self.i * pi * zn))
        if phase(zn) > pi / 4:  # this doesn't have any effect on right of the origin
            y = y * exp(2 * pi * self.i * (1 - self.s))
        return y

    def integrand_h2(self, x):
        zn = self.r2 + self.alpha * self.eps * sinh(x)
        y = (self.alpha * self.eps * cosh(x)) * exp(self.i * pi * power(zn, 2) + 2 * pi
                                                    * self.i * (self.a + self.lam) * zn) * power(zn, self.s - 1) / (
                    exp(2 * pi * self.i * self.lam + self.i * pi * zn) - exp(-self.i * pi * zn))
        if phase(zn) > pi / 4:  # this doesn't have any effect on right of the origin
            y = y * exp(2 * pi * self.i * (1 - self.s))
        return y

    def integral_h1(self):
        sum_h1 = mpf('0.0')
        for jx in self.lin_grid:
            sum_h1 += self.integrand_h1(jx)
        y = self.h * sum_h1 * exp(self.i * pi * self.lam * (self.lam + 1) - pi / 2 * self.i * (1 - self.s)) \
            * gamma(1 - self.s) / power((2 * pi), (1 - self.s))
        if self.debug:
            print('integral_h1 : ', y)
        return y

    def integral_h2(self):
        sum_h2 = mpf('0.0')
        for jx in self.lin_grid:
            sum_h2 += self.integrand_h2(jx)  # should work from 0 to (1-lambda)
        y = -self.h * sum_h2 * exp(self.i * pi * self.lam * (self.lam + 1) - pi / 2 * self.i * (1 - self.s)) \
            * gamma(1 - self.s) / power((2 * pi), (1 - self.s))
        if self.debug:
            print('n2 : ', self.n2)
            print('integral_h2 : ', y)
        return y

    def set_q_value(self):
        mp_org_accuracy = mp.dps
        mp.dps = 20
        # h0 :
        q_est = asinh(sqrt(mp_org_accuracy / pi / 2 * ln(10.)) * self.alpha)
        low_val = log10(abs(self.integrand_h0(0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h0(1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('h0_lower_right', low_val)
            print('h0_upper_right', high_val)

        if low_val * high_val < 0:
            q_right = findroot(lambda q: log10(abs(self.integrand_h0(+q))) + mp_org_accuracy,
                               (0.8 * q_est, 1.5 * q_est),
                               solver='bisect', tol=1.e-8)
        else:
            q_right = q_est

        low_val = log10(abs(self.integrand_h0(-0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h0(-1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('h0_lower_left', low_val)
            print('h0_upper_left', high_val)

        if low_val * high_val < 0:
            q_left = findroot(lambda q: log10(abs(self.integrand_h0(-q))) + mp_org_accuracy,
                              (0.8 * q_est, 1.5 * q_est),
                              solver='bisect', tol=1.e-8)
        else:
            q_left = q_est

        q_h0 = max(q_right, q_left)

        if self.debug:
            print('q_h0', q_h0)

        # h1: this case corresponds to very small correction that in practise is much smaller than the required accuracy
        q_est = asinh(sqrt(mp_org_accuracy / pi / 2 * ln(10.)) / self.alpha)

        low_val = log10(abs(self.integrand_h1(0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h1(1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('h1_lower_right', low_val)
            print('h1_upper_right', high_val)

        if low_val * high_val < 0:
            q_right = findroot(lambda q: log10(abs(self.integrand_h1(+q))) + mp_org_accuracy,
                               (0.8 * q_est, 1.5 * q_est),
                               solver='bisect', tol=1.e-8)
        else:
            q_right = q_est

        low_val = log10(abs(self.integrand_h1(-0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h1(-1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('h1_lower_left', low_val)
            print('h1_upper_left', high_val)

        if low_val * high_val < 0:
            q_left = findroot(lambda q: log10(abs(self.integrand_h1(-q))) + mp_org_accuracy,
                              (0.8 * q_est, 1.5 * q_est),
                              solver='bisect', tol=1.e-8)
        else:
            q_left = q_est

        q_h1 = max(q_right, q_left)

        # h2 :
        q_est = asinh(sqrt(mp_org_accuracy / pi * ln(10.)) / self.alpha)

        low_val = log10(abs(self.integrand_h2(0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h2(1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('h2_lower_right', low_val)
            print('h2_upper_right', high_val)

        if low_val * high_val < 0:
            q_right = findroot(lambda q: log10(abs(self.integrand_h2(+q))) + mp_org_accuracy,
                               (0.8 * q_est, 1.5 * q_est),
                               solver='bisect', tol=1.e-8)
        else:
            q_right = q_est

        low_val = log10(abs(self.integrand_h2(-0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h2(-1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('h2_lower_left', low_val)
            print('h2_upper_left', high_val)

        if low_val * high_val < 0:
            q_left = findroot(lambda q: log10(abs(self.integrand_h2(-q))) + mp_org_accuracy,
                              (0.8 * q_est, 1.5 * q_est),
                              solver='bisect', tol=1.e-8)
        else:
            q_left = q_est

        q_h2 = max(q_right, q_left)

        if self.debug:
            print('q_h2', q_h2)
        q = max([q_h0, q_h1, q_h2])
        mp.dps = mp_org_accuracy

        if self.debug:
            print('q', q)

        return q

    def set_m_and_h_values(self):
        h = 0.25 * power(pi, 2) / (ln(10.) * mp.dps)
        m = 2 * int(self.q / h) + 1
        h = 2.0 * self.q / (m - 1)
        return m, h

    def set_q_m_and_h(self, q, h):
        self.q = q
        self.m = 2 * int(self.q / h) + 1
        self.h = 2.0 * self.q / (self.m - 1)
        self.lin_grid = linspace(-self.q, self.q, self.m)
        return

    def lerch_ours(self):
        series_and_residues_h0 = self.series_h0() + self.series_residues_h0()
        val_h0 = self.integral_h0() + series_and_residues_h0

        series_and_residues_h1 = self.series_h1() + self.series_residues_h1()
        val_h1 = self.integral_h1() + series_and_residues_h1

        series_and_residues_h2 = self.series_h2() + self.series_residues_h2()
        val_h2 = self.integral_h2() + series_and_residues_h2

        val_our = val_h0 + val_h1 + val_h2
        if self.debug:
            print('m = ', self.m, ' q = ', nstr(self.q, 5), ' h = ', nstr(self.h, 5))
            print('series_plus_residue_h0 ', series_and_residues_h0)
            print('series_plus_residue_h1 ', series_and_residues_h1)
            print('series_plus_residue_h2 ', series_and_residues_h2)
            print('val_h0 ', val_h0)
            print('val_h1 ', val_h1)
            print('val_h2 ', val_h2)
        return val_our


def lerch_ours(s, lam, a, alpha=mpf('1.0'), num_of_poles=1):
    val = mpf('0.0')
    if im(s) < 0:
        lerch = Lerch(s, lam, a, alpha, num_of_poles)
        val = lerch.lerch_ours()
    else:
        i = mpc('0.0', '1.0')
        lerch1 = Lerch(1 - s, 1 - a, lam, alpha, num_of_poles)
        lerch2 = Lerch(1 - s, a, 1 - lam, alpha, num_of_poles)
        pre_fac = power(2 * pi, -(1 - s)) * gamma(1 - s)
        pre_fac_1 = exp(+pi / 2 * i * (1 - s) - 2 * pi * i * a * lam)
        pre_fac_2 = exp(-pi / 2 * i * (1 - s) + 2 * pi * i * a * (1 - lam))

        val_1 = lerch1.lerch_ours()
        val_2 = lerch2.lerch_ours()

        val = pre_fac * (pre_fac_1 * val_1 + pre_fac_2 * val_2)

    return val


def lerch_ours_h(s, lam, a, q, h, alpha=mpf('1.0'), num_of_poles=1):
    assert (im(s) < 0)  # only implemented for im(s) < 0
    lerch = Lerch(s, lam, a, alpha, num_of_poles)
    lerch.set_q_m_and_h(q, h)
    return lerch.lerch_ours()


def hurwitz_ours(s, a):
    return lerch_ours(s, 0.0, a)


def riemann_ours(s):
    return lerch_ours(s, 0.0, 0.0)


def dirichlet_ours(s, chi_val):
    m = len(chi_val)
    val = mpf('0.0')
    for k in range(1, m + 1):
        val += hurwitz_ours(s, mpf('1.0') * k / m) * chi_val[k % m]
    return val * power(m, -s)


def check_for_one_setting(s, lam, a, accuracy, accuracy_mpmath):
    mp_org_accuracy = mp.dps
    print('Calculating Lerch function for ..')
    print('s = ', nstr(s, 8))
    print('a = ', nstr(a, 8))
    print('lambda = ', nstr(lam, 8))
    print('------------------------------------------------------------')

    print('Calculating Lerch function by DE method ...')
    start = time.time()
    if abs(im(s)) < 100.0:
        val_our = lerch_ours(s, lam, a, 0.25)  # alpha  = 0.25
    else:
        val_our = lerch_ours(s, lam, a)  # alpha  = 1.0
    end = time.time()
    time_ours = (end - start)
    # print('val_ours  	:', nstr(val_our, 8))
    print('Done')

    print('Calculating using mpmath LerchPhi function ...')
    start = time.time()
    mp.dps = accuracy_mpmath
    i = mpc('0.0', '1.0')
    val_ref = lerchphi(exp(2 * pi * i * lam), s, a)
    end = time.time()

    time_mpmath = (end - start)

    err = abs(val_our - val_ref)
    time_ours = round(time_ours, 3)
    time_mpmath = round(time_mpmath, 3)

    print('mpmath :', nstr(val_ref, accuracy))
    print('ours   :', nstr(val_our, accuracy))
    #
    print('time mpmath:', time_mpmath)
    print('time ours  :', time_ours)

    print('s :', nstr(s, 10))
    error = round(log10(err), 2)
    print('log error  :', error)
    assert (error < -accuracy)

    mp.dps = mp_org_accuracy
    print('------------------------------------------------------------')

    # print('Desired accuracy achieved?	:', flag, )
    # print('Log10 errors in real and imaginary parts :', diff_real, ',', diff_img)
    # print('time taken mpmath :', round((end - start), 3))
    # print('time taken ours   :', round((end1 - start1), 3))
    return


def check_timing():
    mp.dps = 1000
    s = mpc('0.6', '-100.0')
    lam = mpf('0.3')
    a = mpf('0.7')

    accuracy = 50
    extra_accuracy = 20
    mp.dps = accuracy + extra_accuracy

    accuracy_mpmath = 90
    check_for_one_setting(s, lam, a, accuracy, accuracy_mpmath)

    s = mpc('0.6', '-1000.0')
    accuracy_mpmath = 650
    check_for_one_setting(s, lam, a, accuracy, accuracy_mpmath)
    return


def check_for_multiple_settings():
    mp_org_accuracy = mp.dps

    mp.dps = 1000  # temporary
    i = mpc('0.0', '1.0')
    accuracy_vec = [100]  # ensure acc_mpmath is larger than this accuracy
    s_vec = [mpf('0.0'), mpf('0.1'), mpf('0.2'), mpf('0.3'), mpf('0.4'), mpf('0.5'), mpf('0.6'), mpf('0.7'), mpf('0.8'),
             mpf('0.9'), mpf('1.0')]
    t_vec = [mpf('-1.0'), mpf('-10.0'), mpf('-100.0')]
    ls = [(acc, s, t) for acc in accuracy_vec for s in s_vec for t in t_vec]

    a = mpf('0.3')
    lam = mpf('0.7')

    acc_mpmath = 150
    for (acc, sig, t) in ls:
        mp.dps = acc + 20
        s = sig + i * t
        check_for_one_setting(s, lam, a, acc, acc_mpmath)

    mp.dps = mp_org_accuracy
    print('checking for multiple settings finished successfully')
    return


def plot_conformal_mapping(results_dir):
    mp_org_accuracy = mp.dps

    i = mpc('0.0', '1.0')
    xp = range(-10, 11)

    s = mpc('0.6', '-100.0')
    lam = mpf('0.7')
    a = mpf('0.9')

    lerch_obj = Lerch(s, lam, a)

    xr = list(map(lambda x: im(lerch_obj.double_exp_residue_pos_h0(x)), xp))

    plt.plot(xp, xr, 'ro')

    file_name = "lerch_conformal_mapping.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + file_name)

    plt.show()

    mp.dps = mp_org_accuracy
    print('Conformal mapping experiment finished successfully')
    return


def check_plots_h2(results_dir):
    mp_org_accuracy = mp.dps
    num_cols, num_rows = 2, 2
    q = mpf('3.5')
    xp = np.linspace(float(-q), float(q), num=101)

    s = mpc('0.6', '-100.0')
    lam = mpf('0.7')
    a_list = [mpf('10.1'), mpf('0.3'), mpf('0.6'), mpf('0.9')]

    m = 0
    for k in range(1, num_cols + 1):
        for l in range(1, num_rows + 1):
            a = a_list[m]
            print('a', a, mp.dps)
            lerch_obj = Lerch(s, lam, a)
            m += 1
            xr = list(map(lambda x: re(lerch_obj.integrand_h2(x)), xp))
            xi = list(map(lambda x: im(lerch_obj.integrand_h2(x)), xp))
            plt.subplot(num_rows, num_cols, m)
            plt.tight_layout()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(xp, xr, 'r', xp, xi, 'g')
            plt.title(a)

    file_name = "lerch_plots_h.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + file_name)

    mp.dps = mp_org_accuracy

    plt.show()
    print('Experiment2 finished successfully')
    return


def check_plots_h0(results_dir):
    mp_org_accuracy = mp.dps
    # mp.dps = 10
    num_cols, num_rows = 2, 2
    q = mpf('3.5')
    xp = np.linspace(float(-q), float(q), num=101)

    s = mpc('0.6', '-100.0')
    lam = mpf('0.7')
    a_list = [mpf('10.1'), mpf('0.3'), mpf('0.6'), mpf('0.9')]

    m = 0
    for k in range(1, num_cols + 1):
        for l in range(1, num_rows + 1):
            a = a_list[m]
            lerch_obj = Lerch(s, lam, a)
            m += 1
            xr = list(map(lambda x: re(lerch_obj.integrand_h0(x)), xp))
            xi = list(map(lambda x: im(lerch_obj.integrand_h0(x)), xp))
            plt.subplot(num_rows, num_cols, m)
            plt.tight_layout()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(xp, xr, 'r', xp, xi, 'g')
            plt.title(a)

    file_name = "lerch_plots_h0.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + file_name)

    mp.dps = mp_org_accuracy

    plt.show()
    print('Experiment_plot_h0 finished successfully')
    return


def check_for_h_dependence(results_dir):
    accuracy = 700
    extra_accuracy = 20
    mp.dps = accuracy + extra_accuracy

    s = mpc('0.6', '-100000000.0')
    lam = mpf('0.7')
    a = mpf('0.9')

    start = time.time()
    lerch_ref = lerch_ours(s, lam, a)
    # lerch_ref = mpc(
    #     '0.81247019538516162988458183946323724256833496580904919450961814182536062164349287836058069283622909065525733252800346505004056771086293627798453924747001401715853928914907495110853165178593260535020745484593622087300894797417001977136365113528561705236290380246549185958350183158074681872968223381166767505208140148451680862468805651884199967009408874948257629881738422195270219223242998595084788800179736362566514157769198339182465220854221531481253473868686462382901163512466878767479301695158022839',
    #     '-0.38668807953602791631487998284981675490992185017136017621799054617569628032573177414004617051186093974358220543390275296260197536984579239314063338974580085896185136719594840526754207518488008902967600066006397336110935481031531207155346444490366965070609925862764689023840945792670597141174729394680267558313689709271175135410503786971336130588711652706013635156555420179598718094312929455162638724581487454460245773865601084489017230231188225055763676865749716644312520399424369878885703704610301469')

    end = time.time()
    time_mpmath = (end - start)
    q = mpf('6.0')
    h = mpf('0.05')

    col_values = []
    col_names = ['h', 'm', 'error_one_0', 'error_one_1', 'error_quarter_0', 'error_quarter_1']

    alpha = 1.0
    num_of_poles = 0
    lerch_obj0 = Lerch(s, lam, a, alpha, num_of_poles)

    alpha = 1.0
    num_of_poles = 1
    lerch_obj1 = Lerch(s, lam, a, alpha, num_of_poles)

    alpha = 0.25
    num_of_poles = 0
    lerch_obj2 = Lerch(s, lam, a, alpha, num_of_poles)

    alpha = 0.25
    num_of_poles = 1
    lerch_obj3 = Lerch(s, lam, a, alpha, num_of_poles)

    lerch_obj_list = [lerch_obj0, lerch_obj1, lerch_obj2, lerch_obj3]

    file_name = results_dir + r'lerch_h_dependence_n0.csv'

    for k in range(1, 6):
        print(k)
        error_list = []
        for lerch_obj in lerch_obj_list:
            lerch_obj.set_q_m_and_h(q, h)
            riemann_ours_val = lerch_obj.lerch_ours()
            err = abs(riemann_ours_val - lerch_ref)
            error = log10(err)
            error_list.append(error)

        col_values.append([nstr(h, 5), lerch_obj0.m, nstr(error_list[0], 5), nstr(error_list[1], 5),
                           nstr(error_list[2], 5), nstr(error_list[3], 5)])
        # print('theirs ', riemann_ref)
        # print('ours   ', riemann_ours_val)
        # print('error  ', nstr(error, 5))
        h = 0.5 * h
        res = pd.DataFrame.from_records(col_values, columns=col_names)
        res.to_csv(file_name)

    res = pd.DataFrame.from_records(col_values, columns=col_names)

    res.to_csv(file_name, index=False)
    return


if __name__ == "__main__":
    mp.pretty = True
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Experiment/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    accuracy = 200
    extra_accuracy = 20
    accuracy_mpmath = 300
    mp.dps = accuracy + extra_accuracy

    s = mpc('0.6', '100.0')
    lam = mpf('0.3')
    a = mpf('0.7')

    # Uncomment the tests below that you want to run
    check_for_one_setting(s, lam, a, accuracy, accuracy_mpmath)
    # check_for_multiple_settings()
    # check_for_h_dependence(results_dir)
    # check_timing()
    # check_plots_h2(results_dir)
    # check_plots_h0(results_dir)
    # plot_conformal_mapping(results_dir)

