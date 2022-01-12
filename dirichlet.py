from mpmath import *
from matplotlib import pyplot as plt
import numpy as np
import time
import os
from lerch import dirichlet_ours as dirichlet_from_lerch

# consult arxiv paper for h0, h1 and h2 series and integrals


class Dirichlet:
    def __init__(self, s, chi):
        self.s = s
        self.chi = chi
        self.md = len(chi)
        if self.md > 1:
            assert (chi[0] == 0 and chi[1] == 1)
        else:
            assert (chi[0] == 1)
        self.i = mpc('0', '1')
        self.alpha = mpf('1.0')  # sandeep
        self.residue_factor = mpf(
            '1.0')  # if set to 1 it takes residue contributions into account. Set to zero it ignores them
        self.eps = exp(self.i * pi / 4)  # direction of integration for h1 and h2. For h0 it is 1/eps
        self.num_of_residues = 1  # sandeep
        self.debug = False

        self.verify_character_is_primitive()

        self.n0 = self.build_n0()

        self.r0 = self.build_root_h0()
        # self.r0 = self.n0 + 0.5

        if self.debug:
            print('n0', self.n0)
            print('r0', nstr(self.r0, 5))
            # print('r01', nstr(self.r01, 5))

        self.q = self.set_q_value()
        self.m, self.h = self.set_m_and_h_values()
        self.lin_grid = linspace(-self.q, self.q, self.m)

    def build_n0(self):
        w0 = self.build_root_h0()
        n0 = floor(re(w0)-im(w0))
        return n0

    def build_root_h0(self):
        w0 = sqrt(self.s * self.md / (2 * pi * self.i))
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

    def xchar(self, n):
        nm = int(n % self.md)
        return self.chi[nm]

    def xp(self, s):
        cd = nsum(lambda n: self.xchar(n) * exp(-2 * pi * self.i * n / self.md), [0, self.md - 1])
        a = 0.5 * (1. - self.chi[-1 + self.md])
        rho = power(self.i, -a / 2) / sqrt(cd) * power(self.md, 0.25)
        y = rho * power(self.md / pi, s / 2) * gamma(0.5 * (s + a))
        return y

    def wd(self, x):
        if self.md % 2 == 1:
            y = 1. / (2 * self.md * self.i) * nsum(lambda k: self.xchar(k) * exp(-pi * self.i * k * k / self.md)
                                                             * csc(pi * (x - k) / self.md), [1, self.md])
        else:
            y = 1. / (2 * self.md * self.i) * nsum(lambda k: self.xchar(k) * exp(-pi * self.i * k * k / self.md)
                                                             * cot(pi * (x - k) / self.md), [1, self.md])
        return y

    def verify_character_is_primitive(self):
        cd = nsum(lambda n: self.xchar(n) * exp(-2 * pi * self.i * n / self.md), [1, self.md])
        abs_diff = 0
        for j in range(1, self.md + 1):
            sum_val = nsum(lambda n: self.xchar(n) * exp(-2 * pi * self.i * j * n / self.md), [1, self.md])
            expected_val = cd * conj(self.xchar(j))
            abs_diff = max(abs(sum_val - expected_val), abs_diff)
        assert mp.dps - 1 + log10(abs_diff) < 0, 'character supplied is not primitive'  # sandeep
        return

    def series_h0(self, s):
        sum0 = nsum(lambda k: self.xchar(k) * power(k, -s), [1, self.n0])
        if self.debug:
            print('series_h0 : ', sum0)
        return sum0

    def series_residues_h0(self, s):
        val = mpf('0.0')
        for k in range(- self.num_of_residues + 1, self.num_of_residues + 1):
            if self.n0 + k > 0:
                val += self.phi_hat_h0(k) * self.xchar(self.n0 + k) * power(self.n0 + k, -s)

        # val = nsum(lambda k: (self.n0 + k > 0) * self.phi_hat_h0(k) * power(self.n0 + k, -s),
        #            [- self.num_of_residues + 1, self.num_of_residues])
        if self.debug:
            print('n0 : ', self.n0)
            print('series_residues_h0 : ', val)
        return val

    def integrand_h0(self, s, x):
        zn = self.r0 + self.alpha * self.eps * sinh(x)
        y = -(self.alpha * self.eps * cosh(x)) * exp(self.i * pi * zn * zn / self.md) * power(zn, -s) * self.wd(zn)
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

        q_est = asinh(sqrt(mp_org_accuracy / pi * self.md * ln(10.)) / self.alpha)
        low_val = log10(abs(self.integrand_h0(self.s, 0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h0(self.s, 1.5 * q_est))) + mp_org_accuracy
        if self.debug:
            print('q_est', q_est)
            print('lower_right', low_val)
            print('upper_right', high_val)

        if low_val * high_val < 0:
            q_right = findroot(lambda q: log10(abs(self.integrand_h0(self.s, +q))) + mp_org_accuracy,
                               (0.8 * q_est, 1.5 * q_est),
                               solver='bisect', tol=1.e-8)
        else:
            q_right = q_est

        low_val = log10(abs(self.integrand_h0(self.s, -0.8 * q_est))) + mp_org_accuracy
        high_val = log10(abs(self.integrand_h0(self.s, -1.5 * q_est))) + mp_org_accuracy

        if self.debug:
            print('q_est', q_est)
            print('lower_left', low_val)
            print('upper_left', high_val)

        if low_val * high_val < 0:
            q_left = findroot(lambda q: log10(abs(self.integrand_h0(self.s, -q))) + mp_org_accuracy,
                              (0.8 * q_est, 1.5 * q_est),
                              solver='bisect', tol=1.e-8)
        else:
            q_left = q_est

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

    def dirichlet_ours(self):
        # print('m = ', self.m, ' q = ', nstr(self.q, 5), ' h = ', nstr(self.h, 5))
        cs = 1 - conj(self.s)
        sum1 = self.total_value(self.s)
        sum2 = self.total_value(cs)
        pre_fac = conj(self.xp(cs)) / self.xp(self.s)
        dirichlet_val = sum1 + pre_fac * conj(sum2)

        if self.debug:
            print('sum1 ', sum1)
            print('sum2 ', sum2)
            print('dirichlet_val ', dirichlet_val)

        return dirichlet_val


def dirichlet_ours(s, chi):
    val = mpf('0.0')
    if im(s) > 0:
        rz_obj = Dirichlet(s, chi)
        val = rz_obj.dirichlet_ours()
    else:
        i = mpc('0.0', '1.0')
        q = len(chi)
        a = 0 if chi[q - 1] == 1 else -1
        print('a', a)
        pre_fac_1 = power(q / pi, (s + a) / 2) * gamma((s + a) / 2)
        pre_fac_2 = power(q / pi, (1 - s + a) / 2) * gamma((1 - s + a) / 2)
        sum0 = mpf('0.0')
        for k in range(q + 1):
            sum0 += chi[k % q] * exp(2 * pi * i * k / q)
        eX = sum0 / power(i, a) / sqrt(q)
        cs_chi = list(map(lambda x: conj(x), chi))
        rz_obj = Dirichlet(1 - s, cs_chi)
        val = rz_obj.dirichlet_ours()
        val = eX * pre_fac_2 / pre_fac_1 * val
    return val


def dirichlet_ours_h(s, chi, q, h):
    rz_obj = Dirichlet(s, chi)
    rz_obj.set_q_m_and_h(q, h)
    return rz_obj.dirichlet_ours()


def check_for_one_setting(s, chi, accuracy):
    # assert (im(s) > 0)
    mp_org_accuracy = mp.dps
    print('Calculating Dirichlet function for ..')
    print('s = ', nstr(s, 8))
    print('------------------------------------------------------------')

    print('Calculating Dirichlet function by DE method ...')
    start = time.time()
    val_our = dirichlet_ours(s, chi)
    end = time.time()
    time_ours = (end - start)
    # print('val_ours  	:', nstr(val_our, 8))
    print('Done')

    print('Calculating using mpmath DirichletPhi function for t < 10000000 and our Hurwitz zeta otherwise...')
    start = time.time()
    val_ref = mpf('0.0')
    if im(s) < 10000000.0:
        val_ref = dirichlet(s, chi)
    else:
        val_ref = dirichlet_from_lerch(s, chi)
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

    # print('Desired accuracy achieved?	:', flag, )
    # print('Log10 errors in real and imaginary parts :', diff_real, ',', diff_img)
    # print('time taken mpmath :', round((end - start), 3))
    # print('time taken ours   :', round((end1 - start1), 3))
    return


def check_for_multiple_settings():
    mp_org_accuracy = mp.dps

    mp.dps = 1000  # temporary
    i = mpc('0.0', '1.0')
    w = exp(i * pi / 3)
    chi0 = [1]
    chi1 = [0, 1, -1]
    chi2 = [0, 1, power(w, 2), -w, -w, power(w, 2), 1]
    chi3 = [0, 1, 0, -1]  # works with cot function in w(x)
    chi4 = [0, 1, i, -i, -1]
    chi5 = [0, 1, 0, -1, 0, -1, 0, 1]

    chi_vec = [chi0, chi1, chi2, chi3, chi4, chi5]

    accuracy_vec = [100]  # ensure acc_mpmath is larger than this accuracy
    s_vec = [mpf('0.0'), mpf('0.1'), mpf('0.2'), mpf('0.3'), mpf('0.4'), mpf('0.5'), mpf('0.6'), mpf('0.7'), mpf('0.8'),
             mpf('0.9'), mpf('1.0')]

    # t_vec = [mpf('1.0'), mpf('10.0'), mpf('100.0')]

    t_vec = []
    for k in range(10):
        t_vec.append(10000 * rand())

    ls = [(acc, s, t, chi) for acc in accuracy_vec for s in s_vec for t in t_vec for chi in chi_vec]

    for (acc, sig, t, chi) in ls:
        mp.dps = acc + 20
        s = sig + i * t
        check_for_one_setting(s, chi, acc)

    mp.dps = mp_org_accuracy
    print('checking for multiple settings finished successfully')
    return


def plot_conformal_mapping(results_dir):
    mp_org_accuracy = mp.dps

    i = mpc('0.0', '1.0')
    xp = range(-10, 11)

    s = mpc('0.6', '100.0')
    w = exp(i * pi / 3)
    chi_val = [0, 1, power(w, 2), -w, -w, power(w, 2), 1]

    dirichlet_obj = Dirichlet(s, chi_val)

    xr = list(map(lambda x: im(dirichlet_obj.double_exp_residue_pos_h0(x)), xp))

    plt.plot(xp, xr, 'ro')

    file_name = "dirichlet_conformal_mapping.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + file_name)

    plt.show()

    mp.dps = mp_org_accuracy
    print('Conformal mapping experiment finished successfully')
    return


def check_plots_h0(results_dir):
    mp_org_accuracy = mp.dps
    # mp.dps = 10
    num_cols, num_rows = 2, 2
    q = mpf('3.5')
    xp = np.linspace(float(-q), float(q), num=101)

    i = mpc('0.0', '1.0')
    w = exp(i * pi / 3)
    chi = [0, 1, power(w, 2), -w, -w, power(w, 2), 1]

    s_list = [mpc('0.6', '100.0'), mpc('0.6', '1000.0'), mpc('0.6', '10000.0'), mpc('0.6', '100000.0')]

    m = 0
    for k in range(1, num_cols + 1):
        for l in range(1, num_rows + 1):
            s = s_list[m]
            dirichlet_obj = Dirichlet(s, chi)
            m += 1
            xr = list(map(lambda x: re(dirichlet_obj.integrand_h0(s, x)), xp))
            xi = list(map(lambda x: im(dirichlet_obj.integrand_h0(s, x)), xp))
            plt.subplot(num_rows, num_cols, m)
            plt.tight_layout()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(xp, xr, 'r', xp, xi, 'g')
            plt.title(s)

    file_name = "dirichlet_plots_h0.png"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    plt.savefig(results_dir + file_name)

    mp.dps = mp_org_accuracy

    plt.show()
    print('Experiment_plot_h0 finished successfully')
    return


def check_for_h_dependence():
    accuracy = 400
    extra_accuracy = 20
    mp.dps = accuracy + extra_accuracy

    i = mpc('0.0', '1.0')
    w = exp(i * pi / 3)
    chi = [0, 1, power(w, 2), -w, -w, power(w, 2), 1]

    s = mpc('0.0', '1000.0')

    start = time.time()
    dirichlet_ref = dirichlet(s, chi)
    end = time.time()
    time_mpmath = (end - start)
    q = mpf('6.0')
    h = mpf('0.05')

    file_name = 'dirichlet_results_with_residue.csv'
    col_values = []
    col_names = ['h', 'error']
    for k in range(1, 6):
        print(k)
        start = time.time()
        dirichlet_ours_val = dirichlet_ours_h(s, chi, q, h)
        end = time.time()

        err = abs(dirichlet_ours_val - dirichlet_ref)
        # mp.dps = accuracy
        error = log10(err)
        # print 'log error  :', error
        col_values.append([nstr(h, 5), nstr(error, 5)])
        print('theirs ', dirichlet_ref)
        print('ours   ', dirichlet_ours_val)
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

    accuracy = 100
    extra_accuracy = 20
    mp.dps = accuracy + extra_accuracy
    i = mpc('0.0', '1.0')
    s = mpc('0.6', '10000.0')
    w = exp(i * pi / 3)
    chi_val = [0, 1, power(w, 2), -w, -w, power(w, 2), 1]
    check_for_one_setting(s, chi_val, accuracy)
    # check_for_multiple_settings()
    # check_for_h_dependence()
    # check_plots_h0(results_dir)
    # plot_conformal_mapping(results_dir)
