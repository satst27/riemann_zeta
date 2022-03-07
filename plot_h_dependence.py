from matplotlib import pyplot as plt
import os
import pandas as pd

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    print(script_dir)

    # file_name = script_dir + r'/Experiment/riemann_h_dependence.csv'
    file_name = script_dir + r'/Experiment/lerch_h_dependence.csv'
    # file_name = script_dir + r'/Experiment/dirichlet_h_dependence.csv'
    df = pd.read_csv(file_name, index_col=0)
    print(df)
    fig = plt.figure()
    plt.rcParams.update({'font.size': 15})
    # df.plot(x='m', y=['error_one_0', 'error_one_1', 'error_quarter_0', 'error_quarter_1'])
    plt.plot(df['m'], df['error_one_0'], '-ro', label=r'$\alpha=1.00$, num=0')
    plt.plot(df['m'], df['error_one_1'], '-gx', label=r'$\alpha=1.00$, num=1')
    plt.plot(df['m'], df['error_quarter_0'], '-bv', label=r'$\alpha=0.25$, num=0')
    plt.plot(df['m'], df['error_quarter_1'], '-ms', label=r'$\alpha=0.25$, num=1')
    plt.xlabel('m')
    plt.ylabel(r'$\log_{10}|error|$')

    # plt.title(r'Zeta, $s=0.6+10^8 i$')
    plt.title(r'Lerch, $s=0.6-10^8 i$, $\lambda=0.7$, $a=0.9$')
    # plt.title(r'Dirichlet L, $s=0.6+10^8 i$')

    plt.legend(fontsize=14)

    axes = plt.gca()
    axes.set_aspect(5.1)

    plt.savefig(file_name.replace('csv', 'png'), dpi=fig.dpi)
    plt.show()