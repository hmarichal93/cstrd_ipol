import pandas as pd
import matplotlib.pyplot as plt

if __name__=="__main__":
    filename = "/data/maestria/publicaciones/dendrometria/ipol/experimentos/resize_1500_active_contour.csv"
    df_1500 = pd.read_csv(filename)

    filename = "/data/maestria/publicaciones/dendrometria/ipol/experimentos/resize_640_active_contour.csv"
    df_640 = pd.read_csv(filename)
    filename = "/data/maestria/publicaciones/dendrometria/ipol/experimentos/resize_orig_active_contour.csv"
    df_orig = pd.read_csv(filename)


    plt.figure()
    for label, df in zip(['640','1500','orig'],[df_640, df_1500, df_orig]):
        plt.plot(df.s,df.F, label=label)

    plt.vlines(x=2.5,ymin=0,ymax=1,color='r',linestyles='--')
    plt.legend()
    plt.ylabel('f-score')
    plt.xlabel(r'$\sigma$')
    plt.grid(True)
    plt.show()
    # raise
    # plt.figure()
    # plt.plot(df.R,df.P,  'ro', df.R,df.P, 'k')
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.grid(True)
    # plt.show()
    plt.figure()
    for label, df in zip([ '640', '1500', 'orig'] , [df_640, df_1500, df_orig]):
        plt.plot(df.R,df.P, label=label)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.grid(True)
    plt.show()


    plt.figure()
    for label, df in zip([ '640', '1500', 'orig'] , [df_640, df_1500, df_orig]):
        plt.plot(df.s,df.exec_time, label=label)
    plt.vlines(x=2.5,ymin=0,ymax=130,color='r',linestyles='--')
    plt.ylabel('execution time (s)')
    plt.xlabel(r'$\sigma$')
    plt.grid(True)
    plt.show()


