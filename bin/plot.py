import sys
sys.path.append('./') # allows it to be run from parent dir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import neighbors
import os
import re
import argparse
from glob import glob

# local
carson_style = {
   'axes.titlesize' : 14,
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'font.family':'serif',
   'font.serif' : 'Computer Modern Roman',
   'ps.useafm' : True,
   'pdf.use14corefonts' : True,
   'text.usetex' : True
}


def plot_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default='./out', help="results directory")
    parser.add_argument("-p", "--plot", type=str, default='both', help="plot_type")
    return parser

def get_files(results_dir,pattern='*cr*'):
    files = glob(os.path.join(results_dir,pattern))
    return files

def filter_file_list(file_list,fstring):
    filt_files = [f for f in file_list if fstring in f]
    return filt_files

def low_pass_filter(dat,window_size=10):
    if dat.size > window_size:
        dat_filt = np.zeros(dat.size - window_size)
        for i in range(window_size):
            window = dat[i:-(window_size-i)]
            dat_filt = dat_filt + window
        dat_filt = dat_filt / window_size
        return dat_filt

    raise RuntimeError('Size Too Small: %d' % dat.size)

def plot_all_files(file_list,plot_str):
    for f in file_list:
        cr = np.load(f)
        cr_filt = low_pass_filter(cr[:,2])
        update_filt = low_pass_filter(cr[:,0])
        plt.plot(update_filt,cr_filt,plot_str)

def timeseries_dataframe(file_list):
    crs = []
    min_max = 1e20
    for f in file_list:
        cr = np.load(f)
        crs.append(cr)
        cur_max = np.amax(cr[:,0])
        min_max = min_max if min_max < cur_max else cur_max

    fitted_crs = []
    x = np.linspace(0,min_max,500)
    for i,cr in enumerate(crs):
        cr_fitted = knn_regression(cr,x)
        cr_fitted = pd.DataFrame(cr_fitted,columns=['Number of Updates','Cumulative Reward'])
        cr_fitted['Run'] = i

        fitted_crs.append(cr_fitted)
    df = pd.concat(fitted_crs,ignore_index=True)
    return df.reset_index()


def fix_duplicate_entries(df):
    df2 = df.groupby(['Number of Updates','Run','Policy'],as_index=False).mean()
    return df2


def sns_plot(cr_files,plot_out):
    df_gauss = timeseries_dataframe(cr_files)
    df_gauss['Policy'] = 'Multivariate Gaussian'

    sns_plot = sns.tsplot(data=df, time="Number of Updates", condition="Policy", unit='Run',
                            value="Cumulative Reward") #,estimator=np.nanmean,err_style='unit_traces')
    plt.title('Plot of CR')
    plt.savefig(plot_out)
    plt.close()

def knn_regression(cr,x,weights='distance',n_neighbors=8):
    x = x.reshape((-1,1))
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    model = knn.fit(cr[:,0].reshape((-1,1)),cr[:,2])
    pred = model.predict(x)
    cr_new = np.concatenate((x,pred.reshape((-1,1))),axis=1)
    return cr_new

def tseries_plot(cr_files,plot_out):
    plot_all_files(cr_files,'r-')
    plt.savefig(plot_out)
    plt.close()


if __name__ == '__main__':
    # PLOTTING CONFIG
    sns.set(context='paper',style="darkgrid",rc=carson_style)

    # SETUP
    parser = plot_argparser()
    args = parser.parse_args()
    all_cr_files = get_files(args.dir)

    if args.plot == 'tseries':
        plot_out = os.path.join(args.dir,'plot_tseries.pdf')
        tseries_plot(all_cr_files,plot_out)
    elif args.plot == 'sns':
        plot_out = os.path.join(args.dir,'plot_sns.pdf')
        sns_plot(all_cr_files,plot_out)
    elif args.plot == 'both':
        plot_out = os.path.join(args.dir,'plot_tseries.pdf')
        tseries_plot(all_cr_files,plot_out)
        plot_out = os.path.join(args.dir,'plot_sns.pdf')
        sns_plot(all_cr_files,plot_out)
    else:
        raise RuntimeError('Plot: %s, is not valid' % args.plot)