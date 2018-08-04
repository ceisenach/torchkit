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

###############################################
# CONFIGURE DATA FORMAT
# CHANGE THIS IF NEEDED
###############################################
REWARD_COLUMN = 5
REWARD_NAME = 'Cumulative Discounted Reward'
TIME_COLUMN = 0
TIME_NAME = 'Number of Updates'
CONDITION_NAME = 'Algorithm'

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
    parser.add_argument('-d','--dirs', nargs='+', help='<Required> Output directories', required=True)
    parser.add_argument('-n','--names', nargs='+', help='Names for each experiment')
    parser.add_argument("-t", "--title", type=str, default='Comparison Plot', help='Title for the Plot')
    parser.add_argument("-p", "--plot", type=str, default='both', help="plot_type")
    parser.add_argument("-o", "--odir", type=str, default='out', help="out_directory")
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

def plot_all_files(file_list,color,marker):
    for f in file_list:
        cr = np.load(f)
        cr_filt = low_pass_filter(cr[:,REWARD_COLUMN])
        update_filt = low_pass_filter(cr[:,TIME_COLUMN])
        plt.plot(update_filt,cr_filt,color=color,marker=marker)

def timeseries_dataframe(file_list,algorithm_name):
    crs = []
    min_max = 1e20
    for f in file_list:
        cr = np.load(f)
        crs.append(cr)
        cur_max = np.amax(cr[:,TIME_COLUMN])
        min_max = min_max if min_max < cur_max else cur_max

    fitted_crs = []
    x = np.linspace(0,min_max,500)
    for i,cr in enumerate(crs):
        cr_fitted = knn_regression(cr,x)
        cr_fitted = pd.DataFrame(cr_fitted,columns=[TIME_NAME,REWARD_NAME]) # change this if desired
        cr_fitted['Run'] = i
        fitted_crs.append(cr_fitted)

    df = pd.concat(fitted_crs,ignore_index=True)
    df = df.reset_index()
    df[CONDITION_NAME] = algorithm_name
    return df


def fix_duplicate_entries(df):
    df2 = df.groupby([TIME_NAME,'Run',CONDITION_NAME],as_index=False).mean()
    return df2


def knn_regression(cr,x,weights='distance',n_neighbors=8):
    x = x.reshape((-1,1))
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    model = knn.fit(cr[:,TIME_COLUMN].reshape((-1,1)),cr[:,REWARD_COLUMN])
    pred = model.predict(x)
    cr_new = np.concatenate((x,pred.reshape((-1,1))),axis=1)
    return cr_new


def sns_plot(all_cr_files,name_list,title,plot_out):
    dfs = [timeseries_dataframe(crf,name) for crf,name in zip(all_cr_files,name_list)]
    df = pd.concat(dfs,ignore_index=True)
    sns_plot = sns.tsplot(data=df, time=TIME_NAME, condition=CONDITION_NAME, unit='Run', value=REWARD_NAME)
    plt.title(title)
    plt.savefig(plot_out)
    plt.close()

def tseries_plot(cr_files,plot_out):
    palette_iter = iter(sns.color_palette())
    for crf in cr_files:        
        plot_all_files(cr_files,next(palette_iter),'-')
    plt.savefig(plot_out)
    plt.close()

if __name__ == '__main__':
    # PLOTTING CONFIG
    sns.set(context='paper',style="darkgrid",rc=carson_style)

    # SETUP
    parser = plot_argparser()
    args = parser.parse_args()
    dir_list = args.dirs
    name_list = args.names
    name_list = name_list if name_list is not None else ['Experiment %d' % i for i in range(len(dir_list))]
    assert len(name_list) == len(dir_list), "Need to specify same length directory and names lists"


    all_cr_files = [get_files(d,'*results*') for d in dir_list]

    if args.plot == 'tseries':
        plot_out = os.path.join(args.odir,'plot_tseries.pdf')
        tseries_plot(all_cr_files,plot_out)
    elif args.plot == 'sns':
        plot_out = os.path.join(args.odir,'plot_sns.pdf')
        sns_plot(all_cr_files,name_list,args.title,plot_out)
    elif args.plot == 'both':
        plot_out = os.path.join(args.odir,'plot_tseries.pdf')
        tseries_plot(all_cr_files,plot_out)
        plot_out = os.path.join(args.odir,'plot_sns.pdf')
        sns_plot(all_cr_files,name_list,plot_out)
    else:
        raise RuntimeError('Plot: %s, is not valid' % args.plot)