#python3.7
"""
    - utility fct to cross-correlate time series
      with different length, sampling etc.
    - determine significance of cross-correlations

"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import obspy.signal.cross_correlation as obsCorr # correlate, xcorr_max
#=================================1=============================================
#                  synthetic data
#===============================================================================
def syn_tSeries( N, mu1, mu2, lag, random = False, **kwargs):
    """
        - create two random time series with 50% signal 50% noise in the first
               and 100% signal in the second
    Parameter
    ---------------------
     N:   int
         - length of time series
     mu1: float
            mean 1. time series
     mu2: float
            mean 2. time series
     lag: int
            lag between 1. and 2. - in no. of samples
    :return:
    """
    n_per = 4 # n periods in t1 and t2
    if 'n_per' in kwargs.keys() and kwargs['n_per'] is not None:
        n_per = kwargs['n_per']
    # create synthetic data
    if random == True:
        y1 = mu1 + np.random.uniform(-.5, .5, int(N))
        y2 = mu2 + np.random.uniform(-.5, .5, int(N))
    else:
        y1 = mu1 + np.cos(np.linspace(0, n_per * 2 * np.pi, N )) + np.random.uniform(-.5, .5, int(N))
        # 2. time series
        # only 50% of y1 contains signal, the rest is noise, mean of uniform = .5
        y2 = mu2 + np.cos(np.linspace(0, .5 * n_per * 2*np.pi, int(0.5 * N))) + np.random.uniform(-.5, .5, int( 0.5 * N))
        # add time shift
        y2 = np.hstack(( np.random.uniform( -.5, .5, lag), y2[0:-lag]))
        # add noise at the end
        y2 = np.hstack((y2, np.random.uniform( -.5, .5, int(0.5 * N))))
    # create time stamps relative to 1/1/2010
    #pd_date = pd.date_range(start='1/1/2010', periods=N, freq='D')
    return pd.DataFrame( {'y1': y1, 'y2': y2})#, index = pd_date)

def dyToWeek( df):
    """
    - switch daily time sampling to weekly
    Parameter
    ------------
    df['y1'], df['y2'] - time series with weekly sampling
    :return:
    df_weekly
    """
    n = df.shape[0]
    y1_wk = np.array([df['y1'][i * 7:(i + 1) * 7].sum() for i in range(int(n / 7.))])
    y2_wk = np.array([df['y2'][i * 7:(i + 1) * 7].sum() for i in range(int(n / 7.))])
    return pd.DataFrame( {'y1':y1_wk, 'y2':y2_wk})

def dyToMonth( df):
    n = df.shape[0]
    # #C# monthly sampling
    y1_mo = np.array([df['y1'][i * 30:(i + 1) * 30].sum() for i in range(int(n / 30.))])
    y2_mo = np.array([df['y2'][i * 30:(i + 1) * 30].sum() for i in range(int(n / 30.))])
    return pd.DataFrame( {'y1':y1_mo, 'y2':y2_mo})
#=================================2=============================================
#                        cross-correlations
#===============================================================================
def do_crosscorr( df, tag1, tag2, return_vector=False, **kwargs):
    """
    - compute max. correlatiom coefficient and times shift
       between two pandas series specified by tag1 and tag2
    - assume tag2 leads tag1, use acausal = True to allow pos. and neg. shifts
      to align t1 and t2

    Parameter
    ----------
     df:      type pandas data frame
              contains two Series: tag1 and tag2
     tag1:    type string
              - strings specifying the two Pd Sereis
     tag2:    string
     kwargs:
            shift_max  - int, default .1*len(df[tag])
            acausal    - bool, default = False
                         allow shift in pos. and neg. direction
            return_vector - boolean, default = False
                          return  a_CC, a_shift, i_max
            get_conf   = boolean, default = False
                         compute confidence level via bootstrapp resampling
            nBS        - int, default = 1000
                         number of bootstraps for confidence
    :return:
    """
    N         = len( df[tag1])
    shift_max = int(.1 * N)
    conf      = 0. # confidence, float
    if 'shift_max' in kwargs.keys() and kwargs['shift_max'] is not None:
        shift_max = kwargs['shift_max']
    if 'acausal' in kwargs.keys() and kwargs['acausal'] == True:
        a_shift   = np.arange(-int(shift_max), int(shift_max + 1))
    else: # assume tag2 leads tag1
        a_shift = np.arange(-int(shift_max), 0)
    a_CC      = np.array([ _crosscorr(df[tag1], df[tag2], lag, **kwargs) for lag in a_shift])
    # set to max of abs value to allow a-causal shift
    i_max = np.argmax((a_CC))
    #----------------determine confidence level from BS---------------------------------
    if 'get_conf' in kwargs.keys() and kwargs['get_conf'] == True:
        a_cc_bs = _conf_level( df, tag1, tag2, a_shift, **kwargs)
        # sign. level
        sel = a_CC[i_max] < a_cc_bs
        if sel.sum() == 0:
            conf = 99.9
        else:
            conf = .1 * (1000 - sel.sum())
        if 'plotConf' in kwargs.keys() and kwargs['plotConf'] == True:
            nBS = len( a_cc_bs)
            a_cumsum = np.cumsum( np.ones( nBS))
            plt.figure(10)
            plt.clf()
            ax = plt.subplot(111)
            # ax.set_title( 'CC(weekly) = %.2f, conf. level=%.1f'%(cc_wk, conf_wk))
            ax.plot(a_cc_bs, a_cumsum / nBS, 'k-', label='bootstrap')
            ax.axhline(conf / 100, color='r', ls='--', label='observed')
            ax.plot([a_CC[i_max]], [conf / 100], 'ro', ms=6, mew=1, mfc='none')
            ax.set_xlabel('CC')
            ax.set_ylabel('Confidence')
            # ax.set_xlim( 0, ax.get_xlim()[1])
            ax.legend()

    return a_CC[i_max], a_shift[i_max], conf

def _conf_level( df, tag1, tag2, a_shift, **kwargs):
    """
        compute confidence level from bootstrap resampling
    Parameter
    ----------
            df   - pd dataframe
                 - contains two time series specified by tag1 and tag2
            tag1, tag2,
            a_shift - np.array
                       range of permissible time shifts
            kwargs -
                  nBS   - int detaulf = 1000
    :return: - f_conf- float
                     - confidence level between 0 and 100
             - a_cc  - np.array
                       sorted CC values from boostrap resampling
    """
    nBS = 1000
    if 'nBS' in kwargs.keys() and kwargs['nBS'] is not None:
        nBS = kwargs['nBS']
    a_cc_bs = np.zeros(nBS)
    for i in range(nBS):
        df_copy = df.copy()
        df_copy.apply(np.random.shuffle, axis=0)
        a_CC        = np.array([_crosscorr( df_copy[tag1], df_copy[tag2], lag, **kwargs) for lag in a_shift])
        a_cc_bs[i]  = a_CC.max()
        sys.stdout.write('\r---BS resampling: %i out of %i, CC=%.2f,' % (i + 1, nBS, a_cc_bs[i]))  # , end='\r')
        sys.stdout.flush()
    a_cc_bs.sort()
    print('reshuffle CC, mean=', a_cc_bs.mean(), '+/-', a_cc_bs.std())
    return a_cc_bs

def _crosscorr(df_y1, df_y2, shift=0, **kwargs):
    """
        - correlate x and y(shifted by lag in samples)
        - several methods are implemented as detailed below
        - use: do_crosscorr
    Parameters
    ----------
            datax, datay : pandas.Series
                           have to of equal length
            shift        : int, default = 0
                           - relative shift of y , pos or neg
            method       :  str, default='pearson'
                           {‘pearson’, ‘kendall’, ‘spearman’}
                            'fft' -
    Returns
    ----------
    crosscorr : float
    """
    method = 'pearson'
    if 'method' in kwargs.keys() and kwargs['method'] is not None:
        method = kwargs['method']
    y2 = df_y2.shift(shift)
    if method == 'fft': #uses obspy which presumably chosen between direct of fft
        y2 = y2.values
        y1 = df_y1.values[abs(y2) > 0]
        y2 = y2[abs(y2) > 0]
        fct_cc = obsCorr.correlate( y1, y2, shift = 0)
        cc = obsCorr.xcorr_max( fct_cc)[1]
    else:
        cc = df_y1.corr( y2, method = method)
    return cc

def win_crosscorr( df_y1, df_y2, **kwargs):
    """
    - windowed time lagged cross-correlation
    - split time series into equal windows and perform
      cross-correlation and shift for each subsample
    - win size for each individual correlation win is determined from:
        np.sqrt( df_y1.shape[0])
        or can be set directly: win = int
        shift_max = int(.5*(win)
    Parameters
    --------------
    df_y1, df_y - Pandas series,
                  two time series of equal length, equal sampling
    kwargs: 'win'  - int, default = int( np.sqrt(len(y)))
                     moving window size used for correlation
            'step' - int, default = int( .1*win)
                     step between successive windows, smallest values = 1

    :return:
           m_CC   - cross-correlation coefficient for each subwindow
                    size: n_win x 2*shift_max
           a_win  - start ID of forward windows used to calculate CC
           a_shift- range of explored time shifts
    """
    n    = len( df_y1)
    win  = int( np.sqrt( n))
    if win < 10:
        win = 10 # set minimum window and step size
    step = int( .1*win)
    if 'win' in kwargs.keys() and kwargs['win'] is not None:
        win = kwargs['win']
    if 'step' in kwargs.keys() and kwargs['step'] is not None:
        step = kwargs['step']
    shift_max = int(.8*win)
    if 'acausal' in kwargs.keys() and kwargs['acausal'] == True:
        print( 'acausal = True, allow for negative shift between y1 and y2')
        a_dt   = np.arange(-int(shift_max), int(shift_max + 1))
    else: #y2 leads y1
        a_dt   = np.arange(-int(shift_max), 0)
    a_win     = np.arange( 0, n - win, step)
    m_CC      = np.zeros( (len( a_dt), len(a_win)))
    for i in range( a_win.shape[0]):
        i1, i2 = a_win[i], a_win[i]+win
        d1 = df_y1.loc[i1:i2]
        d2 = df_y2.loc[i1:i2]
        #print('windows CC:', i1, i2, n)
        m_CC[:,i] = np.array([_crosscorr(d1, d2, lag) for lag in a_dt])
    return m_CC, a_win, a_dt


#=================================3=============================================
#                     data prep.
#===============================================================================
def demean( df, l_tags):
    for tag in l_tags:
        df[tag] = df[tag] - df[tag].values.mean()

def run_demean( df, l_tags, win = 6, **kwargs):
    """
    - subtract running mean
    Parameter
    ---------
             df - pandas dataframe
                - with two time series: 'y1' and 'y2'
            l_tags - list with strings
                   - define which Series in Pd should be demeaned
             win  - int default = 6
                  forward window used to compute running mean
            kwargs:
            end  = str, default = 'last_mean', i.e. demean last few samples with previous mean
                 - 'last_mean'
                 - 'constant'  - set last few sample to constant value
                 - 'nan'       - set to nan
    :return: df - with demeaned time series y1 and y2
    """
    end = 'last_mean'
    if 'end' in kwargs.keys() and kwargs['end'] is not None:
        end = kwargs['end']
    for tag in l_tags:
        n = len( df[tag])
        for i in range( n - win):
            df[tag][i] = df[tag][i] - df[tag][i:i+win].values.mean()
        # patch last win-size samples
        if end == 'last_mean':
            df[tag][-win::] = df[tag][-win::] - df[tag][i:i + win].mean()
        elif end == 'constant':
            df[tag][-win::] = df[tag][-win-1]
        elif end == 'nan' or end == 'NaN' or end == 'Nan':
            df[tag][-win::] = np.nan
#=================================4=============================================
#                           plots
#===============================================================================








