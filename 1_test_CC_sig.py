#opt/anaconda3/env/obspy/python3.7
"""
    - determine time shift and cross-correlation coefficient
      for two time series
    - determine significance by bootstrap resampling
    - synthetic time series are created as pandas dataframes
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
np.random.seed( 123456)
#-------------------------------------------------
import src.my_crosscorr as corr

#=================================1=============================================
#                 parameters
#===============================================================================
n_t      = 60*30 # ~ 5 years
lag      = 60  # change lag between first and second time series
mu1      = 1.5 #~bbl/dy ~ 10,000 per month
mu2      = 1 #
b_random = False # just noise or noise+signal

#corrMethod = 'pearson' #or fft, pearson
nBS        = 1000
#=================================2=============================================
#                         create synthetic data
#===============================================================================
df_dy = corr.syn_tSeries( n_t, mu1, mu2, lag, random = b_random)
# simulate integration to weekly and monthly
#df_wk = df_dy.resample('W').sum() #
df_wk = corr.dyToWeek( df_dy)
df_mo = corr.dyToMonth( df_dy)

#=================================3=============================================
#                         correlation analysis
#===============================================================================
#A#global corr no shift##
r, p = stats.pearsonr( df_wk['y1'], df_wk['y2'])
print(f"Scipy - Pearson r: {round(r,2)} and p-value: {round(p,3)}")

#B# global corr - with time shift
cc_dy, shift_dy, __ = corr.do_crosscorr( df_dy, 'y1', 'y2')
cc_wk, shift_wk, __ = corr.do_crosscorr( df_wk, 'y1', 'y2')
cc_mo, shift_mo, __ = corr.do_crosscorr( df_mo, 'y1', 'y2',
                                         shift_max=int(.2*len(df_mo['y1'])))

print( 'daily: CC',   round(cc_dy,2), 'offset', shift_dy, 'true offset', lag)
print( 'weekly: CC',  round(cc_wk,2), 'offset', shift_wk, 'true offset', lag)
print( 'monthly: CC', round(cc_mo,2), 'offset', shift_mo, 'true offset', lag)
#C# align  y1 and y2
if shift_dy < 0: #shift to the left
    y2 = np.hstack( (df_dy['y2'][abs(shift_dy)::], np.ones(abs(shift_dy))*np.nan) )
else:#shift right
    y2 = np.hstack( (np.ones(shift_dy)*np.nan, df_dy['y2'][0:-shift_dy]) )

#=================================3=============================================
#                         plots
#===============================================================================
a_t  = np.arange( 0, n_t-7, 7)
f,ax = plt.subplots(3, 1, figsize=(10,10))
#----daily sampling--------------
ax[0].plot( df_dy['y1'], 'b-', lw = 3, alpha = .5, label = 'Inj')
ax[0].plot( df_dy['y2'], '-', color = '#ffa500', lw = 3, alpha =.5, label = 'Seis')
ax[0].plot(  y2, 'r-', label = 'shifted')
ax[0].set(ylabel='Amplitude',  title=f"Overall Pearson r = {np.round(r,2)}")
ax[0].legend()
# median filter
df_dy.rolling(window=30,center=True).median().plot(ax=ax[0], legend=False)
#--------weekly sampling----------
ax[1].plot( a_t, df_wk['y1'], 'b-', lw = 2, alpha = 1 )
ax[1].plot( a_t, df_wk['y2'], '-',  color = '#ffa500',lw = 2, alpha = 1 )
ax[1].set_xlim( ax[0].get_xlim())

#------------------plot shifted time seris CC-----------------------------------
a_shift = np.arange(-int(.1*len(df_dy['y1'])), int(.1*len(df_dy['y1'])) + 1)
a_CC = np.array([ corr._crosscorr(df_dy['y1'], df_dy['y2'], lag) for lag in a_shift])
# set to max of abs value to allow a-causal shift
i_max = np.argmax((a_CC))
ax[2].plot( a_shift, a_CC, 'b-')
ax[2].axvline( shift_dy,  color ='r', ls = '--', label = 'offset=%i (true dt=%i), CC=%.2f'%( shift_dy, lag, a_CC[i_max]))
ax[2].legend()
ax[2].set(xlabel='Sample',ylabel='CC',title='Seis. leads <> Inj. leads')
plt.savefig( 'plots/1_sig_per_ran_%s.png'%(b_random))

#=================================4=============================================
#                    significance from MC resampling
#===============================================================================
#-------------------weekly data-----------------------------------------
a_shift = np.arange(-int(.1*len(df_wk['y1'])), 0)
a_cc_bs = np.zeros( nBS)
for i in range( nBS):
    df_copy = df_wk.copy()
    df_copy.apply( np.random.shuffle, axis=0)
    cc_tmp     = np.array([ corr._crosscorr(df_copy['y1'], df_copy['y2'], lag) for lag in a_shift])
    a_cc_bs[i] = cc_tmp.max()
    sys.stdout.write('\r---BS (weekly): %i out of %i, CC=%.2f,'%(i+1, nBS, a_cc_bs[i]))#, end='\r')
    sys.stdout.flush()
a_cc_bs.sort()
print( 'reshuffle CC, mean=', a_cc_bs.mean(), '+/-', a_cc_bs.std())
a_cumsum = np.cumsum( np.ones( nBS))
# sign. level
sel = cc_wk < a_cc_bs
if sel.sum() == 0:
    conf_wk = 99.9
else:
    conf_wk = .1*(1000 - sel.sum())
#-------------------monthly data-----------------------------------------
a_cc_bs_mo = np.zeros( nBS)
a_shift = np.arange(-int(.2*len(df_mo['y1'])), 0)
for i in range( nBS):
    df_copy = df_mo.copy()
    df_copy.apply( np.random.shuffle, axis=0)
    cc_tmp        = np.array([ corr._crosscorr(df_copy['y1'], df_copy['y2'], lag) for lag in a_shift])
    a_cc_bs_mo[i] = cc_tmp.max()
    sys.stdout.write('\r---BS (monthly): %i out of %i, CC=%.2f, '%(i+1, nBS, a_cc_bs_mo[i]))#, end='\r')
    sys.stdout.flush()
a_cc_bs_mo.sort()
print( '  reshuffle CC, mean=', a_cc_bs_mo.mean(), '+/-', a_cc_bs_mo.std())

print('month: CC=%.2f, shift=%i, conf=%.1f'%( corr.do_crosscorr( df_mo, 'y1', 'y2', get_conf = True,
                                                                 shift_max=int(.2*len(df_mo['y1'])),
                                                                 nBS = nBS)))

a_cumsum = np.cumsum( np.ones( nBS))
# sign. level
sel = cc_mo < a_cc_bs_mo
if sel.sum() == 0:
    conf_mo = 99.9
else:
    conf_mo = .1*(1000 - sel.sum())
#---------------------significance level----------------------------------------
plt.figure(2)
ax = plt.subplot( 111)
#ax.set_title( 'CC(weekly) = %.2f, conf. level=%.1f'%(cc_wk, conf_wk))

ax.plot(  a_cc_bs, a_cumsum/nBS,    'k-',              label = 'weekly')
ax.plot(  a_cc_bs_mo, a_cumsum/nBS, '-', color = '.5', label = 'monthly')
ax.axhline( conf_wk/100,  color = 'r', ls = '--', label = 'obs(wk)')
ax.plot( [cc_wk], [conf_wk/100], 'ro', ms = 6, mew =1, mfc = 'none')
ax.axhline( conf_mo/100,  color = 'b', ls = '--', label = 'obs(mo)')
ax.plot( [cc_mo], [conf_mo/100], 'bo', ms = 6, mew =1, mfc = 'none')
ax.set_xlabel( 'CC')
ax.set_ylabel( 'Confidence')
#ax.set_xlim( 0, ax.get_xlim()[1])

ax.legend()
plt.savefig( 'plots/2_conf_cc_wk_ran_%s.png'%(b_random))
plt.show()
