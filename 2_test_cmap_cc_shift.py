"""
    - check how time shifts and cross-correlation works with pandas

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use( 'Agg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

np.random.seed( 12345)
#-------------------------------------------------
import src.my_crosscorr as corr

#=================================1=============================================
#                 parameters
#===============================================================================
n_t      = 50*30 # ~ three time series, 36 months
lag      = 10  # change lag between first and second time series
mu1      = 1.5 #~bbl/dy ~ 10,000 per month
mu2      = 1 #
b_random = False # just noise or noise+signal
acausal  = False # y2 leads y1

#corrMethod = 'pearson' #or fft, pearson
nBS        = 1000
#=================================2=============================================
#                         create synthetic data
#===============================================================================
df_dy = corr.syn_tSeries( n_t, mu1, mu2, lag, random = b_random, n_per = 12)
# simulate integration to weekly and monthly
#df_wk = df_dy.resample('W').sum() #
df_wk = corr.dyToWeek( df_dy)
df_mo = corr.dyToMonth( df_dy)

#=================================4=============================================
#                         windows, time-lagged correlation
#===============================================================================
m_CC, a_win, a_shift_pos = corr.win_crosscorr( df_wk['y1'], df_wk['y2'], #win = 10, step = 1,
                                               acausal=acausal)
a_shift_pos *= 7
print( 'no. of time win correlations:', m_CC.shape[1], a_win.shape[0])
a_cc_sum   = m_CC.sum(axis = 1)
ave_offset = a_shift_pos[np.argmax(a_cc_sum)]
ave_CC     = m_CC.mean(axis = 1)[np.argmax(a_cc_sum)]

print( 'ave. offset in cc=f(t)', ave_offset, 'cc', ave_CC)
cc_wk, shift_wk, __ = corr.do_crosscorr( df_dy, 'y1', 'y2', acausal=acausal)
print( 'weekly CC and shift:', cc_wk, shift_wk)

#================================= =============================================
#                         plots
#===============================================================================
a_t  = np.arange( 0, n_t-7, 7)
fig,ax = plt.subplots(3, 1, figsize=(12,6))
#----daily sampling--------------
ax[0].plot( df_dy['y1'], 'b-', lw = 3, alpha = .5, label = 'Inj')
ax[0].plot( df_dy['y2'], '-', color = '#ffa500', lw = 3, alpha =.5, label = 'Seis')
#ax[0].plot(  y2, 'r-', label = 'shifted')
ax[0].set(ylabel='Rate (1/dy)')
ax[0].legend()
# median filter
df_dy.rolling(window=30,center=True).median().plot(ax=ax[0], legend=False)
#--------weekly sampling----------
ax[1].plot( a_t, df_wk['y1'], 'b-', lw = 2, alpha = 1 )
ax[1].plot( a_t, df_wk['y2'], '-',  color = '#ffa500',lw = 2, alpha = 1 )
ax[1].set_xlim( ax[0].get_xlim())
ax[1].set( ylabel='Rate (1/wk)', xlabel='Time (dy)')

plot1 = ax[2].imshow( m_CC, extent=[a_t[0], a_t[-1], a_shift_pos[-1], a_shift_pos[0]],
                   cmap = plt.cm.jet)
ax[2].axhline( shift_wk,    color ='w', ls = '--',lw = 2.5, label = 'CC=%.2f'%(    cc_wk))
ax[2].axhline( ave_offset,  color ='.5', ls = '--',  label = 'ave. CC=%.2f'%( ave_CC))
axins1 = inset_axes(ax[2], width="20%",  bbox_to_anchor=( 0, -1.35, 1, 1),
                           height="25%", bbox_transform=ax[2].transAxes,)
fig.colorbar( plot1, cax = axins1, orientation = 'horizontal', label = 'CC', shrink = .3)
ax[2].set( ylabel='Lag Time (dy)', xlabel='Time (dy)', title='ave. CC=%.2f, lag time=%i dy'%( ave_CC, ave_offset))
ax[2].set_xlim( ax[0].get_xlim())
plt.savefig( 'plots/3_cmap_cc_lag.png')
plt.show()
