
"""
Plotting the summary figure  from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt

def plotting_paper(dates,targets,predictions,units,boxnames,period,labels,season,train_metrics_region, 
    train_metrics_season_region, test_metrics_region, test_metrics_season_region):

    """
    Plotting the respective plot.

    Parameters:
        ds(xr.Dataset): The original dataset.
        name(str): Name of the feature we want to plot.
        regions_all(list[str]): List with the regions' coordinates.
        region_names(list[str]): The names of the regions.
        colors(list[str]): List the colors for each region. 

    Parameters:
        fig(object): Figure object.
        ax(object): Axis object.

    """

    years = np.unique(dates.year)

    targets = targets.transpose()
    predictions = predictions.transpose()
    
    targets_masked = np.ma.array(targets)
    predictions_masked = np.ma.array(predictions)

    for year in years:
        targets_masked[(np.where(dates.year==year)[0][-1])] = np.ma.masked
        predictions_masked[(np.where(dates.year==year)[0][-1])] = np.ma.masked

    names = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)']

    k,l = 0,0
    fig, ax = plt.subplots(4, 3, figsize=(17, 15), layout='constrained')

    ticks = np.arange(0,len(years)*len(labels),len(labels)/2)
    ticks = np.int16(ticks)
    labels2 = np.tile(labels,len(years))
    ax[k,l].set_xticks(ticks, labels2[ticks])

    test = np.arange(0,len(targets_masked))

    for j in np.arange (0,len(boxnames)):

        ax[k, l].plot(targets_masked[:,j])
        ax[k, l].plot(test[:ticks[-2]-1],predictions_masked[0:ticks[-2]-1,j])
        ax[k, l].plot(test[ticks[-2]:],predictions_masked[ticks[-2]:,j])
        ax[k,l].fill_between((test[0],test[ticks[2]-2]), np.max((targets_masked[:,j],predictions_masked[:,j])), np.min((targets_masked[:,j],predictions_masked[:,j])), alpha=0.3, color='grey')
        ax[k,l].fill_between((test[ticks[4]-1],test[ticks[6]-2]), np.max((targets_masked[:,j],predictions_masked[:,j])), np.min((targets_masked[:,j],predictions_masked[:,j])), alpha=0.3, color='grey')
        ax[k,l].fill_between((test[ticks[-2]-1],test[-1]), np.max((targets_masked[:,j],predictions_masked[:,j])), np.min((targets_masked[:,j],predictions_masked[:,j])), alpha=0.7, color='grey')

        ax[k, l].set_title(boxnames[j])
        
        ax[k,l].annotate(names[j], (0.02, 0.9), xycoords='axes fraction', fontsize=14)

        ax2 = ax[k, l].secondary_xaxis('bottom')
        ax2.set_xticks(ticks=np.arange(0,len(years)*len(labels),len(labels)), labels=years)
        ax2.tick_params(length=0, pad=30)

        if l == 0:
            ax[k, l].set_ylabel(units +' (no seasonality)')

        l=l+1
        if l==3:
            l=0
            k=k+1

    fig.legend(['targets', 'predictions', '2025'], ncols=3)
    fig.suptitle('Mean DPR' + ' ' + period)

    ax[k, l].plot(season)
    ax[k, l].set_xticks(ticks=np.arange(0,len(labels),len(labels)//8+1), labels=labels[np.arange(0,len(labels),len(labels)//8+1)])
    ax[k, l].set_title('Long-term seasonalities (2007-2020)')
    ax[k, l].set_ylabel(units)
    ax[k, l].legend(boxnames, ncol=2)
    ax[k,l].annotate('(j)', (0.95, 0.05), xycoords='axes fraction', fontsize=14)

    l = l + 1
    ax[k, l].plot(train_metrics_region[j]['r'], label = 'R training', marker = '.', ls = ' ', markersize=12)
    ax[k, l].plot(r_test[0], label = 'R evaluation', marker = '.', ls = ' ', markersize=12)
    ax[k, l].plot(r_test[1], label = 'R testing', marker = '.', ls = ' ', markersize=12)
    ax[k, l].plot(r_train_season, label = 'R training (no seasonality)', marker = '.', ls = ' ', markersize=12)
    ax[k, l].plot(r_test_season[0], label = 'R evaluation (no seasonality)', marker = '.', ls = '', markersize=12)
    ax[k, l].plot(r_test_season[1], label = 'R testing (no seasonality)', marker = '.', ls = '', markersize=12)
    ax[k, l].set_xticks(ticks = np.arange(0, len(boxnames)), labels=boxnames)
    ax[k, l].set_title('Correlation coefficients')
    ax[k, l].legend(ncols=2)
    ax[k,l].annotate('(k)', (0.94, 0.15), xycoords='axes fraction', fontsize=14)
    ax[k, l].set_xlabel('Sub-regions')

    l = l + 1
    ax[k, l].plot(rms_train, label = 'RMSE training', marker = '.', ls = ' ', markersize=12)
    ax[k, l].plot(rms_test[0], label = 'RMSE evaluation', marker = '.', ls = ' ', markersize=12)
    ax[k, l].plot(rms_test[1], label = 'RMSE testing', marker = '.', ls = ' ', markersize=12)
    ax[k, l].set_xticks(ticks = np.arange(0, len(boxnames)), labels=boxnames)
    ax[k, l].set_ylabel('[%]')
    ax[k, l].set_title('Root mean square errors')
    ax[k, l].legend()
    ax[k,l].annotate('(l)', (0.95, 0.05), xycoords='axes fraction', fontsize=14)
    ax[k, l].set_xlabel('Sub-regions')
    