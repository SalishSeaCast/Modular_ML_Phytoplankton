
"""
Plotting functions.
"""

import matplotlib.pyplot as plt
import cmocean.cm as cm
import salishsea_tools.viz_tools as sa_vi
import numpy as np

def plot_regions(ds, name, regions_all, region_names, colors):

    """
    Plotting the selected regions.

    Parameters:
        ds(xr.Dataset): The original dataset.
        name(str): Name of the feature we want to plot.
        regions_all(list[str]): List with the regions' coordinates.
        region_names(list[str]): The names of the regions.
        colors(list[str]): List the colors for each region. 

    Returns:
        fig(object): Figure object.
        ax(object): Axis object.

    """

    fig, ax = plt.subplots(1, 1, figsize=(5, 9))
    mycmap = cm.deep # type: ignore
    mycmap.set_bad('grey')
    ax.pcolormesh(ds[name][0], cmap=mycmap)
    sa_vi.set_aspect(ax)

    for i in range (0, len(regions_all)):
        ax.plot([regions_all[i][2], regions_all[i][3], regions_all[i][3], regions_all[i][2], regions_all[i][2]], 
        [regions_all[i][0], regions_all[i][0], regions_all[i][1], regions_all[i][1], regions_all[i][0]], '-', color=colors[i])
    
    fig.legend(region_names)
    fig.suptitle('Regions of interest in the Salish Sea')

    return fig

def plotting_mean_values(ds_features, data, period_features, units, category, region):

    """
    Plotting the selected regions.

    Parameters:
        ds_features(dictionary): Dictionary containing dates and labels.
        data(dictionary): Dictionary containing targets and predictions.
        period_features(dictionary): Dictionary containing the period.
        units(str): The units of the targets.
        category(str)): The type of the targets.
        region(str): The region of interest.

    Returns:
        fig(object): Figure object.
    """

    years = np.unique(ds_features['dates'].year)

    fig, ax = plt.subplots(figsize=(19,5))
    
    mean_targets = np.ma.array(data['targets'])
    mean_predictions = np.ma.array(data['predictions'])

    for year in years:
        mean_targets[(np.where(ds_features['dates'].year==year)[0][-1])] = np.ma.masked
        mean_predictions[(np.where(ds_features['dates'].year==year)[0][-1])] = np.ma.masked
        
    ax.plot(mean_targets, label = 'targets')
    ax.plot(mean_predictions, label = 'predictions')

    ticks = np.arange(0,len(years)*len(ds_features['labels']),len(ds_features['labels'])/2)
    ticks = np.int16(ticks)
    labels2=np.tile(ds_features['labels'],len(years))

    ax.set_xticks(ticks, labels2[ticks])

    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(ticks=np.arange(0,len(years)*len(ds_features['labels']),len(ds_features['labels'])), labels=years)
    
    ax2.tick_params(length=0, pad=30)

    fig.suptitle('Mean '+category + ' ' +units + ' ' + period_features['period'] + ' ' + region)
    ax.legend()

    return(fig)

def plotting_seasonality(season, labels):
        
    """
    Plotting the long-term seasonality.

    Parameters:
        season(np.array[float]): The long-term seasonality.
        labels(np.array[string]): The labels of each day.

    Returns:
        fig(object): Figure object.
    """

    fig, ax = plt.subplots()

    if len(season) == len(labels): # For the whole domain
        ax.plot(season)
        fig.suptitle('Long-term seasonality (2007-2020)')

    else:
        ax.plot(season.transpose()) # Regional
        fig.suptitle('Long-term seasonality per region (2007-2020)')

    ax.set_xticks(ticks=np.arange(0,len(labels),len(labels)//8+1), labels=labels[np.arange(0,len(labels),len(labels)//8+1)])
    
    return(fig)

