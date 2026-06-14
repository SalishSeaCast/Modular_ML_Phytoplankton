
"""
Utilization tools for various processes.
"""

import numpy as np

def period_identify(path: str):

    """
    Identify the period and months of the data.

    Parameters:
        path (str): Path to dataset.

    Returns:
        period(str): The period of the dataset
        id(str): Identifier for the period (needed for saving)
        months(list[str]): The months included in the period
    """

    if path[35:42] == 'jan_mar': # 75 days, 1st period
        period = '(16 Jan - 31 Mar)'
        id = '1'
        months = ['January', 'February', 'March']

    elif path[35:42] == 'jan_apr': # 120 days, 2nd period
        period = '(01 Jan - 30 Apr)'
        id = '2'
        months = ['January', 'February', 'March', 'April']

    elif path[35:42] == 'feb_apr': # 75 days, 3rd period
        period = '(15 Feb - 30 Apr)'
        id = '3'
        months = ['February', 'March', 'April']

    elif path[35:42] == 'apr_jun': # 76 days, 4th period
        period = '(16 Apr - 30 Jun)'
        id = '4'
        months = ['April', 'May', 'June']

    elif path[35:42] == 'may_sep': # 153 days, 5th period
        period = '(01 May - 30 Sep)'
        id = '5'
        months = ['May', 'June', 'July', 'August', 'September']

    return {'period':period, 'id':id, 'months':months}

def seasonality(targets, dates, region_features, variable):

    """
    Calculate the long-term seasonality.

    Parameters:
        targets(np.array[float]: The target dataset.
        dates(pd.Datetime): The dates of one year.
        region_features(dict): Dictionary containing region features.
        variable(xr.Dataset[float]: The target dataset.

    Returns:
        season(np.array[float]): The long-term seasonality.
        season_broad(np.array[float]): The long-term seasonality, broadcasted to all years.
        season_region(np.array[float]): The long-term seasonality per region.
        season_region_broad(np.array[float]): The long-term seasonality per region, broadcasted to all years.
    """

    # Splitting the season in years.
    season = np.array(np.split(targets,len(np.unique(dates.year)),axis=0))

    # Taking the mean.
    season = np.mean(season, axis=0)

    # Re-constructing it for all years.
    season_broad = np.tile(season,len(np.unique(dates.year)))

    # Now, we do it for each region individually.

    # Defining the variable.
    season_region = np.full((len(region_features['names']), len(season)), np.nan)
                        
    for i in range (len(region_features['names'])):

        # Isolating each region.
        temp = variable[:, region_features['corners'][i][0]:region_features['corners'][i][1], region_features['corners'][i][2]:region_features['corners'][i][3]].to_numpy()

        temp = temp.reshape(len(np.unique(dates.year)), len(season),-1) # Reshaping it to a proper format (14,75,...)
        season_region[i] = np.nanmean(temp, (0,2)) # Taking the mean, leaving 75 values.

     # Re-constructing it for all years.
    season_region_broad = np.tile(season_region,len(np.unique(dates.year))) 

    return {'season':season, 'season_broadcasted':season_broad, 'season_regional':season_region, 'season_regional_broadcasted':season_region_broad}
