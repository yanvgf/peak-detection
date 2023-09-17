import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

def s1_metric(data, k):
    """S1 metric from the article (average between the maximum distance from a point to its k neighbors to the right and the maximum to the left)

    Input:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        k (int): number of neighbors to be considered (in the article, it is suggested to use a value with meaning, ex.: samples referring to 1 day)\n
    
    Output:
        s1 (pd.DataFrame): series with the values of the S1 metric for each point
    """
    
    s1 = np.zeros(data.shape[0])
    
    # For each sample in data, calculates the difference between its value and its k neighbors to the right and left
    for idx in range(data.shape[0]):
        max_right = np.max([data.iloc[idx,:].values[0] - data.iloc[idx+k_i+1,:].values[0]
                        if idx+k_i+1 < data.shape[0] else 0
                        for k_i in range(k)])
        max_left = np.max([data.iloc[idx,:].values[0] - data.iloc[idx-(k_i+1),:].values[0] 
                        if idx-(k_i+1) >= 0 else 0
                        for k_i in range(k)])
        s1[idx] = (max_right + max_left)/2
        
    return pd.DataFrame(s1)

def s2_metric(data, k):
    """S2 metric from the article (average between the average distance from a point to its k neighbors to the right and the mean to the left)

    Input:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        k (int): number of neighbors to be considered (in the article, it is suggested to use a value with meaning, ex.: samples referring to 1 day)\n
    
    Output:
        s2 (pd.DataFrame): series with the values of the S2 metric for each point
    """
    
    s2 = np.zeros(data.shape[0])
    
    # For each sample in data, calculates the difference between its value and its k neighbors to the right and left
    for idx in range(data.shape[0]):
        mean_right = np.mean([data.iloc[idx,:].values[0] - data.iloc[idx+k_i+1,:].values[0]
                            if idx+k_i+1 < data.shape[0] else 0
                            for k_i in range(k)])
        mean_left = np.mean([data.iloc[idx,:].values[0] - data.iloc[idx-(k_i+1),:].values[0] 
                            if idx-(k_i+1) >= 0 else 0
                            for k_i in range(k)])
        s2[idx] = (mean_right + mean_left)/2
        
    return pd.DataFrame(s2)

def s3_metric(data, k):
    """S3 metric from the article (average between the average distance from a point to the mid-point of its k neighbors to the right and the mean to the left)

    Input:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        k (int): number of neighbors to be considered (in the article, it is suggested to use a value with meaning, ex.: samples referring to 1 day)\n
    
    Output:
        S3 (pd.DataFrame): series with the values of the S3 metric for each point
    """
    
    s3 = np.zeros(data.shape[0])
    
    # For each sample in data, calculates the difference between its value and its k neighbors to the right and left
    for idx in range(data.shape[0]):
        diff_right = data.iloc[idx,:].values[0] - np.mean([data.iloc[idx+k_i+1,:].values[0]
                                                        if idx+k_i+1 < data.shape[0] else 0
                                                        for k_i in range(k)])
        diff_left = data.iloc[idx,:].values[0] - np.mean([data.iloc[idx-(k_i+1),:].values[0] 
                                                        if idx-(k_i+1) >= 0 else 0
                                                        for k_i in range(k)])
        s3[idx] = (diff_right + diff_left)/2
        
    return pd.DataFrame(s3)

def get_entropy(data, kde_estimator):
    """Calculates the Shannon entropy of the input data
    
    Input:
        data (pd.DataFrame): dataframe with the data to be calculated the entropy\n
        kde_estimator (sklearn.neighbors.KernelDensity): sklearn density estimator\n
    
    Output:
        entropy (float): Shannon entropy value
    """

    # Adjust estimator to data
    kde_estimator.fit(data.values)

    # Method score_samples returns the log of the probability density of the data
    log_dens = kde_estimator.score_samples(data.values)
    
    # Entropy involves estimating the probability density of the data
    entropy = -np.sum([ np.exp(log_dens) * log_dens ])
    
    return entropy

def s4_metric(data, k, bandwidth, kde_kernel='gaussian'):
    """S4 metric from the article (entropy difference in a window before and after adding a sample)
    
    Input:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        k (int): number of neighbors to be considered (in the article, it is suggested to use a value with meaning, ex.: samples referring to 1 day)\n
        bandwidth (int): width of the probability density estimation window\n
        kd_kernel (str, default='gaussian'): kernel to be used in KDE (gaussian, epanechnikov)\n
    
    Output:
        s4 (pd.DataFrame): series with the values of the S4 metric for each point
    """
    
    s4 = []
    
    # Create a KernelDensity instance
    kde_estimator = KernelDensity(bandwidth=bandwidth, kernel=kde_kernel)
    
    for idx in range(data.shape[0]):
        
        # Avoid problem of window with negative indices or greater than the size of the dataframe
        left_window_index = idx-k
        if left_window_index < 0:
            left_window_index = 0
        #
        right_window_index = idx+k+1
        if right_window_index > data.shape[0]:
            right_window_index = data.shape[0]
        
        # Calculates the entropy without the sample in the window
        entropy_before = get_entropy(pd.concat([data.iloc[left_window_index:idx,:], data.iloc[idx+1:right_window_index,:]]), kde_estimator)
        
        # Calculates the entropy with the sample in the window
        entropy_after = get_entropy(data.iloc[left_window_index:right_window_index,:], kde_estimator)
        
        s4.append(entropy_before - entropy_after)
    
    return(pd.DataFrame(s4))

def parametric_outlier_detection(data, sample, n_std=3):
    """Gets the probability of a point being an outlier in a normal distribution

    Args:
        data (pd.DataFrame): data window\n
        sample (float): sample value for which the probability of being an outlier is desired\n
        n_std (int, default=3): number of standard deviations from which a point is considered an outlier\n
        
    Output:
        is_outlier (float): "probability" of being an outlier (proportional to the rarity of the sample)
    """
    
    data = data.values.flatten()
    
    # Mean and standard deviation of the window
    mean = np.mean(data)
    std = np.std(data)
    
    # Checks if point is outlier (if standard deviation is zero, all values ​​are equal)
    is_outlier = False
    if std != 0:
        is_outlier = (sample - mean) > n_std*std
        stds_from_mean = (sample - mean)/std
    
    return(is_outlier, stds_from_mean)

def s5_metric(data, k, method='parametric', n_std=3):
    """S5 metric from the article (peaks are outliers in a local window)

    Args:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        k (int): number of neighbors to be considered (in the article, it is suggested to use a value with meaning, ex.: samples referring to 1 day)\n
        method (str, default='parametric'): outlier detection method (parametric, chebyshev)\n 
        n_std (int): number of standard deviations from which a point is considered an outlier\n
    
    Output:
        s5_outliers (pd.DataFrame): series with already detected local peaks (True=peak) (in this case, it is a direct peak detection, without assigning a spikiness)\n
        s5_spikiness (pd.DataFrame): series with the values of the S5 metric for each point (how many standard deviations the sample is from the mean in each window)
    """
    
    s5_outliers = []
    s5_spikiness = []
    
    for idx in range(data.shape[0]):
        
        # Avoid problem of window with negative indices or greater than the size of the dataframe
        left_window_index = idx-k
        if left_window_index < 0:
            left_window_index = 0
        #
        right_window_index = idx+k+1
        if right_window_index > data.shape[0]:
            right_window_index = data.shape[0]
            
        data_window = data.iloc[left_window_index:right_window_index,:]
        sample = data.iloc[idx,:].values[0]
            
        if method == 'parametric':
            is_outlier, stds_from_mean = parametric_outlier_detection(data_window, sample, n_std)
            s5_outliers.append(is_outlier)
            s5_spikiness.append(stds_from_mean)

    return(pd.DataFrame(s5_outliers), pd.DataFrame(s5_spikiness))