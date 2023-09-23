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
    
    data = data.to_numpy()
    
    s1 = np.zeros(data.shape[0])
    
    # For each sample in data, calculates the difference between its value and its k neighbors to the right and left
    for idx in range(data.shape[0]):
        max_right = np.max([(data[idx] - data[idx+k_i+1])[0]
                        if idx+k_i+1 < data.shape[0] else 0
                        for k_i in range(k)], axis=0)
        max_left = np.max([(data[idx] - data[idx-(k_i+1)] )[0]
                        if idx-(k_i+1) >= 0 else 0
                        for k_i in range(k)], axis=0)
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
    
    data = data.to_numpy()
    
    s2 = np.zeros(data.shape[0])
   
    # For each sample in data, calculates the difference between its value and its k neighbors to the right and left
    for idx in range(data.shape[0]):
        mean_right = np.mean([(data[idx] - data[idx+k_i+1])[0]
                            if idx+k_i+1 < data.shape[0] else 0
                            for k_i in range(k)])
        mean_left = np.mean([(data[idx] - data[idx-(k_i+1)])[0]
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


def silverman_radius(data):
    """Calculates the bandwidth of the KDE using the Silverman's rule of thumb

    Input:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        
    Output:
        bandwidth (float): bandwidth of the KDE
    """
    
    data = data.values.flatten()
    
    data_std = np.std(data)
    N = data.shape[0]
    
    # Silverman rule of thumb
    bandwidth = 1.06*data_std*N**(-1/5)
    
    return bandwidth
    
def gaussian_kernel(x):
    """Gaussian kernel

    Input:
        x (float): point to be evaluated in the kernel

    Output:
        gaussian (float): kernel value for the point x
    """

    gaussian = (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)
    return gaussian

def get_kde(data, sample_index, w):    
    """Estimates the probability density of the input data using the Parzen window
    
    Input:
        data (pd.DataFrame): dataframe with the data to be estimated the probability function
        sample_index (int): index of the sample to be estimated the probability density
        w (int): parameter for calculating the KDE bandwidth
    
    Output:
        density_estimation (float): estimation of the probability density of the input data
    """
    
    data = data.values.flatten()

    kernel_estimator = gaussian_kernel

    # Estimates the probability density using the Parzen window
    #
    M = data.shape[0]
    # Parzen window length
    h = np.sqrt(((data[sample_index] - data[sample_index+w])**2 + w**2))
    normalizing_factor = 1/(M*h)  
    kernel_factor = np.sum(kernel_estimator((data[sample_index] - data[:-w])/h))  
    density_estimation = normalizing_factor*kernel_factor  

    return density_estimation

def get_entropy(window, w):
    """Calculates the Shannon entropy of the input data
    
    Input:
        window (pd.DataFrame): dataframe with the data to calculate the entropy
        w (int): parameter for calculating the KDE bandwidth
        
    Output:
        entropy (float): Shannon entropy value
    """    

    entropy = -np.sum([get_kde(window, sample_index, w)*np.log2(get_kde(window, sample_index, w)) for sample_index in np.arange(0, window.shape[0]-w, 1)])

    return entropy


def s4_metric(data, k, w):
    """S4 metric from the article (entropy difference in a window before and after adding a sample)
    
    Input:
        data (pd.DataFrame): dataframe with the data to be classified as peak or not\n
        k (int): number of neighbors to be considered (in the article, it is suggested to use a value with meaning, ex.: samples referring to 1 day)\n
        w (int): parameter used to calculate the bandwidth\n    
        
    Output:
        s4 (pd.DataFrame): series with the values of the S4 metric for each point
    """
    
    s4 = []
    
    for idx in range(data.shape[0]):
        
        # Avoid problem of window with negative indices or greater than the size of the dataframe
        left_window_index = idx-k
        if left_window_index < 0:
            left_window_index = 0
        #
        right_window_index = idx+k+1+w
        if right_window_index > data.shape[0]:
            right_window_index = data.shape[0]
        
        # Calculates the entropy without the sample in the window
        entropy_before = get_entropy(pd.concat([data.iloc[left_window_index:idx,:], data.iloc[idx+1:right_window_index,:]]), w)
        
        # Calculates the entropy with the sample in the window
        entropy_after = entropy_before + get_entropy(data.iloc[idx:idx+1+w,:], w)
        
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

def delete_adjacent_peaks(peak_indexes, data, k):
    """Deletes adjacent peaks if the distance between them is less than the size of the window k.
    
    Input:
        peak_indexes (pd.DataFrame): pd.DataFrame containing a list of peak indexes
        data (pd.DataFrame): pd.DataFrame containing the time series
        k (int): size of the window
    
    Output:
        corrected_peak_indexes (pd.DataFrame): pd.DataFrame containing the list of peak indexes without adjacent peaks
    """
    
    corrected_peak_indexes = peak_indexes.values.flatten()

    # Initial conditions
    idx = 0
    corrected_peak_length = corrected_peak_indexes.shape[0]
    min_diff = np.min(np.diff(corrected_peak_indexes, 1))

    # While the minimum difference between peaks is less than the size of the window, delete one of the adjacent peaks
    while min_diff < k:
        
        while idx < corrected_peak_length-1:

            current_peak = corrected_peak_indexes[idx]
            next_peak = corrected_peak_indexes[idx+1]
            
            if next_peak - current_peak < k: 
                
                smaller_peak_idx = np.argmin([data.values[current_peak], data.values[next_peak]])
                
                # Deletes the smaller peak
                corrected_peak_indexes = np.delete(corrected_peak_indexes, idx+smaller_peak_idx)
            
            corrected_peak_length = corrected_peak_indexes.shape[0]
            
            idx+=1
        
        # Reset the initial conditions
        idx = 0
        corrected_peak_length = corrected_peak_indexes.shape[0]
        min_diff = np.min(np.diff(corrected_peak_indexes, 1))

    return pd.DataFrame(corrected_peak_indexes)