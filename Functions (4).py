#!/usr/bin/env python
# coding: utf-8

# Loading in packages

# In[1]:


### Importing libraries

import xarray as xr
import intake
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import datetime
from netCDF4 import Dataset


# removing noise function

# In[ ]:


def noise_reduce (filter_array, variable_array, cond): ### Function to remove noise from the given lidar dataset
    merge_array = xr.merge(objects = (filter_array, variable_array)) ### merging both arrays
    reduced_array = xr.where(cond, variable_array, np.nan) ### xarray where method. results in a dataset where data that doe not fulfill the condition is removed
    print(reduced_array)
    return reduced_array


# Resample function

# In[ ]:


def wind_resample(divergence_data, lidar_data, time_data, dist, wind_data=None):  ### Function to sample given data to match length and time scale of each timestep. ### parameters,  divergence_data: input ascat dataset, lidar_data: input lidar dataset, time_data: input the measurement_time variable dataset from ascact data, wind_flag: 0 if using mean wind speed; 1 if using variable wind speed, wind_data: wind_speed dataset if using variable wind speed, dist: distance of the study area 160000 if total, 80000 if half.
    
    ### removing timesteps that throw an error
    #time_data[67] = np.nan
    #time_data[115] = np.nan
    #time_data[117] = np.nan
    ### Creating new dataset variable for mean values
    mr_mean = xr.DataArray(np.ones((divergence_data['time'].size, lidar_data['alt'].size)),
                                    coords={'time':divergence_data['time'], 'alt':lidar_data['alt']}, name='mr')
    
    dist = dist
    for i in range(len(time_data)):  ### for loop: every time an overpass is found in the time dataset the for loop will calculate the time interval over which the mean mixing ratio has to be taken and the takes the mixing ratio sample and stores it in the new mr_mean dataset.
        
        print(i)
        
        ti = time_data[i]
        ### if no wind speed data can be found mean wind speed for entire dataset is used
        dt = np.timedelta64(14550000000000, 'ns')
        
        if  wind_data [i] != None and np.isnat(time_data[i]) == False:    ### if mean wind speed is used and an error pops up, remove '[i]' from wind_data
            plus_time = np.timedelta64(int(3.6e+12), 'ns')
            
            if wind_data[i] != np.nan:           
                wind_sp = wind_data.sel(time=slice(ti-plus_time, ti+plus_time)).mean(dim='time')                
                time_delta = dist/wind_sp*10**9
                
                if np.isnan(time_delta) == False:  ### if no wind speed is measured, time_delta becomes nan throwing an error in assigning dt
                    dt = np.timedelta64(int(time_delta), 'ns')
                    
            
        
            
         
        
        
        tstart = ti - dt
        tend = ti + dt                                  ### Defining time interval, starting time of slice and end time of slice
        print(dt)
        
        mr_mean[i,:] = lidar_data.sel(time=slice(tstart,tend)).mean('time').data
        print(mr_mean[i,:])
        
            
       
    
    combined_array = xr.combine_by_coords(data_objects =[mr_mean, divergence_data])
    ### The new resampled dataset is combined with the divergence dataset so they can be compared
    return combined_array


# Saving .nc file function

# In[1]:


def save_file (filename, file):          ### Function to create netCDF file from a given dataset file

    filename = filename
    file.to_netcdf(filename, 'w')
    return


# Q1-Q4 plotting

# In[6]:


def fluctuation_plots (file, season=None):       ### Function to extract data from netCDF file, calculate Q1-Q4 and their fluctuations and make plots of Q1-Q4
    
    filename = xr.open_dataset(file).drop_vars(names=('wind_speed', 'measurement_time'), errors='ignore') ### Two unnecessary variables are removed from the dataset
    

    
    ### Quantile values for quantile groups are calculated as well as the total mean of the dataset
    total_mean = filename.mean(dim='time', skipna=True)
    Q0 = filename.quantile(q=0)
    Q1 = filename.quantile(q=0.25)
    Q2 = filename.quantile(q=0.5)
    Q3 = filename.quantile(q=0.75)
    Q4 = filename.quantile(q=1)


    ### Using the calculated quantiles, the entire dataset is grouped into 4 25% groups with increasing divergence
    bins = [float(Q0['wind_divergence']), float(Q1['wind_divergence']), float(Q2['wind_divergence']), float(Q3['wind_divergence']), float(Q4['wind_divergence'])]

    bin_labels = ['Q1, convergence', 'Q2, slight convergence', 'Q3, slight divergence', 'Q4, divergence']
    divergence_bingroup = filename.groupby_bins(filename['wind_divergence'], bins, labels = bin_labels, restore_coord_dims=True)


    Q1_bin = divergence_bingroup['Q1, convergence']
    Q2_bin = divergence_bingroup['Q2, slight convergence']
    Q3_bin = divergence_bingroup['Q3, slight divergence']
    Q4_bin = divergence_bingroup['Q4, divergence']
    

    ### The mean for each quantile is calculated
    Q1_bin_mean = Q1_bin.mean(dim='time', skipna=True)
    Q2_bin_mean = Q2_bin.mean(dim='time', skipna=True)
    Q3_bin_mean = Q3_bin.mean(dim='time', skipna=True)
    Q4_bin_mean = Q4_bin.mean(dim='time', skipna=True)
    
    ### removing unnecessary values
    Q1_bin = Q1_bin.drop_vars(names=('wind_speed', 'measurement_time', 'wind_divergence'), errors='ignore').dropna(dim='time', how='all')
    print(Q1_bin)
    Q2_bin = Q2_bin.drop_vars(names=('wind_speed', 'measurement_time', 'wind_divergence'), errors='ignore').dropna(dim='time', how='all')
    Q3_bin = Q3_bin.drop_vars(names=('wind_speed', 'measurement_time', 'wind_divergence'), errors='ignore').dropna(dim='time', how='all')
    Q4_bin = Q4_bin.drop_vars(names=('wind_speed', 'measurement_time', 'wind_divergence'), errors='ignore').dropna(dim='time', how='all')
    
    
    
    ### The difference between each sample in the quantile and the mean of the quantile is calculated and converted to a data array
    
    Q1_bin_dev =  Q1_bin - Q1_bin_mean
    Q2_bin_dev =  Q2_bin - Q2_bin_mean
    Q3_bin_dev =  Q3_bin - Q3_bin_mean
    Q4_bin_dev =  Q4_bin - Q4_bin_mean

    Q1_bin_dev = Q1_bin_dev.to_dataarray()
    Q2_bin_dev = Q2_bin_dev.to_dataarray()
    Q3_bin_dev = Q3_bin_dev.to_dataarray()
    Q4_bin_dev = Q4_bin_dev.to_dataarray()
  

    ### The standard deviation of each quantile is calculated
    Q1_bin_dev_std = np.std(Q1_bin_dev, axis=1)
    Q2_bin_dev_std = np.std(Q2_bin_dev, axis=1)
    Q3_bin_dev_std = np.std(Q3_bin_dev, axis=1)
    Q4_bin_dev_std = np.std(Q4_bin_dev, axis=1)

   
    ### The fluctuation of the mean of each quantile to the mean of the entire dataset is calculated and plotted
    Q1_bin_mean_fluct = (Q1_bin_mean - total_mean).drop_vars(names='wind_divergence', errors='ignore')
    Q2_bin_mean_fluct = (Q2_bin_mean - total_mean).drop_vars(names='wind_divergence', errors='ignore')
    Q3_bin_mean_fluct = (Q3_bin_mean - total_mean).drop_vars(names='wind_divergence', errors='ignore')
    Q4_bin_mean_fluct = (Q4_bin_mean - total_mean).drop_vars(names='wind_divergence', errors='ignore')


    plt.plot(Q1_bin_mean_fluct['mr'], Q1_bin_mean_fluct['alt'], 'b-', label='Q1')
    plt.plot(Q2_bin_mean_fluct['mr'], Q2_bin_mean_fluct['alt'], 'y-', label='Q2')
    plt.plot(Q3_bin_mean_fluct['mr'], Q3_bin_mean_fluct['alt'], 'g-', label='Q3')
    plt.plot(Q4_bin_mean_fluct['mr'], Q4_bin_mean_fluct['alt'], 'r-', label='Q4')
    plt.ylabel('Height (m)')
    plt.xlabel("q' (kg/kg)")
    plt.legend()
    plt.show()

    ### The fluctuation of the mean of each quantile to the mean of the entire dataset and their standard deviaion is plotted. a value of q'=0 kg/kg indicates the mean mr of the dataset Only Q1 and Q4 are turned on. To also include Q2 and Q3 remove the hashtags in lines 188-192.
    
    plt.plot(Q1_bin_mean_fluct['mr'], Q1_bin_mean_fluct['alt'], 'b-', label='Q1')
    plt.fill_betweenx(Q1_bin_mean_fluct['alt'], Q1_bin_mean_fluct['mr']-Q1_bin_dev_std[0], Q1_bin_mean_fluct['mr']+Q1_bin_dev_std[0], alpha=0.3)

    #plt.plot(Q2_bin_mean_fluct['mr'], Q2_bin_mean_fluct['alt'], 'g-', label='Q2')
    #plt.fill_betweenx(Q2_bin_mean_fluct['alt'], Q2_bin_mean_fluct['mr']-Q2_bin_dev_std[1], Q2_bin_mean_fluct['mr']+Q2_bin_dev_std[1], alpha=0.3)

    #plt.plot(Q3_bin_mean_fluct['mr'], Q3_bin_mean_fluct['alt'], 'y-', label='Q3')
    #plt.fill_betweenx(Q3_bin_mean_fluct['alt'], Q3_bin_mean_fluct['mr']-Q3_bin_dev_std[1], Q3_bin_mean_fluct['mr']+Q3_bin_dev_std[1], alpha=0.3)


    plt.plot(Q4_bin_mean_fluct['mr'], Q4_bin_mean_fluct['alt'], 'r-', label='Q4')
    plt.fill_betweenx(Q4_bin_mean_fluct['alt'], Q4_bin_mean_fluct['mr']-Q4_bin_dev_std[0], Q4_bin_mean_fluct['mr']+Q4_bin_dev_std[0], alpha=0.3)
    
    plt.title(f"Q1 and Q4 with standard deviation {season}")
    plt.ylabel('Height (m)')
    plt.xlabel("q' (kg/kg)")
    plt.legend()
    plt.show()

    # The quantiles are returned so additional calculations could be done.
    return Q1_bin, Q2_bin, Q3_bin, Q4_bin
    


# Seasonal comparison plots

# In[10]:


def seasons(file):  ### Function to divide dataset into seasons
    ds = xr.open_dataset(file) ### opening dataset
    
    
    ### dividing the dataset into months
    ds_month_goups = ds.groupby('time.month').groups
    
    DJF_idxs  = ds_month_goups[12] + ds_month_goups[1] + ds_month_goups[2] ## December, January, February
    MAM_idxs  = ds_month_goups[3] + ds_month_goups[4] + ds_month_goups[5] ## March, April, May
    JJA_idxs  = ds_month_goups[6] + ds_month_goups[7] + ds_month_goups[8] ## June, July, August
    SON_idxs  = ds_month_goups[9] + ds_month_goups[10] + ds_month_goups[11] ## September, October, November
    
    ### grouping the months
    Divergence_DJF = ds.isel(time=DJF_idxs)
    Divergence_MAM = ds.isel(time=MAM_idxs)
    Divergence_JJA = ds.isel(time=JJA_idxs)
    Divergence_SON = ds.isel(time=SON_idxs)
  
    return Divergence_DJF, Divergence_MAM, Divergence_JJA, Divergence_SON


# In[11]:





# comparing two datasets

# In[6]:


def compare (file1, file2): ### Function to compare two datasets and make plots of the difference 
    
    ### opening datasets
    ds1 = xr.open_dataset(file1).sel(alt=slice(60,5000))
    ds2 = xr.open_dataset(file2).sel(alt=slice(60,5000))
    
    ### dividing first dataset into quantiles and calculating mean
    total_mean_1 = ds1.mean(dim='time', skipna=True)
    total_mean_2 = ds2.mean(dim='time', skipna=True)
    ds1_Q0 = ds1.quantile(q=0)
    ds1_Q1 = ds1.quantile(q=0.25)
    ds1_Q2 = ds1.quantile(q=0.5)
    ds1_Q3 = ds1.quantile(q=0.75)
    ds1_Q4 = ds1.quantile(q=1)


    bins = [float(ds1_Q0['wind_divergence']), float(ds1_Q1['wind_divergence']), float(ds1_Q2['wind_divergence']), float(ds1_Q3['wind_divergence']), float(ds1_Q4['wind_divergence'])]

    bin_labels = ['Q1, convergence', 'Q2, slight convergence', 'Q3, slight divergence', 'Q4, divergence']
    divergence_bingroup = ds1.groupby_bins(ds1['wind_divergence'], bins, labels = bin_labels, restore_coord_dims=True)


    ds1_Q1_bin = divergence_bingroup['Q1, convergence']
    ds1_Q2_bin = divergence_bingroup['Q2, slight convergence']
    ds1_Q3_bin = divergence_bingroup['Q3, slight divergence']
    ds1_Q4_bin = divergence_bingroup['Q4, divergence']
    
    
    ds1_Q1_bin_mean = ds1_Q1_bin.mean(dim='time', skipna=True)
    ds1_Q2_bin_mean = ds1_Q2_bin.mean(dim='time', skipna=True)
    ds1_Q3_bin_mean = ds1_Q3_bin.mean(dim='time', skipna=True)
    ds1_Q4_bin_mean = ds1_Q4_bin.mean(dim='time', skipna=True)
    
     
    ### dividing second dataset into quantiles and calculating mean
    ds2_Q0 = ds2.quantile(q=0)
    ds2_Q1 = ds2.quantile(q=0.25)
    ds2_Q2 = ds2.quantile(q=0.5)
    ds2_Q3 = ds2.quantile(q=0.75)
    ds2_Q4 = ds2.quantile(q=1)



    bins = [float(ds2_Q0['wind_divergence']), float(ds2_Q1['wind_divergence']), float(ds2_Q2['wind_divergence']), float(ds2_Q3['wind_divergence']), float(ds2_Q4['wind_divergence'])]

    bin_labels = ['Q1, convergence', 'Q2, slight convergence', 'Q3, slight divergence', 'Q4, divergence']
    divergence_bingroup = ds2.groupby_bins(ds2['wind_divergence'], bins, labels = bin_labels, restore_coord_dims=True)


    ds2_Q1_bin = divergence_bingroup['Q1, convergence']
    ds2_Q2_bin = divergence_bingroup['Q2, slight convergence']
    ds2_Q3_bin = divergence_bingroup['Q3, slight divergence']
    ds2_Q4_bin = divergence_bingroup['Q4, divergence']
    
    
    ds2_Q1_bin_mean = ds2_Q1_bin.mean(dim='time', skipna=True)
    ds2_Q2_bin_mean = ds2_Q2_bin.mean(dim='time', skipna=True)
    ds2_Q3_bin_mean = ds2_Q3_bin.mean(dim='time', skipna=True)
    ds2_Q4_bin_mean = ds2_Q4_bin.mean(dim='time', skipna=True)
    
    ### calculates absolute difference and plots it
    Q1_diff = ds1_Q1_bin_mean - ds2_Q1_bin_mean
    Q2_diff = ds1_Q2_bin_mean - ds2_Q2_bin_mean
    Q3_diff = ds1_Q3_bin_mean - ds2_Q3_bin_mean
    Q4_diff = ds1_Q4_bin_mean - ds2_Q4_bin_mean
    
    plt.plot(Q1_diff['mr'], Q1_diff['alt'], 'b-', label='Q1')
    plt.plot(Q2_diff['mr'], Q2_diff['alt'], 'y-', label='Q2')
    plt.plot(Q3_diff['mr'], Q3_diff['alt'], 'g-', label='Q3')
    plt.plot(Q4_diff['mr'], Q4_diff['alt'], 'r-', label='Q4')
    plt.ylabel('Height (m)')
    plt.xlabel("q' (kg/kg)")
    plt.title('absolute mixing ratio difference')
    plt.legend()
    plt.show()
    
    ### calculates relative difference and plots it
    Ws_reldiff_Q1 = Q1_diff/((ds1_Q1_bin_mean+ds2_Q1_bin_mean)/2)*100
    Ws_reldiff_Q2 = Q2_diff/((ds1_Q2_bin_mean+ds2_Q2_bin_mean)/2)*100
    Ws_reldiff_Q3 = Q3_diff/((ds1_Q3_bin_mean+ds2_Q3_bin_mean)/2)*100
    Ws_reldiff_Q4 = Q4_diff/((ds1_Q4_bin_mean+ds2_Q4_bin_mean)/2)*100
    
    plt.plot(Ws_reldiff_Q1['mr'], Ws_reldiff_Q1['alt'], 'b-', label='Q1')
    plt.plot(Ws_reldiff_Q2['mr'], Ws_reldiff_Q2['alt'], 'y-', label='Q2')
    plt.plot(Ws_reldiff_Q3['mr'], Ws_reldiff_Q3['alt'], 'g-', label='Q3')
    plt.plot(Ws_reldiff_Q4['mr'], Ws_reldiff_Q4['alt'], 'r-', label='Q4')
 
    
    plt.legend()
    
    plt.ylabel('Height (m)')
    plt.xlabel("difference (%)")
    plt.title('relative mixing ratio difference')  
    plt.show()

    
    
    return


# In[7]:



