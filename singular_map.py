# Import Libraries
import os
import geopandas as gpd
import glob
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import wkt
from descartes.patch import PolygonPatch

# Standard Paths
data_path = os.getenv('DATA','/data')
inputs_path = os.path.join(data_path,'inputs')
outputs_path = os.path.join(data_path,'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

# Define and create individual input paths
grid_path = os.path.join(inputs_path,'grid')
boundary_path = os.path.join(inputs_path,'boundary')
parameters_path = os.path.join(inputs_path,'parameters')

# Find the parameter file
parameter_file = glob(parameters_path + "/*.csv", recursive = True)
print('parameter_file:', parameter_file)

if len(parameter_file) == 1 :
    file_path = os.path.splitext(parameter_file[0])
    print('Filepath:',file_path)
    filename=file_path[0].split("/")
    print('Filename:',filename[-1])

    parameters = pd.read_csv(os.path.join(parameters_path + '/' + filename[-1] + '.csv'))
    location = parameters.loc[0][1]
    ssp = str(parameters.loc[1][1])
    year = parameters.loc[2][1]
    depth = str(parameters.loc[5][1])

if len(parameter_file) == 0 :
    location = (os.getenv('LOCATION'))
    ssp = (os.getenv('SSP'))
    year = (os.getenv('YEAR'))
    depth = (os.getenv('DEPTH'))

# Find the scenario datasets, and boundary
scenarios = glob(inputs_path + "/1km_data*.csv", recursive = True)
print('Scenarios:', scenarios)
boundary = glob(boundary_path + "/*.gpkg", recursive = True)
print('Boundary:', boundary)

# Read in the boundary file and set the crs
if len(boundary) != 0:
    boundary1 = gpd.read_file(boundary[0])
    boundary1.set_crs('epsg:27700')

# Create empty panda dataframes for each of the parameters of interest
filename=[]
filename=['xx' for n in range(len(scenarios))]
tot_count=[]
tot_count = pd.DataFrame()
tot_count['index']=[0 for n in range(1000)]
damages=[]
damages = pd.DataFrame()
damages['index']=[0 for n in range(1000)]
results=[]
results=pd.DataFrame(results)

# For each file, read in the data for each parameter, in this instance total building count and damages
# The code uses a loop, in case future modifications on the code wish to compare multiple scenarios
for i in range(0,len(scenarios)):
    test = scenarios[i]
    file_path = os.path.splitext(test)
    filename[i]=file_path[0].split("/")
    unit_name = filename[i][-1]
    parameters_1 = pd.read_csv(os.path.join(inputs_path, unit_name + '.csv'))
    tot_count[unit_name] = parameters_1['Total_Building_Count']
    damages[unit_name] = parameters_1['Damage']

tot_count.pop('index')
damages.pop('index')

# Identify the maximum and minimum values from each array (if multiple scenarios are considered it will look for
# the max and min across all to create a uniform colourbar between all maps)
results['Total_Buildings_max'] = tot_count.max()
results['Total_Damages_max'] = damages.max()
results['Total_Buildings_min'] = tot_count.min()
results['Total_Damages_min'] = damages.min()

# Replace any nan values with zero
tot_count.replace(0, np.nan, inplace = True)
damages.replace(0, np.nan, inplace = True)

# Create a column showing the depth of the event
results['depth']=['xxxx' for n in range(0,len(results))]
for i in range(0,len(results)):
    results['depth'][i]=results.index[i][-4:]

# Create a column showing the year of the event
results['year']=['xxxx' for n in range(0,len(results))]
for i in range(0,len(results)):
    results['year'][i]=results.index[i][-9:-5]

# Create a column showing the scenario of the event
results['scenario']=['xxxx' for n in range(0,len(results))]
for i in range(0,len(results)):
    results['scenario'][i]=results.index[i][-14:-10]
    if results['scenario'][i] == 'line':
        results['scenario'][i] = 'baseline'

damages_min = results.agg({'Total_Damages_min':['min']}).unstack()
damages_max = results.agg({'Total_Damages_max':['max']}).unstack()
build_min = results.agg({'Total_Buildings_min':['min']}).unstack()
build_max = results.agg({'Total_Buildings_max':['max']}).unstack()

# Create a dataframe listing of all the scenario results
dataframes_list=[]
for i in range(0,len(scenarios)):
    temp_df = pd.read_csv(scenarios[i])
    dataframes_list.append(temp_df)

# Identify the geometry column from the csv files to map the data to the os grid cells
# gdf is a dataframe containing all of the datasets
gdf=[]
for i in range(0,len(scenarios)):
    temp_df = dataframes_list[i]
    #temp_df['geometry'] = temp_df['geometry'].apply(wkt.loads)
    temp_gdf = gpd.GeoDataFrame(temp_df)
    temp_gdf['geometry'] = temp_gdf['geometry'].apply(wkt.loads)
    temp_gdf.set_geometry('geometry',crs='epsg:27700')
    gdf.append(temp_gdf)






# if there is only one scenario to view:
if len(scenarios) == 1:

    # Plot the boundary of the city
    fig,axarr = plt.subplots(figsize = (16,8))
    pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5, ax=axarr)
    
    # Read in the data from the gdf database ready to clip to the boundary
    gdf_clip = gdf[0]
    gdf_clip.crs = boundary1.crs

    # Clip the output data to the boundary
    city_clipped = gpd.clip(gdf_clip,boundary1)

    # Plot the clipped data, add a title and x-labels
    pcm = city_clipped.plot(column = "Total_Building_Count",ax=axarr,vmin=build_min[0],vmax=build_max[0],edgecolor = 'black',lw = 0.2)
    pcm.set_title(location + '_' + ssp + '_' + year + '_' +depth, fontsize=12)
    plt.setp(pcm.get_xticklabels(), rotation=30, horizontalalignment='right')

    # Add a colourbar to the figure
    fig = pcm.get_figure()
    cax = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=build_min[0],vmax=build_max[0]))
    sm._A = []
    fig.colorbar(sm, cax=cax)




# if there are two scenarios to view
if len(scenarios) == 2:
    # Create a subplot
    fig,axarr = plt.subplots(1,2,figsize = (16,8),sharex = True, sharey = True)

    # Plot the boundary of the city for both subplots
    for i in range(0,2):
        if len(boundary_path) != 0:
            pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5,ax=axarr[i])#,vmin=build_min[0],vmax=build_max[0])
    
    for i in range(0,len(scenarios)):
        # Read in the data from the gdf database ready to clip to the boundary
        gdf_clip = gdf[i]
        gdf_clip.crs = boundary1.crs

        # Clip the output data to the boundary
        city_clipped = gpd.clip(gdf_clip,boundary1)

        # Plot the clipped data, add a title and x-labels
        pcm = city_clipped.plot(column = "Total_Building_Count",ax=axarr[i],vmin=build_min[0],vmax=build_max[0],edgecolor = 'black',lw = 0.2)

        # Work out the scenario, year and depth of each run
        depth_1 = results['depth'][i]
        ssp_1 = results['scenario'][i]
        year_1 = results['year'][i]

        axarr[i].set_title(location + '_'+ ssp_1 + '_'+ year_1 + '_' + depth_1, fontsize=12)
        plt.setp(axarr[i].get_xticklabels(), rotation=30, horizontalalignment='right')

    # Add a colourbar to the figure
    fig = pcm.get_figure()
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=build_min[0],vmax=build_max[0]))
    sm._A = []
    fig.colorbar(sm, cax=cax)








# if there are four scenarios to view
if len(scenarios) == 4:
    # Create a subplot
    fig,axarr = plt.subplots(2,2,figsize = (16,16),sharex = True, sharey = True)
    m=0

    # Plot the boundary of the city for both subplots
    for i in range(0,2):
        for j in range(0,2):
            if len(boundary_path) != 0:
                pcm = boundary1.boundary.plot(edgecolor = 'black', lw = 0.5,ax=axarr[i,j],vmin=build_min[0],vmax=build_max[0])

    for i in range(0,2):
        for j in range(0,2):
            # Read in the data from the gdf database ready to clip to the boundary
            gdf_clip = gdf[m]
            gdf_clip.crs = boundary1.crs

            # Clip the output data to the boundary
            city_clipped = gpd.clip(gdf_clip,boundary1)

            # Plot the clipped data, add a title and x-labels
            pcm = city_clipped.plot(column = "Total_Building_Count",ax=axarr[i,j],vmin=build_min[0],vmax=build_max[0],edgecolor = 'black',lw = 0.2)

                # Work out the scenario, year and depth of each run
            depth_1 = results['depth'][m]
            ssp_1 = results['scenario'][m]
            year_1 = results['year'][m]

            axarr[i,j].set_title(location + '_'+ ssp_1 + '_'+ year_1 + '_' + depth_1, fontsize=12)
            #plt.setp(axarr[i].get_xticklabels(), rotation=30, horizontalalignment='right')
            m=m+1

    # Add a colourbar to the figure
    fig = pcm.get_figure()
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=build_min[0],vmax=build_max[0]))
    sm._A = []
    fig.colorbar(sm, cax=cax)

# Save the figure to the output path
plt.savefig(os.path.join(outputs_path, location +'_Buildings.png'), bbox_inches='tight' ,dpi=600)