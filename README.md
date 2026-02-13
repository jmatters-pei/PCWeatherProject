# PCWeatherProject
Project for Advanced Concepts Class Using Parks Canada Data. The goal is to clean and analyze the weather data to calculate Fire Risk Index and to determine if any stations show too much similarity between the stations or if any station shows a unique micro climate

It is my understanding that Parks Canada does not intend to have a staff person that can write and maintain python maintain complex issues with this code. It would be better to have someone like this manage this code but there are a number of ways that Parks Canada can keep this code working longer than would otherwise be the case.
In order to keep the code working, it is important that in the future all column data use the columns they are currently using or the following column headers 'Datetime_UTC', 'station', 'Temperature', 'Rh', 'Dew', 'Rain', 'Wind Direction', 'Wind Speed', 'Wind Gust Speed'.

The folder names of the data collected from the sites need to have no spaces. Any new sites can be added by simply adding a folder to the data directory. If a folder with a different name is created the code will create a new station for it. So, any misspelled folder/station names will not be integrated into the original stations data set. 

The code ingests all CSV in the data folder of the GitHub repo. It assigns an additional column in the final product as station name. This entry is taken from the name of the first level folder under the data folder in the repo. 

The code also  changes the header names to combine columns. Any header with 'dew' in it becomes "Dew", it strips out a number of differences in title columns using delimiters. It removes the battery and serial column as this is not relvent to the intended purpose and striping these columns early helps with performance. There are also a list of specific column names in the old data that are converted to a standardize header. 

All the dataframes are then combined into a single date frame and the script prints out the new shape of the interim dataframe. 

The code merges and removes duplicated columns at this point.  

It then merges the date and time column into a single date time column and converts the time to UTC. 

It then removes the empty rows, reorders columns so date time and station are first. It drops columns that are irrlevant for the current purpose (water based readings). It converts all 'ERROR' readings to null to allow for analysis. 

It does some memory optimization. Before providing a summary of the final data frame and saving it as a CSV.

