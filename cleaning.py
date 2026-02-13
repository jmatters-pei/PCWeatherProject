import pandas as pd
import gc
import re
import urllib.request
import json
import time
import numpy as np
import os


def get_csv_files(repo="jmatters-pei/PCWeatherProject", base_path="Data"):
    """Recursively fetch all CSV raw URLs from GitHub repo Data folder."""
    csv_urls = []
    
    # Check for GitHub token in environment variable
    github_token = os.environ.get('GITHUB_TOKEN', None)


    def fetch_json(url):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Add authentication if token is available
            if github_token:
                req.add_header('Authorization', f'token {github_token}')
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"HTTP 403 (Rate limit or authentication required): {url}")
                print("Consider setting GITHUB_TOKEN environment variable for authentication")
            else:
                print(f"HTTP {e.code}: {url}")
            return None
        except Exception as e:
            print(f"Fetch error {url}: {e}")
            return None


    def recurse_contents(path):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        contents = fetch_json(url)
        if not contents:
            return


        for item in contents:
            if item['type'] == 'file' and item['name'].endswith('.csv'):
                csv_urls.append((item['download_url'], item['path']))
            elif item['type'] == 'dir':
                recurse_contents(item['path'])
                time.sleep(0.2)


    print(f"Scanning {repo}/{base_path}")
    if github_token:
        print("Using GitHub token for authentication")
    else:
        print("No GitHub token found - using unauthenticated requests (limited to 60/hour)")
    
    recurse_contents(base_path)
    return csv_urls


# Fetch all CSV files from repo
print("Fetching file list from GitHub...")
csv_files = get_csv_files()
print(f"Found {len(csv_files)} CSV files")


dataframes = []
skipped_files = []


for raw_url, full_path in csv_files:
    try:
        req = urllib.request.Request(raw_url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        df = pd.read_csv(urllib.request.urlopen(req), encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except UnicodeDecodeError:
        try:
            req = urllib.request.Request(raw_url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            df = pd.read_csv(urllib.request.urlopen(req), encoding='latin1', on_bad_lines='skip', low_memory=False)
        except Exception as e:
            skipped_files.append((raw_url, str(e)))
            continue


    if df.empty:
        continue


    # Extract station from path (e.g., Data/Cavendish/2023/file.csv -> 'Cavendish')
    path_parts = full_path.split('/')
    station_folder = path_parts[1] if len(path_parts) > 1 else 'unknown'
    df['station'] = station_folder


    dataframes.append(df)


print(f"Loaded {len(dataframes)} files")

# ============================================================================
# ECCC STANHOPE WEATHER STATION DATA DOWNLOAD
# ============================================================================

def download_eccc_stanhope_data():
    """
    Download hourly data from ECCC Stanhope station (ID: 6545) from 2022 to present.
    ECCC times are in UTC.
    """
    import datetime
    
    station_id = 6545  # Stanhope station
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month
    
    eccc_dataframes = []
    
    # Download data from 2022 to current year
    for year in range(2022, current_year + 1):
        # Determine the last month to download for current year
        last_month = current_month if year == current_year else 12
        
        for month in range(1, last_month + 1):
            url = f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={station_id}&Year={year}&Month={month}&Day=14&timeframe=1&submit=Download+Data"
            
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0')
                df = pd.read_csv(urllib.request.urlopen(req), encoding='utf-8', on_bad_lines='skip')
                
                if not df.empty:
                    df['station'] = 'Stanhope'
                    eccc_dataframes.append(df)
                
                # Be polite to the server
                time.sleep(0.5)
                
            except Exception as e:
                continue
    
    return eccc_dataframes

# Download ECCC data and add to dataframes list
eccc_dfs = download_eccc_stanhope_data()
dataframes.extend(eccc_dfs)


def clean_columns(df):
    """Clean and standardize columns."""
    # Split parens/underscores
    df.columns = [re.split(r'[\(_]', str(col))[0].strip() for col in df.columns]


    # Drop junk
    junk_patterns = ['serial', 'battery']
    cols_to_drop = [col for col in df.columns if any(p in str(col).lower() for p in junk_patterns)]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])


    # Standard replacements
    def standardize(col):
        lower = str(col).lower().strip()
        replacements = {
            'wind gust  speed': 'Wind Gust Speed',
            'wind gust speed': 'Wind Gust Speed',
            'gust speed': 'Wind Gust Speed',
            'avg wind speed': 'Wind Speed',
            'average wind speed': 'Wind Speed',
            'wind spd': 'Wind Speed',
            'windspd': 'Wind Speed',
            'accumulated rain': 'Precipitation',
            'precip. amount': 'Precipitation',
            'temp': 'Temperature',
            'wind dir': 'Wind Direction',
            'rel hum': 'Rh',
            'date/time': 'Date/Time',
        }
        if 'dew' in lower:
            return 'Dew'
        return replacements.get(lower, str(col).title())


    df.columns = ['station' if c == 'station' else standardize(c) for c in df.columns]


    # Dedupe + drop constants
    df = df.loc[:, ~df.columns.duplicated()]
    constant_mask = (df.nunique() <= 1) & (df.columns != 'station')
    df = df.drop(columns=df.columns[constant_mask])


    return df


# Clean all
cleaned_dfs = [clean_columns(df) for df in dataframes]
gc.collect()


# Concat safely
all_weather_data = pd.concat(cleaned_dfs, axis=0, ignore_index=True, sort=False)


# Smart dupe merge
print("Shape pre-merge:", all_weather_data.shape)
numeric_cols = all_weather_data.select_dtypes(include='number').columns
non_datetime_numeric = [col for col in numeric_cols if not pd.api.types.is_datetime64_any_dtype(all_weather_data[col])]
dupe_numeric = all_weather_data[non_datetime_numeric].columns[
    all_weather_data[non_datetime_numeric].columns.duplicated()
].tolist()


if dupe_numeric:
    print(f"Merging dupes: {dupe_numeric}")
    for col in dupe_numeric:
        dup_cols = [c for c in all_weather_data.columns if c == col]
        if len(dup_cols) > 1:
            all_weather_data[col] = all_weather_data[dup_cols].bfill(axis=1).iloc[:, 0]
    
    all_weather_data = all_weather_data.loc[:, ~all_weather_data.columns.duplicated(keep='first')]


# Handle Date/Time column from ECCC Stanhope
if 'Date/Time' in all_weather_data.columns:
    date_time_parsed = pd.to_datetime(all_weather_data['Date/Time'], utc=True, errors='coerce')
    
    if 'Datetime_UTC' in all_weather_data.columns:
        all_weather_data['Datetime_UTC'] = all_weather_data['Datetime_UTC'].fillna(date_time_parsed)
    else:
        all_weather_data['Datetime_UTC'] = date_time_parsed
    
    all_weather_data = all_weather_data.drop(columns=['Date/Time'])

# Convert Datetime_UTC column to proper datetime if it exists
if 'Datetime_UTC' in all_weather_data.columns:
    all_weather_data['Datetime_UTC'] = pd.to_datetime(all_weather_data['Datetime_UTC'], utc=True, errors='coerce')

# MERGE Date+Time columns to Datetime_UTC
date_cols = [c for c in all_weather_data.columns if 'date' in str(c).lower() and c != 'Datetime_UTC']
time_cols = [c for c in all_weather_data.columns if 'time' in str(c).lower() and c != 'Datetime_UTC']


if date_cols and time_cols:
    date_col, time_col = date_cols[0], time_cols[0]
    
    datetime_combined = all_weather_data[date_col].astype(str) + ' ' + all_weather_data[time_col].astype(str)
    temp_datetime = pd.to_datetime(datetime_combined, utc=True, errors='coerce')
    
    if 'Datetime_UTC' in all_weather_data.columns:
        all_weather_data['Datetime_UTC'] = all_weather_data['Datetime_UTC'].fillna(temp_datetime)
    else:
        all_weather_data['Datetime_UTC'] = temp_datetime
    
    all_weather_data = all_weather_data.drop(columns=[date_col, time_col])
    print(f"Merged Date+Time columns into Datetime_UTC")

all_weather_data = all_weather_data.sort_values('Datetime_UTC').reset_index(drop=True)
print(f"Created Datetime_UTC, kept {len(all_weather_data)} data rows")


# Drop Hmdx and Wind Chill columns
hmdx_windchill_cols = ['Hmdx', 'Wind Chill']
dropped_hmdx = [c for c in hmdx_windchill_cols if c in all_weather_data.columns]
all_weather_data = all_weather_data.drop(columns=dropped_hmdx)
if dropped_hmdx:
    print(f"Dropped Hmdx/Wind Chill columns: {dropped_hmdx}")


# Drop Day column if it exists
if 'Day' in all_weather_data.columns:
    all_weather_data = all_weather_data.drop(columns=['Day'])


# Drop rows with only station + Datetime_UTC
mask_keep = ~(all_weather_data.drop(columns=['station', 'Datetime_UTC'], errors='ignore').isnull().all(axis=1))
all_weather_data = all_weather_data[mask_keep].reset_index(drop=True)


# Reorder columns
if 'Datetime_UTC' in all_weather_data.columns:
    other_cols = [col for col in all_weather_data.columns if col not in ['Datetime_UTC', 'station']]
    col_order = ['Datetime_UTC', 'station'] + other_cols
    all_weather_data = all_weather_data[col_order]


# Drop water/solar columns
water_solar_cols = ['Water Pressure', 'Diff Pressure', 'Barometric Pressure', 
                   'Water Temperature', 'Water Level', 'Solar Radiation']
dropped_cols = [c for c in water_solar_cols if c in all_weather_data.columns]
all_weather_data = all_weather_data.drop(columns=dropped_cols)
print(f"Dropped water/solar columns: {dropped_cols}")


# Final cleanup
zero_var_cols = (all_weather_data.nunique() == 1) | all_weather_data.isnull().all()
all_weather_data = all_weather_data.drop(columns=all_weather_data.columns[zero_var_cols])


# Replace 'ERROR' strings with NaN
error_mask = all_weather_data == 'ERROR'
num_errors = error_mask.sum().sum()
all_weather_data[error_mask] = pd.NA
print(f"Replaced {num_errors} 'ERROR' values with NaN")


# Optimize dtypes
int_cols = all_weather_data.select_dtypes(include=['int64']).columns
all_weather_data[int_cols] = all_weather_data[int_cols].apply(pd.to_numeric, downcast='integer')
float_cols = all_weather_data.select_dtypes(include=['float64']).columns
all_weather_data[float_cols] = all_weather_data[float_cols].apply(pd.to_numeric, downcast='float')
all_weather_data['station'] = all_weather_data['station'].astype('category')


print(f"Final: {all_weather_data.shape}")
print(f"Memory: {all_weather_data.memory_usage(deep=True).sum()/1e6:.1f} MB")
print("Stations:", all_weather_data['station'].value_counts().head())
print("Columns:", list(all_weather_data.columns))


all_weather_data.to_csv('PEINP_all_weather_data.csv', index=False)
print("Saved!")


# ============================================================================
# HOURLY AGGREGATION
# ============================================================================

def circular_mean_degrees(angles):
    """
    Calculate circular mean of wind directions in degrees.
    Handles 360° = 0° circular nature using vector averaging.
    """
    angles = angles.dropna()
    if len(angles) == 0:
        return np.nan
    
    angles_rad = np.deg2rad(angles)
    sin_mean = np.sin(angles_rad).mean()
    cos_mean = np.cos(angles_rad).mean()
    mean_angle_rad = np.arctan2(sin_mean, cos_mean)
    mean_angle_deg = np.rad2deg(mean_angle_rad)
    
    if mean_angle_deg < 0:
        mean_angle_deg += 360
    
    return mean_angle_deg


print("\nCreating hourly aggregates...")

# Ensure Datetime_UTC is datetime type
all_weather_data['Datetime_UTC'] = pd.to_datetime(all_weather_data['Datetime_UTC'], utc=True)

# Convert all numeric columns to proper numeric types, coercing errors to NaN
numeric_cols_to_fix = all_weather_data.select_dtypes(include=['object']).columns
numeric_cols_to_fix = [col for col in numeric_cols_to_fix if col not in ['station', 'Datetime_UTC']]

for col in numeric_cols_to_fix:
    all_weather_data[col] = pd.to_numeric(all_weather_data[col], errors='coerce')

print(f"Converted {len(numeric_cols_to_fix)} columns to numeric, coercing invalid values to NaN")

# Create hour label (each row gets assigned to nearest hour)
all_weather_data['hour_label'] = all_weather_data['Datetime_UTC'].dt.floor('h')

# Filter to ±30 minutes window around each hour
all_weather_data['minutes_from_hour'] = (
    all_weather_data['Datetime_UTC'] - all_weather_data['hour_label']
).dt.total_seconds() / 60

# Keep only data within 30 minutes before to 30 minutes after the hour
hourly_data = all_weather_data[
    (all_weather_data['minutes_from_hour'] >= -30) & 
    (all_weather_data['minutes_from_hour'] <= 30)
].copy()

print(f"Rows in time window: {len(hourly_data)} of {len(all_weather_data)}")

# Group by station and hour (with observed=True to silence warning)
grouped = hourly_data.groupby(['station', 'hour_label'], observed=True)

# Build aggregation dictionary
agg_dict = {}

for col in hourly_data.columns:
    if col in ['station', 'hour_label', 'Datetime_UTC', 'minutes_from_hour']:
        continue
    
    # Skip non-numeric columns
    if not pd.api.types.is_numeric_dtype(hourly_data[col]):
        continue
    
    col_lower = col.lower()
    
    if 'wind gust speed' in col_lower or 'gust speed' in col_lower:
        agg_dict[col] = 'max'
    elif 'precipitation' in col_lower:
        agg_dict[col] = 'sum'
    elif 'wind direction' in col_lower or col == 'Wind Direction':
        agg_dict[col] = circular_mean_degrees
    else:
        agg_dict[col] = 'mean'

# Perform aggregation
hourly_aggregated = grouped.agg(agg_dict).reset_index()

# Rename hour_label to Datetime_UTC for clarity
hourly_aggregated = hourly_aggregated.rename(columns={'hour_label': 'Datetime_UTC'})

# Round all numeric columns to 2 decimal places
numeric_columns = hourly_aggregated.select_dtypes(include=[np.number]).columns
hourly_aggregated[numeric_columns] = hourly_aggregated[numeric_columns].round(2)

# Reorder columns
col_order = ['Datetime_UTC', 'station'] + [c for c in hourly_aggregated.columns 
                                            if c not in ['Datetime_UTC', 'station']]
hourly_aggregated = hourly_aggregated[col_order]

print(f"Hourly aggregated shape: {hourly_aggregated.shape}")
print(f"Date range: {hourly_aggregated['Datetime_UTC'].min()} to {hourly_aggregated['Datetime_UTC'].max()}")
print(f"Stations: {hourly_aggregated['station'].nunique()}")

# Save to CSV
hourly_aggregated.to_csv('PEINP_hourly_weather_data.csv', index=False)
print("Saved hourly data to 'PEINP_hourly_weather_data.csv'!")
