import pandas as pd
import glob
import os
import gc
import re

# Set the main folder path
weather_data_folder = r'C:\\Users\\matte\\Desktop\\Parks Canada Project\\PEINP Advanced Concept Project Files\\PEINP Weather Station Data 2022-2025'

csv_files = glob.glob(os.path.join(weather_data_folder, '**', '*.csv'), recursive=True)

dataframes = []
skipped_files = []

for file_path in csv_files:
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            skipped_files.append(file_path)
            continue
    
    if df.empty:
        continue
    
    # Station = GRANDPARENT folder (/Cavendish/2023/XXX.csv -> 'Cavendish')
    parent_dir = os.path.dirname(file_path)
    station_folder = os.path.basename(os.path.dirname(parent_dir))
    df['station'] = station_folder
    
    dataframes.append(df)

print(f"Loaded {len(dataframes)} files")

def clean_columns(df):
    """Clean and standardize columns."""
    # Split parens/underscores - FIXED regex escape (single backslash)
    df.columns = [re.split(r'[\\(_]', str(col))[0].strip() for col in df.columns]
    
    # Drop junk
    junk_patterns = ['serial', 'battery']
    cols_to_drop = [col for col in df.columns if any(p in str(col).lower() for p in junk_patterns)]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Standard replacements + FIXED Wind columns + Dew rule
    def standardize(col):
        lower = str(col).lower()
        replacements = {
            # Wind gust mappings
            'wind gust  speed': 'Wind Gust Speed',
            'wind gust speed': 'Wind Gust Speed',
            'gust speed': 'Wind Gust Speed',       # Gust Speed -> Wind Gust Speed
            # Wind speed mappings
            'avg wind speed': 'Wind Speed',
            'average wind speed': 'Wind Speed',
            # Rain
            'accumulated rain': 'Rain',
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

# Smart dupe merge: numeric mean - FIXED skip timezone-aware cols + dupe detection
print("Shape pre-merge:", all_weather_data.shape)
numeric_cols = all_weather_data.select_dtypes(include='number').columns
non_datetime_numeric = [col for col in numeric_cols if not pd.api.types.is_datetime64_any_dtype(all_weather_data[col])]
dupe_numeric = all_weather_data[non_datetime_numeric].columns[
    all_weather_data[non_datetime_numeric].columns.duplicated()
].tolist()

if dupe_numeric:
    print(f"Merging dupes: {dupe_numeric}")
    all_weather_data[dupe_numeric] = all_weather_data.groupby('station')[dupe_numeric].transform('mean')
    all_weather_data = all_weather_data.loc[:, ~all_weather_data.columns.duplicated(keep='first')]

# MERGE Date+Time → Datetime_UTC + cleanup - FIXED error handling + streamlined
date_cols = [c for c in all_weather_data.columns if 'date' in str(c).lower()]
time_cols = [c for c in all_weather_data.columns if 'time' in str(c).lower()]

if date_cols and time_cols:
    date_col, time_col = date_cols[0], time_cols[0]
    
    # Combine and parse to UTC
    datetime_combined = all_weather_data[date_col].astype(str) + ' ' + all_weather_data[time_col].astype(str)
    all_weather_data['Datetime_UTC'] = pd.to_datetime(datetime_combined, utc=True, errors='coerce')
    
    # Drop originals + sort
    all_weather_data = all_weather_data.drop(columns=[date_col, time_col])
    all_weather_data = all_weather_data.sort_values('Datetime_UTC').reset_index(drop=True)
    
    print(f"Created Datetime_UTC, kept {len(all_weather_data)} data rows")
else:
    print("No date/time columns found")

# Drop rows with ONLY station + Datetime_UTC (null everywhere else) - Safe with errors='ignore'
mask_keep = ~(all_weather_data.drop(columns=['station', 'Datetime_UTC'], errors='ignore').isnull().all(axis=1))
all_weather_data = all_weather_data[mask_keep].reset_index(drop=True)

# Reorder: Datetime_UTC first - FIXED safe version
if 'Datetime_UTC' in all_weather_data.columns:
    other_cols = [col for col in all_weather_data.columns if col not in ['Datetime_UTC', 'station']]
    col_order = ['Datetime_UTC', 'station'] + other_cols
    all_weather_data = all_weather_data[col_order]

# Drop water/hydrology + solar radiation columns
water_solar_cols = ['Water Pressure', 'Diff Pressure', 'Barometric Pressure', 
                   'Water Temperature', 'Water Level', 'Solar Radiation']
dropped_cols = [c for c in water_solar_cols if c in all_weather_data.columns]
all_weather_data = all_weather_data.drop(columns=dropped_cols)
print(f"Dropped water/solar columns: {dropped_cols}")

# Final cleanup
zero_var_cols = (all_weather_data.nunique() == 1) | all_weather_data.isnull().all()
all_weather_data = all_weather_data.drop(columns=all_weather_data.columns[zero_var_cols])

# NEW: Replace ALL 'ERROR' strings with NaN (null)
error_mask = all_weather_data == 'ERROR'
num_errors = error_mask.sum().sum()
all_weather_data[error_mask] = pd.NA
print(f"Replaced {num_errors} 'ERROR' values with NaN")

# Optimize dtypes - FIXED downcast logic to preserve integers
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

