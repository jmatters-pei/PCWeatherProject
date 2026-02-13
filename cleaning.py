import pandas as pd
import gc
import re
import urllib.request
import json
import time

def get_csv_files(repo="jmatters-pei/PCWeatherProject", base_path="Data"):
    """Recursively fetch all CSV raw URLs from GitHub repo Data folder."""
    csv_urls = []

    def fetch_json(url):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
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
                time.sleep(0.1)

    print(f"Scanning {repo}/{base_path}")
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
        lower = str(col).lower()
        replacements = {
            'wind gust  speed': 'Wind Gust Speed',
            'wind gust speed': 'Wind Gust Speed',
            'gust speed': 'Wind Gust Speed',
            'avg wind speed': 'Wind Speed',
            'average wind speed': 'Wind Speed',
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

# Smart dupe merge
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

# MERGE Date+Time to Datetime_UTC
date_cols = [c for c in all_weather_data.columns if 'date' in str(c).lower()]
time_cols = [c for c in all_weather_data.columns if 'time' in str(c).lower()]

if date_cols and time_cols:
    date_col, time_col = date_cols[0], time_cols[0]
    datetime_combined = all_weather_data[date_col].astype(str) + ' ' + all_weather_data[time_col].astype(str)
    all_weather_data['Datetime_UTC'] = pd.to_datetime(datetime_combined, utc=True, errors='coerce')
    all_weather_data = all_weather_data.drop(columns=[date_col, time_col])
    all_weather_data = all_weather_data.sort_values('Datetime_UTC').reset_index(drop=True)
    print(f"Created Datetime_UTC, kept {len(all_weather_data)} data rows")
else:
    print("No date/time columns found")

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
