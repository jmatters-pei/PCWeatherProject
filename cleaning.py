"""
Weather Data Processing Pipeline for Parks Canada
Fetches, cleans, and aggregates weather data from multiple sources.
"""
import pandas as pd
import gc
import re
import urllib.request
import json
import time
import numpy as np
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'ECCC_STATION_ID': 6545,
    'ECCC_START_YEAR': 2022,
    'API_DELAY': 0.5,
    'GITHUB_REPO': 'jmatters-pei/PCWeatherProject',
    'GITHUB_BASE_PATH': 'Data',
    'OUTPUT_ALL_DATA': 'PEINP_all_weather_data.csv',
    'OUTPUT_HOURLY': 'PEINP_hourly_weather_data.csv',
    'CACHE_DIR': 'cache',
    'MAX_WORKERS': 4
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('weather_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# CACHING UTILITIES
# ============================================================================

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    Path(CONFIG['CACHE_DIR']).mkdir(exist_ok=True)

def get_cache_path(cache_key):
    """Get path to cache file."""
    return Path(CONFIG['CACHE_DIR']) / f"{cache_key}.pkl"

def load_from_cache(cache_key):
    """Load data from cache if available."""
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        try:
            logger.info(f"Loading from cache: {cache_key}")
            return pd.read_pickle(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
    return None

def save_to_cache(df, cache_key):
    """Save dataframe to cache."""
    try:
        ensure_cache_dir()
        cache_path = get_cache_path(cache_key)
        df.to_pickle(cache_path)
        logger.debug(f"Saved to cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_key}: {e}")

# ============================================================================
# GITHUB DATA FETCHING
# ============================================================================

def fetch_json(url, github_token=None):
    """
    Fetch JSON from URL with proper error handling.

    Args:
        url: URL to fetch
        github_token: Optional GitHub authentication token

    Returns:
        Parsed JSON or None if failed
    """
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        if github_token:
            req.add_header('Authorization', f'token {github_token}')

        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            logger.error(f"HTTP 403 (Rate limit or authentication required): {url}")
            logger.info("Consider setting GITHUB_TOKEN environment variable for authentication")
        else:
            logger.error(f"HTTP {e.code}: {url}")
        return None
    except Exception as e:
        logger.error(f"Fetch error {url}: {e}")
        return None

def recurse_github_contents(path, repo, github_token=None):
    """
    Recursively fetch CSV files from GitHub repository.

    Args:
        path: Path within repository
        repo: Repository name (owner/repo)
        github_token: Optional GitHub token

    Returns:
        List of (download_url, file_path) tuples
    """
    csv_urls = []
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    contents = fetch_json(url, github_token)

    if not contents:
        return csv_urls

    for item in contents:
        if item['type'] == 'file' and item['name'].endswith('.csv'):
            csv_urls.append((item['download_url'], item['path']))
        elif item['type'] == 'dir':
            csv_urls.extend(recurse_github_contents(item['path'], repo, github_token))
            time.sleep(0.2)

    return csv_urls

def get_csv_files_from_github():
    """Fetch all CSV file URLs from GitHub repository."""
    github_token = os.environ.get('GITHUB_TOKEN', None)
    repo = CONFIG['GITHUB_REPO']
    base_path = CONFIG['GITHUB_BASE_PATH']

    logger.info(f"Scanning {repo}/{base_path}")
    if github_token:
        logger.info("Using GitHub token for authentication")
    else:
        logger.warning("No GitHub token found - using unauthenticated requests (limited to 60/hour)")

    csv_urls = recurse_github_contents(base_path, repo, github_token)
    logger.info(f"Found {len(csv_urls)} CSV files")

    return csv_urls

def load_single_csv(url_info):
    """
    Load a single CSV file from URL with error handling.

    Args:
        url_info: Tuple of (raw_url, full_path)

    Returns:
        Tuple of (dataframe, station_name, error_message)
    """
    raw_url, full_path = url_info

    try:
        req = urllib.request.Request(raw_url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        df = pd.read_csv(urllib.request.urlopen(req), encoding='utf-8', 
                        on_bad_lines='skip', low_memory=False)
    except UnicodeDecodeError:
        try:
            req = urllib.request.Request(raw_url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            df = pd.read_csv(urllib.request.urlopen(req), encoding='latin1', 
                            on_bad_lines='skip', low_memory=False)
        except Exception as e:
            logger.error(f"Failed to load {full_path}: {e}")
            return None, None, str(e)
    except Exception as e:
        logger.error(f"Failed to load {full_path}: {e}")
        return None, None, str(e)

    if df.empty:
        logger.warning(f"Empty dataframe from {full_path}")
        return None, None, "Empty dataframe"

    # Extract station from path
    path_parts = full_path.split('/')
    station = path_parts[1] if len(path_parts) > 1 else 'unknown'

    return df, station, None

# ============================================================================
# ECCC DATA FETCHING WITH CACHING
# ============================================================================

def download_eccc_month(year, month, station_id):
    """
    Download a single month of ECCC data with caching.

    Args:
        year: Year to download
        month: Month to download
        station_id: ECCC station ID

    Returns:
        DataFrame or None if failed
    """
    cache_key = f"eccc_{station_id}_{year}_{month:02d}"

    # Try to load from cache first
    cached_df = load_from_cache(cache_key)
    if cached_df is not None:
        return cached_df

    # Download if not cached
    url = (f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
           f"format=csv&stationID={station_id}&Year={year}&Month={month}&"
           f"Day=14&timeframe=1&submit=Download+Data")

    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        df = pd.read_csv(urllib.request.urlopen(req), encoding='utf-8', on_bad_lines='skip')

        if not df.empty:
            df['station'] = 'Stanhope'
            save_to_cache(df, cache_key)
            logger.info(f"Downloaded ECCC data: {year}-{month:02d} ({len(df)} rows)")
            return df
        else:
            logger.warning(f"Empty ECCC data for {year}-{month:02d}")
            return None

    except Exception as e:
        logger.error(f"Failed to download ECCC data for {year}-{month:02d}: {e}")
        return None

def download_eccc_stanhope_data():
    """
    Download hourly data from ECCC Stanhope station with caching.
    ECCC times are in UTC.
    """
    station_id = CONFIG['ECCC_STATION_ID']
    current_year = datetime.now().year
    current_month = datetime.now().month

    logger.info(f"Downloading ECCC Stanhope data (Station ID: {station_id})")
    logger.info(f"Date range: {CONFIG['ECCC_START_YEAR']}-01 to {current_year}-{current_month:02d}")

    eccc_dataframes = []
    failed_downloads = []

    for year in range(CONFIG['ECCC_START_YEAR'], current_year + 1):
        last_month = current_month if year == current_year else 12

        for month in range(1, last_month + 1):
            df = download_eccc_month(year, month, station_id)

            if df is not None:
                eccc_dataframes.append(df)
            else:
                failed_downloads.append(f"{year}-{month:02d}")

            time.sleep(CONFIG['API_DELAY'])

    if failed_downloads:
        logger.warning(f"Failed to download {len(failed_downloads)} months: {failed_downloads[:5]}...")

    logger.info(f"Successfully downloaded {len(eccc_dataframes)} months of ECCC data")

    return eccc_dataframes

# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_columns(df):
    """
    Clean and standardize column names.

    Args:
        df: Input dataframe

    Returns:
        Cleaned dataframe
    """
    # Split parens/underscores
    df.columns = [re.split(r'[\(_]', str(col))[0].strip() for col in df.columns]

    # Drop junk columns
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
            'accumulated rain': 'Rain',
            'precip. amount': 'Rain',
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

def process_single_file(url_info):
    """
    Load and clean a single CSV file.

    Args:
        url_info: Tuple of (url, path)

    Returns:
        Cleaned dataframe or None
    """
    df, station, error = load_single_csv(url_info)

    if df is None:
        return None

    # Clean columns immediately (clean as you go)
    df = clean_columns(df)
    df['station'] = station

    return df

def load_and_clean_github_data(csv_files):
    """
    Load and clean all GitHub CSV files using parallel processing.

    Args:
        csv_files: List of (url, path) tuples

    Returns:
        List of cleaned dataframes
    """
    logger.info(f"Loading and cleaning {len(csv_files)} files...")

    dataframes = []
    failed_files = []

    # Parallel processing
    with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
        future_to_url = {executor.submit(process_single_file, url_info): url_info 
                        for url_info in csv_files}

        for future in as_completed(future_to_url):
            url_info = future_to_url[future]
            try:
                df = future.result()
                if df is not None:
                    dataframes.append(df)
                else:
                    failed_files.append(url_info[1])
            except Exception as e:
                logger.error(f"Error processing {url_info[1]}: {e}")
                failed_files.append(url_info[1])

    logger.info(f"Successfully loaded {len(dataframes)} files")
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files")

    return dataframes

# ============================================================================
# DATA QUALITY REPORTING
# ============================================================================

def generate_data_quality_report(df, stage=""):
    """
    Generate and log data quality metrics.

    Args:
        df: DataFrame to analyze
        stage: Stage name for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Data Quality Report - {stage}")
    logger.info(f"{'='*60}")

    # Basic stats
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    # Stations
    if 'station' in df.columns:
        logger.info(f"\nStations ({df['station'].nunique()}):")
        for station, count in df['station'].value_counts().items():
            logger.info(f"  {station}: {count:,} rows")

    # Date range
    if 'Datetime_UTC' in df.columns:
        logger.info(f"\nDate range:")
        logger.info(f"  Start: {df['Datetime_UTC'].min()}")
        logger.info(f"  End: {df['Datetime_UTC'].max()}")

    # Missing data
    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    if null_pct.any():
        logger.info(f"\nMissing data (top 5):")
        for col, pct in null_pct.head().items():
            if pct > 0:
                logger.info(f"  {col}: {pct:.1f}%")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"\nDuplicate rows: {dup_count:,} ({dup_count/len(df)*100:.1f}%)")

    # Columns
    logger.info(f"\nColumns: {list(df.columns)}")

    logger.info(f"{'='*60}\n")

# ============================================================================
# DATA PROCESSING
# ============================================================================

def merge_duplicate_columns(df):
    """Merge duplicate numeric columns."""
    numeric_cols = df.select_dtypes(include='number').columns
    non_datetime_numeric = [col for col in numeric_cols 
                           if not pd.api.types.is_datetime64_any_dtype(df[col])]
    dupe_numeric = df[non_datetime_numeric].columns[
        df[non_datetime_numeric].columns.duplicated()
    ].tolist()

    if dupe_numeric:
        logger.info(f"Merging duplicate columns: {dupe_numeric}")
        for col in dupe_numeric:
            dup_cols = [c for c in df.columns if c == col]
            if len(dup_cols) > 1:
                df[col] = df[dup_cols].bfill(axis=1).iloc[:, 0]

        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    return df

def process_datetime_columns(df):
    """Process and standardize datetime columns."""
    # Handle Date/Time column from ECCC
    if 'Date/Time' in df.columns:
        date_time_parsed = pd.to_datetime(df['Date/Time'], utc=True, errors='coerce')

        if 'Datetime_UTC' in df.columns:
            df['Datetime_UTC'] = df['Datetime_UTC'].fillna(date_time_parsed)
        else:
            df['Datetime_UTC'] = date_time_parsed

        df = df.drop(columns=['Date/Time'])

    # Convert Datetime_UTC
    if 'Datetime_UTC' in df.columns:
        df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC'], utc=True, errors='coerce')

    # Merge Date+Time columns
    date_cols = [c for c in df.columns if 'date' in str(c).lower() and c != 'Datetime_UTC']
    time_cols = [c for c in df.columns if 'time' in str(c).lower() and c != 'Datetime_UTC']

    if date_cols and time_cols:
        date_col, time_col = date_cols[0], time_cols[0]
        datetime_combined = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
        temp_datetime = pd.to_datetime(datetime_combined, utc=True, errors='coerce')

        if 'Datetime_UTC' in df.columns:
            df['Datetime_UTC'] = df['Datetime_UTC'].fillna(temp_datetime)
        else:
            df['Datetime_UTC'] = temp_datetime

        df = df.drop(columns=[date_col, time_col])
        logger.info(f"Merged Date+Time columns into Datetime_UTC")

    df = df.sort_values('Datetime_UTC').reset_index(drop=True)

    return df

def clean_weather_data(df):
    """Apply all cleaning operations to weather data."""
    logger.info("Starting data cleaning...")

    # Merge duplicates
    df = merge_duplicate_columns(df)

    # Process datetime
    df = process_datetime_columns(df)

    # Drop unwanted columns (including Precipitation)
    drop_cols = ['Hmdx', 'Wind Chill', 'Day', 'Water Pressure', 'Diff Pressure', 
                'Barometric Pressure', 'Water Temperature', 'Water Level', 'Solar Radiation',
                'Precipitation']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        logger.info(f"Dropped columns: {existing_drop_cols}")

    # Drop rows with only station + Datetime_UTC
    mask_keep = ~(df.drop(columns=['station', 'Datetime_UTC'], errors='ignore').isnull().all(axis=1))
    df = df[mask_keep].reset_index(drop=True)

    # Reorder columns
    if 'Datetime_UTC' in df.columns:
        other_cols = [col for col in df.columns if col not in ['Datetime_UTC', 'station']]
        col_order = ['Datetime_UTC', 'station'] + other_cols
        df = df[col_order]

    # Drop zero variance columns
    zero_var_cols = (df.nunique() == 1) | df.isnull().all()
    df = df.drop(columns=df.columns[zero_var_cols])

    # Replace 'ERROR' strings with NaN
    error_mask = df == 'ERROR'
    num_errors = error_mask.sum().sum()
    if num_errors > 0:
        df[error_mask] = pd.NA
        logger.info(f"Replaced {num_errors} 'ERROR' values with NaN")

    # Optimize dtypes
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    df['station'] = df['station'].astype('category')

    logger.info("Data cleaning complete")

    return df

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

def create_hourly_aggregates(df):
    """Create hourly aggregated weather data."""
    logger.info("Creating hourly aggregates...")

    # Ensure Datetime_UTC is datetime type
    df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC'], utc=True)

    # Convert all numeric columns
    numeric_cols_to_fix = df.select_dtypes(include=['object']).columns
    numeric_cols_to_fix = [col for col in numeric_cols_to_fix 
                          if col not in ['station', 'Datetime_UTC']]

    for col in numeric_cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"Converted {len(numeric_cols_to_fix)} columns to numeric")

    # Create hour label
    df['hour_label'] = df['Datetime_UTC'].dt.floor('h')

    # Filter to ±30 minutes window
    df['minutes_from_hour'] = (df['Datetime_UTC'] - df['hour_label']).dt.total_seconds() / 60
    hourly_data = df[
        (df['minutes_from_hour'] >= -30) & 
        (df['minutes_from_hour'] <= 30)
    ].copy()

    logger.info(f"Rows in time window: {len(hourly_data):,} of {len(df):,}")

    # Group by station and hour
    grouped = hourly_data.groupby(['station', 'hour_label'], observed=True)

    # Build aggregation dictionary
    agg_dict = {}
    for col in hourly_data.columns:
        if col in ['station', 'hour_label', 'Datetime_UTC', 'minutes_from_hour']:
            continue

        if not pd.api.types.is_numeric_dtype(hourly_data[col]):
            continue

        col_lower = col.lower()

        if 'wind gust speed' in col_lower or 'gust speed' in col_lower:
            agg_dict[col] = 'max'
        elif 'rain' in col_lower or 'precipitation' in col_lower:
            agg_dict[col] = 'sum'
        elif 'wind direction' in col_lower or col == 'Wind Direction':
            agg_dict[col] = circular_mean_degrees
        else:
            agg_dict[col] = 'mean'

    # Perform aggregation
    hourly_aggregated = grouped.agg(agg_dict).reset_index()
    hourly_aggregated = hourly_aggregated.rename(columns={'hour_label': 'Datetime_UTC'})

    # Round numeric columns
    numeric_columns = hourly_aggregated.select_dtypes(include=[np.number]).columns
    hourly_aggregated[numeric_columns] = hourly_aggregated[numeric_columns].round(2)

    # Reorder columns
    col_order = ['Datetime_UTC', 'station'] + [c for c in hourly_aggregated.columns 
                                                if c not in ['Datetime_UTC', 'station']]
    hourly_aggregated = hourly_aggregated[col_order]

    logger.info(f"Hourly aggregation complete: {hourly_aggregated.shape}")

    return hourly_aggregated

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main processing pipeline."""
    logger.info("="*60)
    logger.info("Starting Weather Data Processing Pipeline")
    logger.info("="*60)

    try:
        # Step 1: Fetch GitHub data
        csv_files = get_csv_files_from_github()

        # Step 2: Load and clean GitHub data in parallel
        github_dataframes = load_and_clean_github_data(csv_files)

        # Step 3: Download ECCC data with caching
        eccc_dataframes = download_eccc_stanhope_data()

        # Step 4: Clean ECCC dataframes
        logger.info("Cleaning ECCC dataframes...")
        eccc_cleaned = [clean_columns(df) for df in eccc_dataframes]

        # Step 5: Combine all dataframes
        logger.info("Combining all dataframes...")
        all_dataframes = github_dataframes + eccc_cleaned
        all_weather_data = pd.concat(all_dataframes, axis=0, ignore_index=True, sort=False)

        # Free memory
        del github_dataframes, eccc_dataframes, eccc_cleaned, all_dataframes
        gc.collect()

        # Step 6: Generate quality report
        generate_data_quality_report(all_weather_data, "After Initial Load")

        # Step 7: Clean weather data
        all_weather_data = clean_weather_data(all_weather_data)

        # Step 8: Generate quality report after cleaning
        generate_data_quality_report(all_weather_data, "After Cleaning")

        # Step 9: Save all data
        output_file = CONFIG['OUTPUT_ALL_DATA']
        all_weather_data.to_csv(output_file, index=False)
        logger.info(f"Saved all weather data to: {output_file}")

        # Step 10: Create hourly aggregates
        hourly_aggregated = create_hourly_aggregates(all_weather_data)

        # Step 11: Generate quality report for hourly data
        generate_data_quality_report(hourly_aggregated, "Hourly Aggregated")

        # Step 12: Save hourly data
        hourly_output = CONFIG['OUTPUT_HOURLY']
        hourly_aggregated.to_csv(hourly_output, index=False)
        logger.info(f"Saved hourly weather data to: {hourly_output}")

        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
