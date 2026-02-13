"""
Weather Data Processing Pipeline for Parks Canada
Fetches, cleans, aggregates, and imputes weather data from multiple sources.
VERSION: 2.5 - With 25% Threshold + Data Quality + Dew Bounds Check
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
    'OUTPUT_DAILY': 'PEINP_daily_weather_data.csv',
    'OUTPUT_DATA_QUALITY': 'PEINP_data_quality_report.csv',
    'CACHE_DIR': 'cache',
    'MAX_WORKERS': 4,
    # Imputation settings
    'INTERPOLATE_LIMIT_HOURS': 3,
    'FORWARD_FILL_LIMIT_HOURS': 6,
    'BACKWARD_FILL_LIMIT_HOURS': 3,
    'IMPUTATION_THRESHOLD_PCT': 25.0,  # Don't impute if >25% missing
    'TEMP_MIN': -40,  # PEI reasonable bounds (°C)
    'TEMP_MAX': 40,
    'RH_MIN': 0,
    'RH_MAX': 100,
    'DEW_MIN': -100,  # Dew point reasonable bounds (°C)
    'DEW_MAX': 100,
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

def create_data_quality_csv(df):
    """
    Create comprehensive data quality report CSV with statistics.

    Shows missing values, imputation counts, and statistical measures by station and column.

    Args:
        df: DataFrame with weather data and imputation flags

    Returns:
        DataFrame with data quality metrics
    """
    logger.info("Creating data quality report with statistics...")

    # Get all data columns (not imputation flags)
    data_cols = [c for c in df.columns 
                if c not in ['Datetime_UTC', 'station'] 
                and not c.endswith('_imputed')]

    # Build report rows
    report_rows = []

    for station in sorted(df['station'].unique()):
        station_df = df[df['station'] == station]
        total_rows = len(station_df)

        for col in data_cols:
            if col not in station_df.columns:
                continue

            # Count missing values
            missing_count = station_df[col].isnull().sum()
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0

            # Count imputation types if flag column exists
            flag_col = f'{col}_imputed'
            if flag_col in station_df.columns:
                imputed_0_original = (station_df[flag_col] == 0).sum()
                imputed_1_interpolated = (station_df[flag_col] == 1).sum()
                imputed_2_forward_back = (station_df[flag_col] == 2).sum()
                imputed_3_calculated = (station_df[flag_col] == 3).sum()
                total_imputed = imputed_1_interpolated + imputed_2_forward_back + imputed_3_calculated
            else:
                imputed_0_original = total_rows - missing_count
                imputed_1_interpolated = 0
                imputed_2_forward_back = 0
                imputed_3_calculated = 0
                total_imputed = 0

            # Calculate statistics on non-missing values
            valid_data = station_df[col].dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                median_val = valid_data.median()
                min_val = valid_data.min()
                max_val = valid_data.max()
                q1_val = valid_data.quantile(0.25)
                q3_val = valid_data.quantile(0.75)
                iqr_val = q3_val - q1_val
            else:
                mean_val = median_val = min_val = max_val = q1_val = q3_val = iqr_val = np.nan

            report_rows.append({
                'station': station,
                'column': col,
                'total_rows': total_rows,
                'missing_count': missing_count,
                'missing_percent': round(missing_pct, 2),
                'original_data_count': imputed_0_original,
                'interpolated_count': imputed_1_interpolated,
                'forward_backward_filled_count': imputed_2_forward_back,
                'calculated_count': imputed_3_calculated,
                'total_imputed_count': total_imputed,
                'imputation_percent': round((total_imputed / total_rows * 100) if total_rows > 0 else 0, 2),
                'mean': round(mean_val, 2) if not np.isnan(mean_val) else np.nan,
                'median': round(median_val, 2) if not np.isnan(median_val) else np.nan,
                'min': round(min_val, 2) if not np.isnan(min_val) else np.nan,
                'max': round(max_val, 2) if not np.isnan(max_val) else np.nan,
                'q1': round(q1_val, 2) if not np.isnan(q1_val) else np.nan,
                'q3': round(q3_val, 2) if not np.isnan(q3_val) else np.nan,
                'iqr': round(iqr_val, 2) if not np.isnan(iqr_val) else np.nan,
            })

    # Add summary row for each column across all stations
    summary_rows = []
    for col in data_cols:
        if col not in df.columns:
            continue

        total_rows = len(df)
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0

        flag_col = f'{col}_imputed'
        if flag_col in df.columns:
            imputed_0_original = (df[flag_col] == 0).sum()
            imputed_1_interpolated = (df[flag_col] == 1).sum()
            imputed_2_forward_back = (df[flag_col] == 2).sum()
            imputed_3_calculated = (df[flag_col] == 3).sum()
            total_imputed = imputed_1_interpolated + imputed_2_forward_back + imputed_3_calculated
        else:
            imputed_0_original = total_rows - missing_count
            imputed_1_interpolated = 0
            imputed_2_forward_back = 0
            imputed_3_calculated = 0
            total_imputed = 0

        # Calculate statistics across all stations
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            mean_val = valid_data.mean()
            median_val = valid_data.median()
            min_val = valid_data.min()
            max_val = valid_data.max()
            q1_val = valid_data.quantile(0.25)
            q3_val = valid_data.quantile(0.75)
            iqr_val = q3_val - q1_val
        else:
            mean_val = median_val = min_val = max_val = q1_val = q3_val = iqr_val = np.nan

        summary_rows.append({
            'station': 'ALL_STATIONS',
            'column': col,
            'total_rows': total_rows,
            'missing_count': missing_count,
            'missing_percent': round(missing_pct, 2),
            'original_data_count': imputed_0_original,
            'interpolated_count': imputed_1_interpolated,
            'forward_backward_filled_count': imputed_2_forward_back,
            'calculated_count': imputed_3_calculated,
            'total_imputed_count': total_imputed,
            'imputation_percent': round((total_imputed / total_rows * 100) if total_rows > 0 else 0, 2),
            'mean': round(mean_val, 2) if not np.isnan(mean_val) else np.nan,
            'median': round(median_val, 2) if not np.isnan(median_val) else np.nan,
            'min': round(min_val, 2) if not np.isnan(min_val) else np.nan,
            'max': round(max_val, 2) if not np.isnan(max_val) else np.nan,
            'q1': round(q1_val, 2) if not np.isnan(q1_val) else np.nan,
            'q3': round(q3_val, 2) if not np.isnan(q3_val) else np.nan,
            'iqr': round(iqr_val, 2) if not np.isnan(iqr_val) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Combine station-specific and summary data
    quality_df = pd.DataFrame(report_rows)
    quality_df = pd.concat([quality_df, summary_df], ignore_index=True)

    logger.info(f"Data quality report complete: {len(quality_df)} rows")

    return quality_df

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
                'Barometric Pressure', 'Water Temperature', 'Water Level', 'Solar Radiation']
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
# IMPUTATION FUNCTIONS
# ============================================================================

def calculate_rh_from_temp_dew(temp, dew):
    """
    Calculate relative humidity from temperature and dew point using Magnus formula.

    Args:
        temp: Temperature in Celsius
        dew: Dew point in Celsius

    Returns:
        Relative humidity as percentage (0-100)
    """
    # Magnus formula constants
    a = 17.625
    b = 243.04

    # Calculate vapor pressures
    try:
        vapor_pressure_dew = np.exp((a * dew) / (b + dew))
        vapor_pressure_temp = np.exp((a * temp) / (b + temp))
        rh = 100 * (vapor_pressure_dew / vapor_pressure_temp)

        # Clip to valid range
        rh = np.clip(rh, 0, 100)
        return rh
    except:
        return np.nan

def impute_missing_values(df):
    """
    Implement tiered imputation strategy for weather data.

    NEW: Skip imputation for station-column combinations with ≥25% missing data.

    Tiers:
    1. Linear interpolation for gaps < 3 hours
    2. Forward/backward fill for gaps < 6 hours
    3. Variable-specific rules (Rain=0, Rh from Temp+Dew, etc.)
    4. Leave long gaps as NaN (sensor failures)

    Args:
        df: DataFrame with weather data

    Returns:
        df: DataFrame with imputed values and imputation flags
    """
    logger.info("="*60)
    logger.info("Starting Missing Data Imputation")
    logger.info(f"Threshold: Skip imputation if ≥{CONFIG['IMPUTATION_THRESHOLD_PCT']}% missing")
    logger.info("="*60)

    # Ensure sorted by time within each station
    df = df.sort_values(['station', 'Datetime_UTC']).reset_index(drop=True)

    # Get all columns except datetime and station
    numeric_cols = [c for c in df.columns 
                   if c not in ['Datetime_UTC', 'station'] 
                   and not c.endswith('_imputed')]

    # Convert to numeric if needed (handles object/category dtypes)
    for col in numeric_cols:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Converted {col} from {df[col].dtype} to numeric")

    logger.info(f"Found {len(numeric_cols)} columns to impute: {numeric_cols}")

    # Track statistics
    imputation_stats = {}

    for col in numeric_cols:
        original_missing = df[col].isnull().sum()
        original_pct = (original_missing / len(df)) * 100

        if original_missing == 0:
            logger.info(f"{col}: No missing values")
            continue

        logger.info(f"\n{col}: {original_missing:,} missing ({original_pct:.1f}%)")

        # Create imputation flag column
        # 0 = original data, 1 = interpolated, 2 = forward/backward filled, 3 = calculated/special
        flag_col = f'{col}_imputed'
        df[flag_col] = 0
        df.loc[df[col].isnull(), flag_col] = 1

        # NEW: Check missing percentage per station and decide which to impute
        stations_to_impute = []
        stations_to_skip = []

        for station in df['station'].unique():
            station_mask = df['station'] == station
            station_total = station_mask.sum()
            station_missing = df.loc[station_mask, col].isnull().sum()
            station_missing_pct = (station_missing / station_total * 100) if station_total > 0 else 0

            if station_missing_pct >= CONFIG['IMPUTATION_THRESHOLD_PCT']:
                stations_to_skip.append(station)
                logger.info(f"  SKIPPING {station}: {station_missing_pct:.1f}% missing (≥{CONFIG['IMPUTATION_THRESHOLD_PCT']}%)")
            else:
                stations_to_impute.append(station)

        if stations_to_skip:
            logger.info(f"  Imputing for {len(stations_to_impute)} stations, skipping {len(stations_to_skip)}")

        # TIER 1: Linear interpolation for short gaps (< 3 hours)
        for station in stations_to_impute:
            mask = df['station'] == station
            station_df = df.loc[mask].copy()

            # Set datetime index for time-based interpolation
            station_df = station_df.set_index('Datetime_UTC')

            # Interpolate with time-based method
            station_df[col] = station_df[col].interpolate(
                method='time', 
                limit=CONFIG['INTERPOLATE_LIMIT_HOURS'],
                limit_direction='both'
            )

            # Reset index and update original dataframe
            station_df = station_df.reset_index()
            df.loc[mask, col] = station_df[col].values

        after_tier1 = df[col].isnull().sum()
        tier1_imputed = original_missing - after_tier1
        logger.info(f"  Tier 1 (interpolation): Filled {tier1_imputed:,} values")

        # TIER 2: Forward/backward fill for medium gaps
        # Only for specific variables that exhibit persistence
        if col in ['Temperature', 'Dew', 'Rh', 'Wind Speed']:
            df.loc[df[col].isnull(), flag_col] = 2

            for station in stations_to_impute:
                mask = df['station'] == station
                station_df = df.loc[mask].copy()

                # Forward fill
                station_df[col] = station_df[col].fillna(method='ffill', 
                                                          limit=CONFIG['FORWARD_FILL_LIMIT_HOURS'])
                # Backward fill (shorter limit)
                station_df[col] = station_df[col].fillna(method='bfill',
                                                          limit=CONFIG['BACKWARD_FILL_LIMIT_HOURS'])

                df.loc[mask, col] = station_df[col].values

            after_tier2 = df[col].isnull().sum()
            tier2_imputed = after_tier1 - after_tier2
            logger.info(f"  Tier 2 (forward/back fill): Filled {tier2_imputed:,} values")
        else:
            after_tier2 = after_tier1
            tier2_imputed = 0

        # TIER 3: Variable-specific imputation
        tier3_imputed = 0

        if col == 'Rain':
            # Missing rain data almost certainly means no rain
            # Only impute for stations under threshold
            for station in stations_to_impute:
                mask = df['station'] == station
                missing_rain = df.loc[mask, col].isnull()
                df.loc[mask & missing_rain, flag_col] = 3
                df.loc[mask & missing_rain, col] = 0

            after_tier3 = df[col].isnull().sum()
            tier3_imputed = after_tier2 - after_tier3
            logger.info(f"  Tier 3 (rain=0): Filled {tier3_imputed:,} values with 0")

        elif col == 'Wind Gust Speed':
            # If Wind Gust Speed missing but Wind Speed available, use Wind Speed
            if 'Wind Speed' in df.columns:
                for station in stations_to_impute:
                    mask = df['station'] == station
                    missing_gust = df.loc[mask, col].isnull()
                    has_wind_speed = df.loc[mask, 'Wind Speed'].notnull()
                    impute_mask_station = mask & missing_gust & has_wind_speed

                    df.loc[impute_mask_station, flag_col] = 3
                    df.loc[impute_mask_station, col] = df.loc[impute_mask_station, 'Wind Speed']

                after_tier3 = df[col].isnull().sum()
                tier3_imputed = after_tier2 - after_tier3
                logger.info(f"  Tier 3 (gust=wind): Filled {tier3_imputed:,} values")
            else:
                after_tier3 = after_tier2

        elif col == 'Rh':
            # Calculate Rh from Temperature and Dew Point where possible
            if 'Temperature' in df.columns and 'Dew' in df.columns:
                for station in stations_to_impute:
                    mask = df['station'] == station
                    missing_rh = df.loc[mask, col].isnull()
                    has_temp_dew = df.loc[mask, 'Temperature'].notnull() & df.loc[mask, 'Dew'].notnull()
                    impute_mask_station = mask & missing_rh & has_temp_dew

                    if impute_mask_station.sum() > 0:
                        df.loc[impute_mask_station, flag_col] = 3
                        calculated_rh = calculate_rh_from_temp_dew(
                            df.loc[impute_mask_station, 'Temperature'],
                            df.loc[impute_mask_station, 'Dew']
                        )
                        df.loc[impute_mask_station, col] = calculated_rh

                after_tier3 = df[col].isnull().sum()
                tier3_imputed = after_tier2 - after_tier3
                logger.info(f"  Tier 3 (calc from T+Dew): Filled {tier3_imputed:,} values")
            else:
                after_tier3 = after_tier2
        else:
            after_tier3 = after_tier2

        # Apply bounds checking
        if col == 'Temperature':
            out_of_bounds = (df[col] < CONFIG['TEMP_MIN']) | (df[col] > CONFIG['TEMP_MAX'])
            if out_of_bounds.sum() > 0:
                logger.warning(f"  Found {out_of_bounds.sum()} out-of-bounds temperatures, setting to NaN")
                df.loc[out_of_bounds, col] = np.nan

        elif col == 'Rh':
            out_of_bounds = (df[col] < CONFIG['RH_MIN']) | (df[col] > CONFIG['RH_MAX'])
            if out_of_bounds.sum() > 0:
                logger.warning(f"  Found {out_of_bounds.sum()} out-of-bounds Rh values, clipping")
                df[col] = df[col].clip(CONFIG['RH_MIN'], CONFIG['RH_MAX'])

        elif col == 'Dew':
            out_of_bounds = (df[col] < CONFIG['DEW_MIN']) | (df[col] > CONFIG['DEW_MAX'])
            if out_of_bounds.sum() > 0:
                logger.warning(f"  Found {out_of_bounds.sum()} out-of-bounds Dew values, setting to NaN")
                df.loc[out_of_bounds, col] = np.nan
                # Update imputation flags for removed values
                df.loc[out_of_bounds, flag_col] = 0

        # Final statistics
        final_missing = df[col].isnull().sum()
        total_imputed = original_missing - final_missing
        imputation_rate = (total_imputed / original_missing * 100) if original_missing > 0 else 0

        imputation_stats[col] = {
            'original_missing': original_missing,
            'original_pct': original_pct,
            'tier1_imputed': tier1_imputed,
            'tier2_imputed': tier2_imputed,
            'tier3_imputed': tier3_imputed,
            'total_imputed': total_imputed,
            'final_missing': final_missing,
            'final_pct': (final_missing / len(df)) * 100,
            'imputation_rate': imputation_rate,
            'stations_skipped': len(stations_to_skip)
        }

        logger.info(f"  RESULT: {original_missing:,} → {final_missing:,} missing "
                   f"({imputation_rate:.1f}% imputed, {len(stations_to_skip)} stations skipped)")

    # Summary report
    logger.info("\n" + "="*60)
    logger.info("IMPUTATION SUMMARY")
    logger.info("="*60)

    total_original = sum(s['original_missing'] for s in imputation_stats.values())
    total_imputed = sum(s['total_imputed'] for s in imputation_stats.values())
    total_remaining = sum(s['final_missing'] for s in imputation_stats.values())

    logger.info(f"Total original missing values: {total_original:,}")

    if total_original > 0:
        imputation_pct = (total_imputed / total_original * 100)
        logger.info(f"Successfully imputed: {total_imputed:,} ({imputation_pct:.1f}%)")
        logger.info(f"Remaining missing: {total_remaining:,} (sensor failures or ≥{CONFIG['IMPUTATION_THRESHOLD_PCT']}% missing)")
        logger.info("")
        logger.info("By Method:")
        tier1_total = sum(s['tier1_imputed'] for s in imputation_stats.values())
        tier2_total = sum(s['tier2_imputed'] for s in imputation_stats.values())
        tier3_total = sum(s['tier3_imputed'] for s in imputation_stats.values())
        logger.info(f"  Tier 1 (Interpolation): {tier1_total:,}")
        logger.info(f"  Tier 2 (Forward/Back Fill): {tier2_total:,}")
        logger.info(f"  Tier 3 (Variable-Specific): {tier3_total:,}")
    else:
        logger.info("No missing values found - data is complete!")

    logger.info("="*60 + "\n")

    return df

# ============================================================================
# AGGREGATION FUNCTIONS
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

    # Drop imputation flag columns before aggregating
    flag_cols = [c for c in hourly_data.columns if c.endswith('_imputed')]
    hourly_data = hourly_data.drop(columns=flag_cols)

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

def create_daily_aggregates(df):
    """
    Create daily aggregated weather data with min, max, and mean statistics.
    Uses pd.concat() for mixed aggregation types.

    Args:
        df: DataFrame with weather data (must have Datetime_UTC and station columns)

    Returns:
        DataFrame with daily aggregates
    """
    logger.info("Creating daily aggregates...")

    # Ensure Datetime_UTC is datetime type
    df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC'], utc=True)

    # Convert all numeric columns
    numeric_cols_to_fix = df.select_dtypes(include=['object']).columns
    numeric_cols_to_fix = [col for col in numeric_cols_to_fix 
                          if col not in ['station', 'Datetime_UTC']]

    for col in numeric_cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create date label (UTC date)
    df['date_label'] = df['Datetime_UTC'].dt.date

    # Drop imputation flag columns before aggregating
    flag_cols = [c for c in df.columns if c.endswith('_imputed')]
    df_for_agg = df.drop(columns=flag_cols)

    # Build list of aggregations to perform
    agg_list = []

    for col in df_for_agg.columns:
        if col in ['station', 'date_label', 'Datetime_UTC']:
            continue

        if not pd.api.types.is_numeric_dtype(df_for_agg[col]):
            continue

        col_lower = col.lower()

        # Special handling for specific columns
        if 'wind gust speed' in col_lower or 'gust speed' in col_lower:
            # Wind gust: daily maximum
            agg_list.append(df_for_agg.groupby(['station', 'date_label'], observed=True)[col].max().rename(col))
        elif 'rain' in col_lower or 'precipitation' in col_lower:
            # Rain: daily sum
            agg_list.append(df_for_agg.groupby(['station', 'date_label'], observed=True)[col].sum().rename(col))
        elif 'wind direction' in col_lower or col == 'Wind Direction':
            # Wind direction: circular mean for the day
            agg_list.append(df_for_agg.groupby(['station', 'date_label'], observed=True)[col].agg(circular_mean_degrees).rename(col))
        else:
            # For all other variables: min, max, and mean
            agg_list.append(df_for_agg.groupby(['station', 'date_label'], observed=True)[col].min().rename(f'{col}_min'))
            agg_list.append(df_for_agg.groupby(['station', 'date_label'], observed=True)[col].max().rename(f'{col}_max'))
            agg_list.append(df_for_agg.groupby(['station', 'date_label'], observed=True)[col].mean().rename(f'{col}_mean'))

    # Combine all aggregations
    daily_aggregated = pd.concat(agg_list, axis=1).reset_index()

    # Convert date_label back to datetime for consistency
    daily_aggregated['Datetime_UTC'] = pd.to_datetime(daily_aggregated['date_label'])
    daily_aggregated = daily_aggregated.drop(columns=['date_label'])

    # Round numeric columns to 2 decimal places
    numeric_columns = daily_aggregated.select_dtypes(include=[np.number]).columns
    daily_aggregated[numeric_columns] = daily_aggregated[numeric_columns].round(2)

    # Reorder columns
    col_order = ['Datetime_UTC', 'station'] + [c for c in daily_aggregated.columns 
                                                if c not in ['Datetime_UTC', 'station']]
    daily_aggregated = daily_aggregated[col_order]

    logger.info(f"Daily aggregation complete: {daily_aggregated.shape}")
    logger.info(f"Date range: {daily_aggregated['Datetime_UTC'].min()} to {daily_aggregated['Datetime_UTC'].max()}")

    return daily_aggregated

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main processing pipeline."""
    logger.info("="*60)
    logger.info("Starting Weather Data Processing Pipeline v2.5")
    logger.info("With 25% Threshold + Data Quality + Dew Bounds Check")
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

        # Step 9: IMPUTE MISSING VALUES (with 25% threshold and Dew bounds)
        all_weather_data = impute_missing_values(all_weather_data)

        # Step 10: Generate quality report after imputation
        generate_data_quality_report(all_weather_data, "After Imputation")

        # Step 11: CREATE DATA QUALITY CSV REPORT (with statistics)
        data_quality_report = create_data_quality_csv(all_weather_data)
        quality_output = CONFIG['OUTPUT_DATA_QUALITY']
        data_quality_report.to_csv(quality_output, index=False)
        logger.info(f"Saved data quality report to: {quality_output}")

        # Step 12: Save all data
        output_file = CONFIG['OUTPUT_ALL_DATA']
        all_weather_data.to_csv(output_file, index=False)
        logger.info(f"Saved all weather data to: {output_file}")

        # Step 13: Create hourly aggregates
        hourly_aggregated = create_hourly_aggregates(all_weather_data)

        # Step 14: Generate quality report for hourly data
        generate_data_quality_report(hourly_aggregated, "Hourly Aggregated")

        # Step 15: Save hourly data
        hourly_output = CONFIG['OUTPUT_HOURLY']
        hourly_aggregated.to_csv(hourly_output, index=False)
        logger.info(f"Saved hourly weather data to: {hourly_output}")

        # Step 16: Create daily aggregates
        daily_aggregated = create_daily_aggregates(all_weather_data)

        # Step 17: Generate quality report for daily data
        generate_data_quality_report(daily_aggregated, "Daily Aggregated")

        # Step 18: Save daily data
        daily_output = CONFIG['OUTPUT_DAILY']
        daily_aggregated.to_csv(daily_output, index=False)
        logger.info(f"Saved daily weather data to: {daily_output}")

        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        logger.info("\nOutput files:")
        logger.info(f"  1. {output_file} - All raw data (cleaned + imputed)")
        logger.info(f"  2. {hourly_output} - Hourly aggregates")
        logger.info(f"  3. {daily_output} - Daily aggregates")
        logger.info(f"  4. {quality_output} - Data quality report with statistics")
        logger.info("\nNOTE: Imputation flags (*_imputed columns) saved in all_weather_data.csv")
        logger.info("  0 = original data")
        logger.info("  1 = interpolated (< 3 hours)")
        logger.info("  2 = forward/backward filled (3-6 hours)")
        logger.info("  3 = calculated or special method")
        logger.info(f"\nIMPUTATION RULES:")
        logger.info(f"  - Stations with ≥{CONFIG['IMPUTATION_THRESHOLD_PCT']}% missing data NOT imputed")
        logger.info(f"  - Dew values outside [{CONFIG['DEW_MIN']}, {CONFIG['DEW_MAX']}] removed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
