### rawdataonlyduplicatesandunimportantcolumnsremoved

"""
Simple Weather Data Cleaning Script
- Loads local CSV files + ECCC data
- Adds station names
- Standardizes datetime to UTC
- Removes unwanted columns
- Removes duplicates
- Saves to CSV
"""
import pandas as pd
import re
import urllib.request
import time
import logging
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'ECCC_STATION_ID': 6545,
    'ECCC_START_YEAR': 2022,
    'API_DELAY': 0.5,
    'LOCAL_DATA_PATH': r'C:\WeatherData\Data',
    'OUTPUT_FILE': 'cleaned_weather_data.csv',
}

# Columns to remove
COLUMNS_TO_REMOVE = [
    'solar', 'battery', 'serial',
    'Hmdx', 'Wind Chill', 'Day',
    'Water Pressure', 'Diff Pressure', 'Barometric Pressure',
    'Water Temperature', 'Water Level', 'Solar Radiation'
]

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weather_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_local_csv_files():
    """Load all CSV files from local directory."""
    local_path = Path(CONFIG['LOCAL_DATA_PATH'])

    if not local_path.exists():
        logger.error(f"Path does not exist: {local_path}")
        return []

    logger.info(f"Scanning: {local_path}")
    dataframes = []

    for csv_file in local_path.rglob('*.csv'):
        try:
            # Get station name from folder
            relative_path = csv_file.relative_to(local_path)
            station = str(relative_path).replace('\\', '/').split('/')[0]

            # Load CSV
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip', low_memory=False)

            if not df.empty:
                df['station'] = station
                dataframes.append(df)
                logger.info(f"Loaded: {station} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")

    logger.info(f"Loaded {len(dataframes)} local files")
    return dataframes

def download_eccc_data():
    """Download ECCC data."""
    station_id = CONFIG['ECCC_STATION_ID']
    current_year = datetime.now().year
    current_month = datetime.now().month

    logger.info(f"Downloading ECCC data (Station {station_id})...")
    dataframes = []

    for year in range(CONFIG['ECCC_START_YEAR'], current_year + 1):
        last_month = current_month if year == current_year else 12

        for month in range(1, last_month + 1):
            url = (f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
                   f"format=csv&stationID={station_id}&Year={year}&Month={month}&"
                   f"Day=14&timeframe=1&submit=Download+Data")

            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0')
                df = pd.read_csv(urllib.request.urlopen(req), encoding='utf-8', on_bad_lines='skip')

                if not df.empty:
                    df['station'] = 'Stanhope'
                    dataframes.append(df)
                    logger.info(f"Downloaded: {year}-{month:02d} ({len(df)} rows)")

                time.sleep(CONFIG['API_DELAY'])

            except Exception as e:
                logger.error(f"Failed to download {year}-{month:02d}: {e}")

    logger.info(f"Downloaded {len(dataframes)} ECCC months")
    return dataframes

# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_column_names(df):
    """Clean column names."""
    # Split on parentheses/underscores
    df.columns = [re.split(r'[\(_]', str(col))[0].strip() for col in df.columns]

    # Remove duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    return df

def standardize_datetime(df):
    """
    Standardize all datetime columns to Datetime_UTC.

    Handles:
    - Date/Time column from ECCC
    - Separate Date and Time columns
    - Various datetime formats
    """
    logger.info("Standardizing datetime to UTC...")

    # Step 1: Handle ECCC 'Date/Time' column
    if 'Date/Time' in df.columns:
        logger.info("  Found 'Date/Time' column (ECCC format)")
        date_time_parsed = pd.to_datetime(df['Date/Time'], utc=True, errors='coerce')

        if 'Datetime_UTC' in df.columns:
            # Fill missing values in existing Datetime_UTC
            df['Datetime_UTC'] = df['Datetime_UTC'].fillna(date_time_parsed)
        else:
            # Create new Datetime_UTC column
            df['Datetime_UTC'] = date_time_parsed

        # Drop the original Date/Time column
        df = df.drop(columns=['Date/Time'])
        logger.info("  Converted 'Date/Time' → 'Datetime_UTC'")

    # Step 2: Convert existing Datetime_UTC to UTC if needed
    if 'Datetime_UTC' in df.columns:
        df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC'], utc=True, errors='coerce')

    # Step 3: Merge separate Date and Time columns
    date_cols = [c for c in df.columns if 'date' in str(c).lower() and c != 'Datetime_UTC']
    time_cols = [c for c in df.columns if 'time' in str(c).lower() and c != 'Datetime_UTC']

    if date_cols and time_cols:
        date_col = date_cols[0]
        time_col = time_cols[0]

        logger.info(f"  Found separate date/time columns: '{date_col}' + '{time_col}'")

        # Combine Date + Time
        datetime_combined = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
        temp_datetime = pd.to_datetime(datetime_combined, utc=True, errors='coerce')

        if 'Datetime_UTC' in df.columns:
            # Fill missing values
            df['Datetime_UTC'] = df['Datetime_UTC'].fillna(temp_datetime)
        else:
            # Create new column
            df['Datetime_UTC'] = temp_datetime

        # Drop the original columns
        df = df.drop(columns=[date_col, time_col])
        logger.info(f"  Merged '{date_col}' + '{time_col}' → 'Datetime_UTC'")

    # Step 4: Sort by datetime
    if 'Datetime_UTC' in df.columns:
        df = df.sort_values('Datetime_UTC').reset_index(drop=True)
        logger.info(f"  Sorted by Datetime_UTC")

        # Show date range
        logger.info(f"  Date range: {df['Datetime_UTC'].min()} to {df['Datetime_UTC'].max()}")
    else:
        logger.warning("  WARNING: No datetime column found!")

    return df

def remove_unwanted_columns(df):
    """Remove unwanted columns."""
    cols_to_drop = []

    for col in df.columns:
        col_lower = str(col).lower()
        # Check if column matches any unwanted pattern
        if any(unwanted.lower() in col_lower for unwanted in COLUMNS_TO_REMOVE):
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Removed columns: {cols_to_drop}")

    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    original_count = len(df)
    dup_count = df.duplicated().sum()

    if dup_count > 0:
        logger.info(f"Found {dup_count:,} duplicate rows ({dup_count/original_count*100:.1f}%)")

        # Show by station
        dup_mask = df.duplicated(keep=False)
        if dup_mask.sum() > 0 and 'station' in df.columns:
            dup_by_station = df[dup_mask].groupby('station').size()
            logger.info("Duplicates by station:")
            for station, count in dup_by_station.items():
                logger.info(f"  {station}: {count:,}")

        # Remove duplicates
        df = df.drop_duplicates(keep='first')
        logger.info(f"Removed {dup_count:,} duplicates. Rows remaining: {len(df):,}")
    else:
        logger.info("No duplicates found")

    return df

def reorder_columns(df):
    """Put Datetime_UTC and station first."""
    if 'Datetime_UTC' in df.columns and 'station' in df.columns:
        other_cols = [col for col in df.columns if col not in ['Datetime_UTC', 'station']]
        col_order = ['Datetime_UTC', 'station'] + other_cols
        df = df[col_order]
        logger.info("Reordered columns: Datetime_UTC, station, ...")

    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main processing."""
    logger.info("="*60)
    logger.info("Simple Weather Data Cleaning (with Datetime Standardization)")
    logger.info("="*60)

    try:
        # Step 1: Load local CSV files
        local_data = load_local_csv_files()

        # Step 2: Download ECCC data
        eccc_data = download_eccc_data()

        # Step 3: Combine all data
        logger.info("Combining data...")
        all_data = local_data + eccc_data
        df = pd.concat(all_data, axis=0, ignore_index=True, sort=False)
        logger.info(f"Combined: {len(df):,} rows, {len(df.columns)} columns")

        # Step 4: Clean column names
        logger.info("Cleaning column names...")
        df = clean_column_names(df)

        # Step 5: Standardize datetime to UTC
        df = standardize_datetime(df)

        # Step 6: Remove unwanted columns
        logger.info("Removing unwanted columns...")
        df = remove_unwanted_columns(df)
        logger.info(f"Columns after removal: {len(df.columns)}")

        # Step 7: Remove duplicates
        logger.info("Removing duplicates...")
        df = remove_duplicates(df)

        # Step 8: Reorder columns
        df = reorder_columns(df)

        # Step 9: Save to CSV
        output_file = CONFIG['OUTPUT_FILE']
        df.to_csv(output_file, index=False)
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Saved to: {output_file}")
        logger.info(f"✓ Final rows: {len(df):,}")
        logger.info(f"✓ Final columns: {len(df.columns)}")
        logger.info(f"✓ Stations: {df['station'].nunique() if 'station' in df.columns else 'N/A'}")
        if 'Datetime_UTC' in df.columns:
            logger.info(f"✓ Date range: {df['Datetime_UTC'].min()} to {df['Datetime_UTC'].max()}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
