#!/usr/bin/env python3
"""
Weather download - simplified to just download ALL counties
No filtering needed - we have weather for all counties
"""

import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime
import json
import sys

# Configuration - Paths relative to src/download_data/
DATA_DIR = '../../data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
WEATHER_OUTPUT = f'{RAW_DATA_DIR}/weather_data_comprehensive.csv'
CHECKPOINT_FILE = f'{RAW_DATA_DIR}/weather_checkpoint.json'
NASA_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
WEATHER_PARAMS = ['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR', 'RH2M', 'WS2M', 'ALLSKY_SFC_SW_DWN']

def download_weather(lat, lon):
    params = {
        'parameters': ','.join(WEATHER_PARAMS),
        'community': 'AG',
        'longitude': lon,
        'latitude': lat,
        'start': "19810101",
        'end': "20231231",
        'format': 'JSON'
    }
    try:
        response = requests.get(NASA_API, params=params, timeout=60)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def parse_response(data, state, county, lat, lon):
    if not data or 'properties' not in data:
        return None
    params_data = data['properties']['parameter']
    dates = list(params_data[WEATHER_PARAMS[0]].keys())
    records = []
    for date_str in dates:
        date = datetime.strptime(date_str, '%Y%m%d')
        record = {'Date': date, 'Year': date.year, 'Month': date.month,
                 'State': state, 'County': county, 'Latitude': lat, 'Longitude': lon}
        for param in WEATHER_PARAMS:
            value = params_data[param].get(date_str, None)
            record[param] = None if value in [-999, -999.0] else value
        records.append(record)
    return pd.DataFrame(records)

def aggregate_weekly(daily_df):
    daily_df['Week'] = daily_df['Date'].dt.isocalendar().week
    weekly = daily_df.groupby(['State', 'County', 'Year', 'Week']).agg({
        'Latitude': 'first', 'Longitude': 'first',
        'T2M': 'mean', 'T2M_MAX': 'max', 'T2M_MIN': 'min',
        'PRECTOTCORR': 'sum', 'RH2M': 'mean', 'WS2M': 'mean', 'ALLSKY_SFC_SW_DWN': 'mean'
    }).reset_index()
    weekly = weekly.rename(columns={
        'T2M': 'tavg', 'T2M_MAX': 'tmax', 'T2M_MIN': 'tmin',
        'PRECTOTCORR': 'prcp', 'RH2M': 'rh',
        'WS2M': 'wind_speed', 'ALLSKY_SFC_SW_DWN': 'solar_radiation'
    })
    weekly['temp_range'] = weekly['tmax'] - weekly['tmin']
    weekly['heat_stress_score'] = (weekly['tmax'] > 35).astype(int)
    weekly['gdd_week'] = ((weekly['tmax'] + weekly['tmin']) / 2 - 10).clip(lower=0) * 7
    return weekly

def main():
    print("\n" + "="*80)
    print("WEATHER DOWNLOAD FOR ALL COUNTIES")
    print("="*80 + "\n")
    
    centroids = pd.read_csv(f'{RAW_DATA_DIR}/county_centroids.csv')
    print(f"Total counties: {len(centroids)}\n")
    
    completed = set()
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            completed = set(json.load(f).get('completed', []))
        print(f"Already completed: {len(completed)}")
    
    remaining = len(centroids) - len(completed)
    print(f"Remaining: {remaining}")
    print(f"Estimated time: {remaining * 1.5 / 60:.1f} minutes\n")
    print("="*80 + "\n")
    
    all_data = []
    failed = []
    count = 0
    
    for idx, row in centroids.iterrows():
        key = f"{row['State']}_{row['County']}"
        if key in completed:
            continue
        
        count += 1
        print(f"[{count}/{remaining}] {row['County']}, {row['State']}", end=" ", flush=True)
        
        data = download_weather(row['Latitude'], row['Longitude'])
        if data:
            daily = parse_response(data, row['State'], row['County'], row['Latitude'], row['Longitude'])
            if daily is not None and len(daily) > 0:
                weekly = aggregate_weekly(daily)
                all_data.append(weekly)
                completed.add(key)
                print(f"✓ ({len(weekly)} weeks)")
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({'completed': list(completed)}, f)
                time.sleep(0.5)
            else:
                print("✗ Parse failed")
                failed.append(key)
        else:
            print("✗ Download failed")
            failed.append(key)
        
        if len(all_data) % 50 == 0 and all_data:
            df_temp = pd.concat(all_data, ignore_index=True)
            df_temp.to_csv(WEATHER_OUTPUT, index=False)
            print(f"  → Saved checkpoint: {len(all_data)} counties, {len(df_temp):,} records\n")
    
    if all_data:
        final = pd.concat(all_data, ignore_index=True)
        final.to_csv(WEATHER_OUTPUT, index=False)
        print(f"\n{'='*80}\nCOMPLETE!\n{'='*80}\n")
        print(f"File: {WEATHER_OUTPUT}")
        print(f"Size: {Path(WEATHER_OUTPUT).stat().st_size / (1024**2):.1f} MB")
        print(f"Records: {len(final):,}")
        print(f"Counties: {final['County'].nunique()}")
        if failed:
            print(f"\nFailed: {len(failed)}")

if __name__ == '__main__':
    main()
