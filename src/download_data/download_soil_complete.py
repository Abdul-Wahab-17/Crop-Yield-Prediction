#!/usr/bin/env python3
"""
Soil Data Download Script for 12 Major Crops
============================================

SOIL DATA SOURCES (in order of preference):
1. SoilGrids API - Global soil data (may be slow/down sometimes)
2. Generated regional estimates - Based on USDA soil regions

This script handles API failures gracefully with retries and fallbacks.

Target crops: Corn, Wheat, Soybeans, Cotton, Barley, Sorghum, Rice, Oats, Hay, Beans, Peanuts, Sunflower
"""

import pandas as pd
import requests
import time
from pathlib import Path
import json
import sys
import logging
from datetime import datetime

# Configuration - Paths relative to src/download_data/
DATA_DIR = '../../data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
OUTPUT_FILE = f'{RAW_DATA_DIR}/soil_data_complete.csv'
CHECKPOINT_FILE = f'{RAW_DATA_DIR}/soil_download_checkpoint.json'
LOG_FILE = f'{RAW_DATA_DIR}/soil_download_detailed.log'

# 12 Major crops for yield prediction
MAJOR_CROPS = [
    'CORN', 'WHEAT', 'SOYBEANS', 'COTTON', 'BARLEY', 'SORGHUM',
    'RICE', 'OATS', 'HAY', 'BEANS', 'PEANUTS', 'SUNFLOWER'
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SoilGrids API
SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Typical soil values by US region (based on USDA STATSGO2 averages)
# Used as fallback when API fails
REGIONAL_SOIL_DATA = {
    # Corn Belt (IA, IL, IN, OH, MO)
    'corn_belt': {
        'states': ['IA', 'IL', 'IN', 'OH', 'MO', 'NE', 'KS', 'MN', 'WI', 'MI', 'SD', 'ND'],
        'clay_pct': 28, 'sand_pct': 35, 'silt_pct': 37,
        'organic_matter_pct': 3.5, 'ph': 6.5, 'bulk_density': 1.35,
        'cec': 22, 'awc': 0.18
    },
    # Great Plains (KS, NE, OK, TX panhandle)
    'great_plains': {
        'states': ['KS', 'NE', 'OK', 'MT', 'WY', 'CO'],
        'clay_pct': 22, 'sand_pct': 45, 'silt_pct': 33,
        'organic_matter_pct': 2.0, 'ph': 7.2, 'bulk_density': 1.45,
        'cec': 18, 'awc': 0.15
    },
    # Southeast (GA, AL, SC, NC, FL, MS, LA, AR)
    'southeast': {
        'states': ['GA', 'AL', 'SC', 'NC', 'FL', 'MS', 'LA', 'AR', 'VA', 'TN', 'KY'],
        'clay_pct': 18, 'sand_pct': 55, 'silt_pct': 27,
        'organic_matter_pct': 1.5, 'ph': 5.8, 'bulk_density': 1.50,
        'cec': 12, 'awc': 0.12
    },
    # Delta (MS delta, LA, AR)
    'delta': {
        'states': ['MS', 'LA', 'AR'],
        'clay_pct': 45, 'sand_pct': 15, 'silt_pct': 40,
        'organic_matter_pct': 2.5, 'ph': 6.2, 'bulk_density': 1.30,
        'cec': 28, 'awc': 0.20
    },
    # Pacific (CA, OR, WA)
    'pacific': {
        'states': ['CA', 'OR', 'WA', 'ID'],
        'clay_pct': 25, 'sand_pct': 40, 'silt_pct': 35,
        'organic_matter_pct': 2.2, 'ph': 6.8, 'bulk_density': 1.40,
        'cec': 20, 'awc': 0.16
    },
    # Southwest (AZ, NM, TX)
    'southwest': {
        'states': ['AZ', 'NM', 'TX', 'NV', 'UT'],
        'clay_pct': 20, 'sand_pct': 55, 'silt_pct': 25,
        'organic_matter_pct': 1.0, 'ph': 7.8, 'bulk_density': 1.55,
        'cec': 15, 'awc': 0.10
    },
    # Northeast
    'northeast': {
        'states': ['NY', 'PA', 'NJ', 'CT', 'MA', 'VT', 'NH', 'ME', 'MD', 'DE', 'WV'],
        'clay_pct': 22, 'sand_pct': 42, 'silt_pct': 36,
        'organic_matter_pct': 3.0, 'ph': 6.0, 'bulk_density': 1.38,
        'cec': 18, 'awc': 0.15
    },
    # Default
    'default': {
        'states': [],
        'clay_pct': 25, 'sand_pct': 40, 'silt_pct': 35,
        'organic_matter_pct': 2.0, 'ph': 6.5, 'bulk_density': 1.40,
        'cec': 18, 'awc': 0.15
    }
}


def get_region_for_state(state):
    """Get soil region for a given state."""
    for region, data in REGIONAL_SOIL_DATA.items():
        if state in data.get('states', []):
            return region
    return 'default'


def get_soil_from_soilgrids(lat, lon, state, county, max_retries=3):
    """
    Query SoilGrids API for soil properties at a specific location.
    """
    properties = ['clay', 'sand', 'silt', 'phh2o', 'soc', 'bdod', 'cec']
    depths = ['0-5cm', '5-15cm', '15-30cm']
    
    params = {
        'lon': lon,
        'lat': lat,
        'property': properties,
        'depth': depths,
        'value': 'mean'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(SOILGRIDS_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                record = {
                    'State': state,
                    'County': county,
                    'Latitude': lat,
                    'Longitude': lon,
                    'data_source': 'SoilGrids'
                }
                
                # Extract and average across depths
                if 'properties' in data and 'layers' in data['properties']:
                    for layer in data['properties']['layers']:
                        prop_name = layer['name']
                        depths_data = layer.get('depths', [])
                        
                        values = [d.get('values', {}).get('mean') for d in depths_data 
                                 if d.get('values', {}).get('mean') is not None]
                        
                        if values:
                            avg_val = sum(values) / len(values)
                            
                            # Convert units
                            if prop_name in ['clay', 'sand', 'silt']:
                                record[f'{prop_name}_pct'] = round(avg_val / 10, 1)
                            elif prop_name == 'soc':
                                record['organic_carbon_pct'] = round(avg_val / 10, 2)
                            elif prop_name == 'phh2o':
                                record['ph'] = round(avg_val / 10, 1)
                            elif prop_name == 'bdod':
                                record['bulk_density'] = round(avg_val / 100, 2)
                            elif prop_name == 'cec':
                                record['cec'] = round(avg_val / 10, 1)
                
                return record
            
            elif response.status_code == 502:
                # Server temporarily unavailable
                wait_time = (attempt + 1) * 5
                logger.debug(f"SoilGrids 502 error, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                break
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            break
        except Exception as e:
            logger.debug(f"SoilGrids error: {e}")
            break
    
    return None


def get_regional_soil_data(state, county, lat, lon):
    """
    Get soil data based on regional averages.
    Used as fallback when API fails.
    """
    region = get_region_for_state(state)
    data = REGIONAL_SOIL_DATA.get(region, REGIONAL_SOIL_DATA['default'])
    
    return {
        'State': state,
        'County': county,
        'Latitude': lat,
        'Longitude': lon,
        'clay_pct': data['clay_pct'],
        'sand_pct': data['sand_pct'],
        'silt_pct': data['silt_pct'],
        'organic_matter_pct': data['organic_matter_pct'],
        'ph': data['ph'],
        'bulk_density': data['bulk_density'],
        'cec': data['cec'],
        'awc': data['awc'],
        'data_source': f'regional_estimate_{region}'
    }


def load_checkpoint():
    """Load download checkpoint."""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'data': [], 'api_failures': 0}


def save_checkpoint(checkpoint):
    """Save download checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def main():
    print("\n" + "=" * 80)
    print("SOIL DATA DOWNLOAD FOR 12 MAJOR CROPS")
    print("=" * 80)
    print(f"Crops: {', '.join(MAJOR_CROPS[:6])}")
    print(f"       {', '.join(MAJOR_CROPS[6:])}")
    print("=" * 80 + "\n")
    
    # Load county centroids
    if not Path(f'{RAW_DATA_DIR}/county_centroids.csv').exists():
        logger.error("county_centroids.csv not found in data/raw/!")
        logger.info("Please run the crop data filter script first.")
        sys.exit(1)
    
    centroids = pd.read_csv(f'{RAW_DATA_DIR}/county_centroids.csv')
    logger.info(f"Total counties: {len(centroids)}")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint['completed'])
    
    remaining = len(centroids) - len(completed_set)
    logger.info(f"Already completed: {len(completed_set)}")
    logger.info(f"Remaining: {remaining}")
    
    if remaining == 0:
        logger.info("All counties already completed!")
        if checkpoint['data']:
            df = pd.DataFrame(checkpoint['data'])
            df.to_csv(OUTPUT_FILE, index=False)
            logger.info(f"Saved to {OUTPUT_FILE}")
        return
    
    # First, test if SoilGrids API is available
    print("\nTesting SoilGrids API availability...")
    test_row = centroids.iloc[0]
    test_result = get_soil_from_soilgrids(
        test_row['Latitude'], test_row['Longitude'], 
        test_row['State'], test_row['County'],
        max_retries=1
    )
    
    use_api = test_result is not None
    if use_api:
        print("✓ SoilGrids API is available - will use API data")
        est_time = remaining * 0.5 / 60
    else:
        print("✗ SoilGrids API is unavailable - will use regional estimates")
        print("  (This is still useful data based on USDA soil surveys)")
        est_time = remaining * 0.01 / 60
    
    print(f"\nEstimated time: {est_time:.1f} minutes")
    print("-" * 80)
    
    soil_data = checkpoint['data'].copy() if checkpoint['data'] else []
    count = 0
    api_successes = 0
    regional_fallbacks = 0
    
    for idx, row in centroids.iterrows():
        key = f"{row['State']}_{row['County']}"
        
        if key in completed_set:
            continue
        
        count += 1
        state = row['State']
        county = row['County']
        lat = row['Latitude']
        lon = row['Longitude']
        
        # Try API first if available
        result = None
        if use_api:
            result = get_soil_from_soilgrids(lat, lon, state, county)
            if result:
                api_successes += 1
            time.sleep(0.3)  # Rate limiting
        
        # Fallback to regional estimates
        if result is None:
            result = get_regional_soil_data(state, county, lat, lon)
            regional_fallbacks += 1
        
        soil_data.append(result)
        completed_set.add(key)
        checkpoint['completed'].append(key)
        checkpoint['data'] = soil_data
        
        source_indicator = "API" if result.get('data_source') == 'SoilGrids' else "REG"
        print(f"[{count}/{remaining}] ✓ {county}, {state} ({source_indicator})")
        
        # Save checkpoint every 100 counties
        if count % 100 == 0:
            save_checkpoint(checkpoint)
            logger.info(f"Checkpoint: {count}/{remaining} complete")
    
    # Final save
    save_checkpoint(checkpoint)
    
    if soil_data:
        df = pd.DataFrame(soil_data)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE!")
        print("=" * 80)
        print(f"\nOutput file: {OUTPUT_FILE}")
        print(f"Total records: {len(df)}")
        print(f"API successes: {api_successes}")
        print(f"Regional estimates: {regional_fallbacks}")
        
        if 'data_source' in df.columns:
            print(f"\nData sources breakdown:")
            print(df['data_source'].value_counts().to_string())
        
        print(f"\nColumns: {list(df.columns)}")
        print(f"File size: {Path(OUTPUT_FILE).stat().st_size / 1024:.1f} KB")
        
        # Show sample
        print("\nSample data (first 3 rows):")
        print(df.head(3).to_string())
    else:
        logger.error("No data generated!")


if __name__ == '__main__':
    main()
