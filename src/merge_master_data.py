#!/usr/bin/env python3
"""
Master Data Merge Script for Crop Yield Prediction
===================================================

Merges crop yield, weather, and soil data into a single modeling-ready dataset.

Target: 12 Major Crops
- CORN, WHEAT, SOYBEANS, COTTON, BARLEY, SORGHUM
- RICE, OATS, HAY, BEANS, PEANUTS, SUNFLOWER
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Configuration - Paths relative to project root
RAW_DATA_DIR = '../data/raw'
PROCESSED_DATA_DIR = '../data/processed'
OUTPUT_FILE = f'{PROCESSED_DATA_DIR}/master_yield_dataset.csv'

# 12 Major crops for yield prediction
MAJOR_CROPS = [
    'CORN', 'WHEAT', 'SOYBEANS', 'COTTON', 'BARLEY', 'SORGHUM',
    'RICE', 'OATS', 'HAY', 'BEANS', 'PEANUTS', 'SUNFLOWER'
]

# Growing season weeks (April - October, weeks 14-44)
GROWING_SEASON_START = 14
GROWING_SEASON_END = 44


def load_crop_data():
    """Load and filter crop data to major crops with yield values."""
    print("Loading crop data...")
    
    # Read in chunks to handle large file
    chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv(f'{RAW_DATA_DIR}/crop_data_pivoted.csv', chunksize=chunk_size):
        # Filter to major crops with yield values
        # Only keep 'ALL PRODUCTION PRACTICES' to avoid duplicates (irrigated/non-irrigated)
        filtered = chunk[
            (chunk['COMMODITY_DESC'].isin(MAJOR_CROPS)) & 
            (chunk['YIELD'].notna()) &
            (chunk['PRODN_PRACTICE_DESC'] == 'ALL PRODUCTION PRACTICES')
        ]
        
        # For crops with ALL CLASSES, keep only those
        # For crops without ALL CLASSES (BEANS, COTTON), keep dominant class
        has_all_classes = filtered[filtered['CLASS_DESC'] == 'ALL CLASSES']
        
        # For BEANS: use 'DRY EDIBLE, INCL CHICKPEAS' (most comprehensive)
        beans = filtered[
            (filtered['COMMODITY_DESC'] == 'BEANS') & 
            (filtered['CLASS_DESC'] == 'DRY EDIBLE, INCL CHICKPEAS')
        ]
        
        # For COTTON: use 'UPLAND' (dominant type)
        cotton = filtered[
            (filtered['COMMODITY_DESC'] == 'COTTON') & 
            (filtered['CLASS_DESC'] == 'UPLAND')
        ]
        
        combined = pd.concat([has_all_classes, beans, cotton], ignore_index=True)
        
        if len(combined) > 0:
            chunks.append(combined)
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Rename columns for consistency
    df = df.rename(columns={
        'COMMODITY_DESC': 'Crop',
        'STATE_NAME': 'State_Name',
        'COUNTY_NAME': 'County',
        'YEAR': 'Year',
        'YIELD': 'Yield'
    })
    
    # Keep essential columns
    cols_to_keep = ['Crop', 'State_Name', 'County', 'Year', 'Yield', 
                    'STATE_FIPS_CODE', 'COUNTY_CODE', 'AREA_HARVESTED', 'PRODUCTION']
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available]
    
    # Handle duplicates by aggregating (mean yield, sum area/production)
    # Source data has duplicates due to survey methodology
    print(f"  Before dedup: {len(df):,} records")
    
    agg_dict = {'Yield': 'mean'}
    if 'STATE_FIPS_CODE' in df.columns:
        agg_dict['STATE_FIPS_CODE'] = 'first'
    if 'COUNTY_CODE' in df.columns:
        agg_dict['COUNTY_CODE'] = 'first'
    if 'AREA_HARVESTED' in df.columns:
        agg_dict['AREA_HARVESTED'] = 'mean'
    if 'PRODUCTION' in df.columns:
        agg_dict['PRODUCTION'] = 'mean'
    
    df = df.groupby(['Crop', 'State_Name', 'County', 'Year']).agg(agg_dict).reset_index()
    
    print(f"  Loaded {len(df):,} yield records for {df['Crop'].nunique()} crops")
    print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
    
    return df


def load_and_aggregate_weather():
    """Load weather data and aggregate to growing season summaries."""
    print("\nLoading and aggregating weather data...")
    
    weather = pd.read_csv(f'{RAW_DATA_DIR}/weather_data_weekly.csv')
    
    # Filter to growing season (April-October, weeks 14-44)
    growing = weather[
        (weather['Week'] >= GROWING_SEASON_START) & 
        (weather['Week'] <= GROWING_SEASON_END)
    ]
    
    # Aggregate by State, County, Year
    agg = growing.groupby(['State', 'County', 'Year']).agg({
        'tmin': 'mean',
        'tmax': 'mean', 
        'tavg': 'mean',
        'prcp': 'sum',           # Total precipitation
        'rh': 'mean',            # Average humidity
        'temp_range': 'mean',    # Average daily temp range
        'heat_stress_score': 'sum',  # Total heat stress days
        'gdd_week': 'sum',       # Total growing degree days
        'Latitude': 'first',
        'Longitude': 'first'
    }).reset_index()
    
    # Rename columns
    agg = agg.rename(columns={
        'tmin': 'tmin_growing_avg',
        'tmax': 'tmax_growing_avg',
        'tavg': 'tavg_growing_avg',
        'prcp': 'prcp_growing_total',
        'rh': 'rh_growing_avg',
        'temp_range': 'temp_range_avg',
        'heat_stress_score': 'heat_stress_days',
        'gdd_week': 'gdd_total'
    })
    
    print(f"  Aggregated to {len(agg):,} county-year records")
    print(f"  Years: {agg['Year'].min()} - {agg['Year'].max()}")
    
    return agg


def load_soil_data():
    """Load soil data."""
    print("\nLoading soil data...")
    
    soil = pd.read_csv(f'{RAW_DATA_DIR}/soil_data_complete.csv')
    
    # Drop data_source column (not needed for modeling)
    if 'data_source' in soil.columns:
        soil = soil.drop('data_source', axis=1)
    
    print(f"  Loaded {len(soil):,} county soil records")
    
    return soil


def create_state_mapping():
    """Create mapping from state name to abbreviation."""
    return {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI', 'WYOMING': 'WY', 'PUERTO RICO': 'PR'
    }


def main():
    print("=" * 70)
    print("MASTER DATASET MERGE FOR CROP YIELD PREDICTION")
    print("=" * 70)
    print(f"\nTarget crops: {', '.join(MAJOR_CROPS[:6])}")
    print(f"              {', '.join(MAJOR_CROPS[6:])}")
    print("=" * 70)
    
    # Load all datasets
    crop_df = load_crop_data()
    weather_df = load_and_aggregate_weather()
    soil_df = load_soil_data()
    
    # Create state abbreviation mapping
    state_map = create_state_mapping()
    
    # Convert crop state names to abbreviations
    print("\nPreparing data for merge...")
    crop_df['State'] = crop_df['State_Name'].str.upper().map(state_map)
    
    # Standardize county names (uppercase, strip whitespace)
    crop_df['County'] = crop_df['County'].str.upper().str.strip()
    weather_df['County'] = weather_df['County'].str.upper().str.strip()
    soil_df['County'] = soil_df['County'].str.upper().str.strip()
    
    # Filter crop data to years with weather (1981+)
    crop_df = crop_df[crop_df['Year'] >= 1981]
    print(f"  Filtered crop data to 1981+: {len(crop_df):,} records")
    
    # Merge: Crop + Weather
    print("\nMerging crop + weather data...")
    merged = crop_df.merge(
        weather_df,
        on=['State', 'County', 'Year'],
        how='inner'
    )
    print(f"  After crop+weather merge: {len(merged):,} records")
    
    # Merge: + Soil
    print("Merging with soil data...")
    merged = merged.merge(
        soil_df,
        on=['State', 'County'],
        how='left',
        suffixes=('', '_soil')
    )
    
    # Drop duplicate lat/lon columns
    if 'Latitude_soil' in merged.columns:
        merged = merged.drop(['Latitude_soil', 'Longitude_soil'], axis=1)
    
    print(f"  After adding soil: {len(merged):,} records")
    
    # Check for missing values
    missing = merged.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n  Missing values:")
        for col, count in missing_cols.items():
            print(f"    {col}: {count} ({count/len(merged)*100:.1f}%)")
    
    # Final cleanup
    print("\nFinal cleanup...")
    
    # Reorder columns
    id_cols = ['Crop', 'State', 'County', 'Year', 'Yield']
    weather_cols = ['tavg_growing_avg', 'tmin_growing_avg', 'tmax_growing_avg', 
                   'prcp_growing_total', 'rh_growing_avg', 'gdd_total', 
                   'heat_stress_days', 'temp_range_avg']
    soil_cols = ['clay_pct', 'sand_pct', 'silt_pct', 'organic_matter_pct', 
                'ph', 'bulk_density', 'cec', 'awc']
    geo_cols = ['Latitude', 'Longitude']
    other_cols = ['State_Name', 'STATE_FIPS_CODE', 'COUNTY_CODE', 'AREA_HARVESTED', 'PRODUCTION']
    
    all_cols = id_cols + weather_cols + soil_cols + geo_cols
    other_available = [c for c in other_cols if c in merged.columns]
    all_cols += other_available
    
    # Keep only columns that exist
    final_cols = [c for c in all_cols if c in merged.columns]
    merged = merged[final_cols]
    
    # Save
    merged.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ MASTER DATASET CREATED!")
    print("=" * 70)
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"File size: {Path(OUTPUT_FILE).stat().st_size / (1024**2):.1f} MB")
    print(f"\nTotal records: {len(merged):,}")
    print(f"Unique crops: {merged['Crop'].nunique()}")
    print(f"Unique counties: {merged.groupby(['State', 'County']).ngroups}")
    print(f"Year range: {merged['Year'].min()} - {merged['Year'].max()}")
    
    print(f"\n📊 Records by crop:")
    print(merged['Crop'].value_counts().to_string())
    
    print(f"\n📋 Columns ({len(merged.columns)}):")
    print(f"  Target: Yield")
    print(f"  Weather: {', '.join([c for c in weather_cols if c in merged.columns])}")
    print(f"  Soil: {', '.join([c for c in soil_cols if c in merged.columns])}")
    print(f"  Location: {', '.join([c for c in geo_cols if c in merged.columns])}")
    
    print("\n" + "=" * 70)
    print("🎯 Ready for modeling!")
    print("=" * 70)


if __name__ == '__main__':
    main()
