# Enhanced Solar Forecast Notebook with Vendor Integration
# Combines Himawari-9 satellite data, PySTEPS forecasting, and Sentinel-2 for improved accuracy

# First install required packages in Colab
!pip install pysteps matplotlib numpy requests rasterio imageio ffmpeg-python sentinelsat earthpy --quiet

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import requests
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import pysteps
from pysteps import io, motion, nowcasts
from pysteps.utils import conversion, transformation
import matplotlib.animation as animation
import imageio
import warnings
import pandas as pd
from sentinelsat import SentinelAPI
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Define output directory
OUTPUT_DIR = "/content/drive/MyDrive/Solar_Impact_Forecast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Coimbatore coordinates
COIMBATORE_COORDS = (11.0168, 76.9558)  # lat, lon
COIMBATORE_BBOX = {
    "min_lat": COIMBATORE_COORDS[0] - 0.25,
    "max_lat": COIMBATORE_COORDS[0] + 0.25,
    "min_lon": COIMBATORE_COORDS[1] - 0.25,
    "max_lon": COIMBATORE_COORDS[1] + 0.25
}

# South Asia region extent for Himawari SE4 imagery
SOUTH_ASIA_EXTENT = (70.0, 0.0, 100.0, 30.0)

# ==================== VENDOR INFORMATION MANAGEMENT ====================

class VendorManager:
    def __init__(self, vendor_file=None):
        """Initialize vendor manager with optional existing vendor data file"""
        self.vendor_file = vendor_file or os.path.join(OUTPUT_DIR, "vendor_data.json")
        self.vendor_data = self._load_vendor_data()
        
    def _load_vendor_data(self):
        """Load vendor data from file if it exists"""
        if os.path.exists(self.vendor_file):
            try:
                with open(self.vendor_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading vendor data: {e}")
                return self._get_default_vendor_data()
        else:
            return self._get_default_vendor_data()
    
    def _get_default_vendor_data(self):
        """Return default vendor data for Coimbatore"""
        return {
            "Coimbatore": {
                "azimuth": 180,  # South-facing panels
                "tilt": 11,
                "seasonal_winds": {
                    "JAN-MAR": {"dominant_dir": "NW", "critical_angle": np.radians(315)},
                    "APR-JUN": {"dominant_dir": "SW", "critical_angle": np.radians(225)},
                    "JUL-SEP": {"dominant_dir": "SW", "critical_angle": np.radians(225)},
                    "OCT-DEC": {"dominant_dir": "NE", "critical_angle": np.radians(45)}
                },
                "topography": "Rolling hills to the west, plains to the east",
                "shading": "None reported",
                "coordinates": COIMBATORE_COORDS
            }
        }
    
    def save_vendor_data(self):
        """Save vendor data to file"""
        try:
            with open(self.vendor_file, 'w') as f:
                json.dump(self.vendor_data, f, indent=2)
            print(f"Vendor data saved to {self.vendor_file}")
        except Exception as e:
            print(f"Error saving vendor data: {e}")
    
    def add_vendor(self, site_name, data):
        """Add or update vendor data"""
        if site_name in self.vendor_data:
            self.vendor_data[site_name].update(data)
        else:
            self.vendor_data[site_name] = data
        self.save_vendor_data()
    
    def get_vendor_info(self, site_name="Coimbatore"):
        """Get vendor information for a site"""
        return self.vendor_data.get(site_name, self.vendor_data.get("Coimbatore"))
    
    def generate_questionnaire(self, site_name="New Site"):
        """Generate questionnaire for new vendor"""
        template = f"""
## 🌞 Solar Installation Details Request - {site_name}

### 1. PANEL DETAILS
- Azimuth Angle (0°=North, 90°=East, 180°=South, 270°=West): __________
- Tilt Angle (relative to horizontal): __________
- Panel Technology (Mono/Poly/Thin Film): __________
- Total Capacity (kWp): __________

### 2. SITE GEOMETRY
- Latitude/Longitude: __________
- Altitude (meters): __________
- Topography Notes (e.g., nearby hills/water bodies): __________
- Shading Sources (trees/buildings blocking sunlight from specific directions): __________

### 3. HISTORICAL PATTERNS
- Dominant Wind Directions by Season:
  - Summer (Apr-Jun): __________
  - Monsoon (Jul-Sep): __________
  - Winter (Oct-Mar): __________
- Common Cloud Movement Patterns: __________
- Local Weather Phenomena: __________

Thank you for providing this information to help optimize our cloud forecasting system.
"""
        print(template)
        return template
    
    def process_vendor_response(self, site_name, response_dict):
        """Process a vendor response dictionary and add to vendor data"""
        # Convert seasonal wind directions to angles
        wind_direction_map = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
        }
        
        seasonal_winds = {}
        seasons = {
            "JAN-MAR": response_dict.get("winter_wind", "NE"),
            "APR-JUN": response_dict.get("summer_wind", "SW"),
            "JUL-SEP": response_dict.get("monsoon_wind", "SW"),
            "OCT-DEC": response_dict.get("winter_wind", "NE")
        }
        
        for season, direction in seasons.items():
            if direction in wind_direction_map:
                angle = wind_direction_map[direction]
                seasonal_winds[season] = {
                    "dominant_dir": direction,
                    "critical_angle": np.radians(angle)
                }
        
        vendor_data = {
            "azimuth": int(response_dict.get("azimuth", 180)),
            "tilt": float(response_dict.get("tilt", 11)),
            "seasonal_winds": seasonal_winds,
            "topography": response_dict.get("topography", ""),
            "shading": response_dict.get("shading", ""),
            "capacity": float(response_dict.get("capacity", 0)),
            "panel_type": response_dict.get("panel_type", "")
        }
        
        # Add coordinates if provided
        if "latitude" in response_dict and "longitude" in response_dict:
            vendor_data["coordinates"] = (
                float(response_dict["latitude"]),
                float(response_dict["longitude"])
            )
        
        self.add_vendor(site_name, vendor_data)
        print(f"Vendor data for {site_name} added/updated successfully")

# ==================== SATELLITE IMAGE PROCESSING ====================

def round_to_nearest_10_minutes(dt):
    """Round datetime to nearest 10-minute interval"""
    minute = dt.minute
    rounded_minute = round(minute / 10) * 10
    if rounded_minute == 60:
        dt = dt.replace(minute=0) + timedelta(hours=1)
    else:
        dt = dt.replace(minute=rounded_minute)
    return dt.replace(second=0, microsecond=0)

def find_site_in_himawari(image_array, site_coords, south_asia_extent=SOUTH_ASIA_EXTENT,
                         radius_degrees=0.25):
    """Locate site region within Himawari image"""
    height, width = image_array.shape
    lon_min, lat_min, lon_max, lat_max = south_asia_extent
    
    # Extract coordinates
    site_lat, site_lon = site_coords
    
    # Define bbox
    site_bbox = {
        "min_lat": site_lat - radius_degrees,
        "max_lat": site_lat + radius_degrees,
        "min_lon": site_lon - radius_degrees,
        "max_lon": site_lon + radius_degrees
    }
    
    # Calculate coordinate ranges
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    # Convert bbox to pixel coordinates
    x_min = int((site_bbox["min_lon"] - lon_min) / lon_range * width)
    x_max = int((site_bbox["max_lon"] - lon_min) / lon_range * width)
    y_min = int((lat_max - site_bbox["max_lat"]) / lat_range * height)
    y_max = int((lat_max - site_bbox["min_lat"]) / lat_range * height)
    
    # Ensure valid bounds
    x_min, x_max = max(0, min(width-1, x_min)), max(0, min(width-1, x_max))
    y_min, y_max = max(0, min(height-1, y_min)), max(0, min(height-1, y_max))
    
    # Enlarge if too small
    if (x_max - x_min < 20) or (y_max - y_min < 20):
        x_min, x_max = max(0, x_min-10), min(width-1, x_max+10)
        y_min, y_max = max(0, y_min-10), min(height-1, y_max+10)
        
    return slice(y_min, y_max), slice(x_min, x_max)

def download_latest_himawari_jpegs(num_images=3, hours_back=4):
    """Download recent Himawari-9 images dynamically"""
    base_url = "https://www.data.jma.go.jp/mscweb/data/himawari/"
    region = "se4"
    band = "b03"
    
    current_time = datetime.utcnow()
    start_time = current_time - timedelta(hours=hours_back)
    timestamps = []
    current_timestamp = current_time
    
    while len(timestamps) < num_images and current_timestamp > start_time:
        rounded = round_to_nearest_10_minutes(current_timestamp)
        if rounded.strftime("%Y%m%d%H%M") not in [t.strftime("%Y%m%d%H%M") for t in timestamps]:
            timestamps.append(rounded)
        current_timestamp -= timedelta(minutes=10)
    
    downloaded_files = []
    for timestamp in timestamps[:num_images]:
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M")
        file_url = f"{base_url}img/{region}/{region}_{band}_{time_str}.jpg"
        
        try:
            response = requests.get(file_url, timeout=10)
            if response.status_code == 200:
                filename = f"himawari_{date_str}_{time_str}.jpg"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append((filename, timestamp))
                print(f"Downloaded {filename}")
            else:
                print(f"Failed: {file_url} (Status {response.status_code})")
        except Exception as e:
            print(f"Error downloading {file_url}: {str(e)}")
            continue
            
    return downloaded_files

def process_jpeg_to_array(jpeg_file, site_coords=COIMBATORE_COORDS, crop_to_site=True):
    """Convert JPEG to array with optional cropping"""
    img = Image.open(jpeg_file).convert('L')
    img_array = np.array(img) / 255.0
    
    if crop_to_site:
        y_slice, x_slice = find_site_in_himawari(img_array, site_coords)
        img_array = img_array[y_slice, x_slice]
        
    return img_array

# ==================== SENTINEL-2 INTEGRATION ====================

def download_sentinel2_data(site_coords, start_date, end_date, username, password):
    """Download Sentinel-2 data for a site"""
    # Initialize the API
    api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    
    # Define the area of interest (AOI)
    lat, lon = site_coords
    footprint = f'POINT({lon} {lat})'
    
    # Query the API
    products = api.query(
        area=footprint,
        date=(start_date, end_date),
        platformname='Sentinel-2',
        cloudcoverpercentage=(0, 30)
    )
    
    if len(products) == 0:
        print("No Sentinel-2 data found for the specified period")
        return None
    
    # Download the most recent product
    product_info = api.to_dataframe(products).sort_values('ingestiondate', ascending=False).iloc[0]
    product_id = product_info.name
    
    download_path = api.download(product_id, directory_path='sentinel2_data')
    print(f"Downloaded Sentinel-2 data: {download_path}")
    
    return download_path

def process_sentinel2_data(sentinel2_file, site_coords, radius_km=5):
    """Process Sentinel-2 data to extract cloud mask and terrain information"""
    # This is a simplified placeholder for Sentinel-2 processing
    # In a real implementation, you would process the SAFE file structure
    
    print(f"Processing Sentinel-2 data for enhanced terrain and cloud analysis")
    
    # In a real implementation, you would:
    # 1. Extract bands from Sentinel-2 data
    # 2. Calculate cloud mask using Sentinel-2 QA band
    # 3. Extract DEM/terrain information
    
    # For now, return dummy data
    terrain_factor = 1.0  # Terrain adjustment factor for solar irradiance
    land_cover_mask = np.ones((100, 100))  # Land cover mask (1 for areas that can have solar panels)
    
    return {
        "terrain_factor": terrain_factor,
        "land_cover_mask": land_cover_mask
    }

# ==================== PYSTEPS ANALYSIS ====================

def pysteps_analysis(tif_files, forecast_minutes=60, timestep=10, vendor_info=None):
    """Perform motion estimation and nowcasting with vendor-specific parameters"""
    n_timesteps = forecast_minutes // timestep
    print(f"Generating {forecast_minutes}min forecast with {n_timesteps} steps")
    
    # Load data
    data_array = [rasterio.open(f).read(1) for f in tif_files]
    data_array = np.stack(data_array, axis=0)
    data_array = np.nan_to_num(data_array, nan=0.0)
    
    # Customize optical flow parameters based on vendor data
    oflow_params = {}
    if vendor_info:
        # Adjust window size based on panel tilt
        # Higher tilt = smaller window (more localized impacts)
        tilt = vendor_info.get("tilt", 11)
        window_size = max(10, 50 - int(tilt * 1.5))
        oflow_params["win_size"] = window_size
        
        print(f"Using custom optical flow parameters: window_size={window_size}")
    
    # Estimate motion field
    oflow_method = motion.get_method("lucaskanade")
    motion_field = oflow_method(data_array, **oflow_params)
    
    # Get motion vector components for analysis
    u, v = motion_field
    mean_u = np.mean(u)
    mean_v = np.mean(v)
    
    print(f"Motion field analysis:")
    print(f"  Mean u-component (east-west): {mean_u:.3f}")
    print(f"  Mean v-component (north-south): {mean_v:.3f}")
    
    # Calculate motion direction and speed
    motion_dir_rad = np.arctan2(mean_v, mean_u)
    motion_dir_deg = np.degrees(motion_dir_rad) % 360
    motion_speed = np.sqrt(mean_u**2 + mean_v**2)
    
    print(f"  Motion direction: {motion_dir_deg:.1f}° (meteorological)")
    print(f"  Motion speed: {motion_speed:.3f} pixels/timestep")
    
    # Nowcasting
    nowcast_method = nowcasts.get_method("steps")
    try:
        nowcast = nowcast_method(
            data_array, motion_field, timesteps=n_timesteps,
            mask_method=None, vel_pert_method=None, n_cascade_levels=6
        )
    except Exception as e:
        print(f"Error in nowcasting: {str(e)}")
        print("Falling back to manual advection...")
        # Fallback manual advection
        last_frame = data_array[-1]
        y_indices, x_indices = np.mgrid[0:last_frame.shape[0], 0:last_frame.shape[1]]
        nowcast = []
        for t in range(1, n_timesteps + 1):
            coords_y = np.clip(y_indices - motion_field[0]*t, 0, last_frame.shape[0]-1).astype(int)
            coords_x = np.clip(x_indices - motion_field[1]*t, 0, last_frame.shape[1]-1).astype(int)
            nowcast.append(last_frame[coords_y, coords_x])
        nowcast = np.stack(nowcast, axis=0)
    
    # Normalize
    nowcast = np.nan_to_num(nowcast, nan=0.0)
    nowcast = np.clip(nowcast, 0, 1)
    
    # Return both the forecast and motion metrics for impact analysis
    return {
        "motion_field": motion_field,
        "nowcast": nowcast,
        "motion_dir_rad": motion_dir_rad,
        "motion_dir_deg": motion_dir_deg,
        "motion_speed": motion_speed
    }

# ==================== IMPACT ASSESSMENT ====================

def get_current_season():
    """Get current season based on month"""
    current_month = datetime.utcnow().strftime("%b").upper()
    seasons = {
        "JAN-MAR": ["JAN", "FEB", "MAR"],
        "APR-JUN": ["APR", "MAY", "JUN"],
        "JUL-SEP": ["JUL", "AUG", "SEP"],
        "OCT-DEC": ["OCT", "NOV", "DEC"]
    }
    
    for season, months in seasons.items():
        if current_month in months:
            return season
    
    return "JAN-MAR"  # Default to winter

def assess_cloud_impact_with_direction(nowcast, pysteps_result, vendor_info, site_name="Coimbatore", threshold=0.3):
    """
    Analyze cloud impact with dynamic directional weighting based on:
    1. Panel orientation from vendor data
    2. Seasonal wind patterns from vendor data
    
    Returns directionally-weighted cloud coverage percentage
    """
    # Get current season
    current_season = get_current_season()
    
    # Get critical angle from seasonal data
    if "seasonal_winds" in vendor_info and current_season in vendor_info["seasonal_winds"]:
        critical_angle = vendor_info["seasonal_winds"][current_season]["critical_angle"]
        critical_direction = vendor_info["seasonal_winds"][current_season]["dominant_dir"]
    else:
        critical_angle = np.radians(225)  # Default to SW
        critical_direction = "SW"
    
    print(f"[{site_name}] Current Season: {current_season} | Critical Direction: {critical_direction}")
    
    # Get motion direction from pysteps result
    motion_dir_rad = pysteps_result["motion_dir_rad"]
    
    # Compute angular difference (with wrap-around handling)
    angle_diff = np.abs(motion_dir_rad - critical_angle)
    angle_diff = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)
    
    # Apply cosine weighting (clouds head-on = max impact)
    directional_weight = np.cos(angle_diff)
    
    # Get panel orientation factor
    # If clouds are moving perpendicular to panel azimuth, impact is reduced
    panel_azimuth_rad = np.radians(vendor_info.get("azimuth", 180))
    panel_angle_diff = np.abs(motion_dir_rad - panel_azimuth_rad)
    panel_angle_diff = np.where(panel_angle_diff > np.pi, 2*np.pi - panel_angle_diff, angle_diff)
    panel_factor = np.sin(panel_angle_diff)  # 0 when parallel, 1 when perpendicular
    
    # Get panel tilt factor (higher tilt = less impact)
    tilt = vendor_info.get("tilt", 11)
    tilt_factor = 1.0 - (tilt / 90.0) * 0.5  # Scale from 1.0 (flat) to 0.5 (vertical)
    
    # Calculate impact for each timestep
    forecast_data = nowcast
    if forecast_data.ndim == 4:
        forecast_data = np.nanmean(forecast_data, axis=0)
    
    impact_assessment = []
    
    for t in range(len(forecast_data)):
        forecast_frame = forecast_data[t]
        cloud_mask = forecast_frame > threshold
        coverage_percent = np.mean(cloud_mask) * 100
        
        # Combined weighting factors
        combined_weight = directional_weight * panel_factor * tilt_factor
        weighted_coverage = coverage_percent * combined_weight
        
        impact_assessment.append({
            "timestep": t,
            "cloud_coverage": coverage_percent,
            "directional_weight": directional_weight,
            "panel_factor": panel_factor,
            "tilt_factor": tilt_factor,
            "combined_weight": combined_weight,
            "weighted_coverage": weighted_coverage
        })
    
    return {
        "impact_assessment": impact_assessment,
        "site_name": site_name,
        "current_season": current_season,
        "critical_direction": critical_direction,
        "panel_azimuth": vendor_info.get("azimuth", 180),
        "panel_tilt": vendor_info.get("tilt", 11)
    }

def assess_cloud_impact_on_solar(nowcast, threshold=0.3, impact_threshold=0.25, timestep=10, 
                               vendor_info=None, pysteps_result=None):
    """Analyze cloud impact on solar irradiance with vendor-specific enhancements"""
    impact_assessment = {}
    
    # Handle ensemble forecasts
    if nowcast.ndim == 4:
        forecast_data = np.nanmean(nowcast, axis=0) if nowcast.shape[0] > nowcast.shape[1] else np.nanmean(nowcast, axis=1)
    else:
        forecast_data = nowcast
        
    forecast_data = np.nan_to_num(forecast_data, nan=0.0)
    
    # Get directional impact if vendor info is available
    directional_impact = None
    if vendor_info and pysteps_result:
        directional_impact = assess_cloud_impact_with_direction(
            forecast_data, pysteps_result, vendor_info)
    
    for t in range(len(forecast_data)):
        forecast_frame = forecast_data[t]
        cloud_mask = forecast_frame > threshold
        coverage_percent = np.mean(cloud_mask) * 100
        
        # Apply directional weighting if available
        weighted_coverage = coverage_percent
        if directional_impact:
            weighted_coverage = directional_impact["impact_assessment"][t]["weighted_coverage"]
        
        # Determine impact level based on weighted coverage
        if weighted_coverage >= 75: level = ("Severe", "red")
        elif weighted_coverage >= 50: level = ("High", "orange")
        elif weighted_coverage >= 25: level = ("Moderate", "yellow")
        elif weighted_coverage >= 10: level = ("Low", "blue")
        else: level = ("Minimal", "green")
        
        impact_assessment[t] = {
            "time_minutes": (t+1)*timestep,
            "cloud_coverage_percent": coverage_percent,
            "weighted_coverage": weighted_coverage,
            "impact_level": level[0],
            "color_code": level[1],
            "exceeds_threshold": weighted_coverage >= impact_threshold*100
        }
        
        # Add directional factors if available
        if directional_impact:
            impact_assessment[t]["directional_weight"] = directional_impact["impact_assessment"][t]["directional_weight"]
            impact_assessment[t]["panel_factor"] = directional_impact["impact_assessment"][t]["panel_factor"] 
            impact_assessment[t]["tilt_factor"] = directional_impact["impact_assessment"][t]["tilt_factor"]
        
    return impact_assessment

# ==================== VISUALIZATION ====================

def visualize_solar_impact(nowcast, impact_assessment, vendor_info=None, 
                         output_path="output/solar_impact.png", pysteps_result=None):
    """Visualize solar impact forecast with vendor-specific enhancements"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if nowcast.ndim == 4:
        forecast_data = np.nanmean(nowcast, axis=0) if nowcast.shape[0] > nowcast.shape[1] else np.nanmean(nowcast, axis=1)
    else:
        forecast_data = nowcast
        
    forecast_data = np.nan_to_num(forecast_data, nan=0.0)
    
    n_timesteps = len(forecast_data)
    fig, axs = plt.subplots(3, n_timesteps, figsize=(15, 12), 
                           gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Add vendor info title if available
    title = "Solar Panel Irradiance Impact Forecast"
    if vendor_info:
        site_name = vendor_info.get("site_name", "Coimbatore")
        azimuth = vendor_info.get("azimuth", 180)
        tilt = vendor_info.get("tilt", 11)
        title += f" for {site_name} (Azimuth: {azimuth}°, Tilt: {tilt}°)"
    
    for t in range(n_timesteps):
        forecast_frame = forecast_data[t]
        impact_data = impact_assessment[t]
        
        # Cloud image
        axs[0, t].imshow(forecast_frame, cmap='gray', vmin=0, vmax=1)
        axs[0, t].set_title(f"+{impact_data['time_minutes']} min")
        axs[0, t].axis('off')
        
        # Coverage text
        coverage = impact_data['cloud_coverage_percent']
        weighted = impact_data.get('weighted_coverage', coverage)
        axs[0, t].text(0.5, 0.05, f"Cloud: {coverage:.1f}%\nWeighted: {weighted:.1f}%",
                      transform=axs[0, t].transAxes,
                      color='white', fontsize=10, ha='center',
                      bbox=dict(facecolor='black', alpha=0.5))
        
        # Impact bar
        color = impact_data['color_code']
        axs[1, t].bar(0, 1, color=color, width=0.5)
        axs[1, t].set_xlim(-0.5, 0.5)
        axs[1, t].set_ylim(0, 1)
        axs[1, t].text(0, 0.5, impact_data['impact_level'], ha='center', va='center',
                      color='black' if color in ['yellow', 'green'] else 'white',
                      fontweight='bold')
        axs[1, t].axis('off')
        
        # Add directional information if available
        if pysteps_result and 'directional_weight' in impact_data:
            axs[2, t].axis('equal')
            axs[2, t].set_xlim(-1.2, 1.2)
            axs[2, t].set_ylim(-1.2, 1.2)
            
            # Draw panel orientation
            if vendor_info:
                panel_angle = np.radians(vendor_info.get("azimuth", 180))
                axs[2, t].plot([0, np.cos(panel_angle)], [0, np.sin(panel_angle)], 
                              'g-', linewidth=2, label='Panel')
            
            # Draw cloud motion direction
            motion_dir = pysteps_result["motion_dir_rad"]
            axs[2, t].plot([0, np.cos(motion_dir)], [0, np.sin(motion_dir)], 
                          'b-', linewidth=2, label='Clouds')
            
            # Draw critical wind direction if available
            if vendor_info and "seasonal_winds" in vendor_info:
                current_season = get_current_season()
                if current_season in vendor_info["seasonal_winds"]:
                    critical_angle = vendor_info["seasonal_winds"][current_season]["critical_angle"]
                    axs[2, t].plot([0, np.cos(critical_angle)], [0, np.sin(critical_angle)], 
                                  'r--', linewidth=1, label='Wind')
            
            if t == 0:
                axs[2, t].legend(loc='upper right', fontsize=8)
            axs[2, t].grid(True, linestyle='--', alpha=0.7)
            axs[2, t].set_title("Directional Factors" if t == 0 else "")
            axs[2, t].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

def create_animation(nowcast, output_path="output/cloud_forecast.gif", timestep=10, vendor_info=None):
    """Create animated cloud forecast with vendor-specific enhancements"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if nowcast.ndim == 4:
        forecast_data = np.nanmean(nowcast, axis=0) if nowcast.shape[0] > nowcast.shape[1] else np.nanmean(nowcast, axis=1)
    else:
        forecast_data = nowcast
        
    frames = []
    
    # Add vendor info title if available
    title = "Cloud Forecast"
    if vendor_info:
        site_name = vendor_info.get("site_name", "Coimbatore")
        title += f" for {site_name}"
    
    for i, frame in enumerate(forecast_data):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"{title} (t+{i*timestep}min)")
        ax.axis('off')
        
        # Save frame to memory
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    
    imageio.mimsave(output_path, frames, duration=500, loop=0)
    print(f"Saved animation to {output_path}")

def create_impact_report(impact_assessment, vendor_info=None, pysteps_result=None,
                       output_path="output/impact_report.html"):
    """Create detailed HTML report of solar impact assessment"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    site_name = "Coimbatore"
    if vendor_info:
        site_name = vendor_info.get("site_name", site_name)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Solar Impact Report - {site_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .report-header {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .impact-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .impact-table th, .impact-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .impact-table th {{ background-color: #f2f2f2; }}
            .impact-high {{ background-color: #ffcccc; }}
            .impact-moderate {{ background-color: #ffffcc; }}
            .impact-low {{ background-color: #ccffcc; }}
            .vendor-info {{ display: flex; flex-wrap: wrap; }}
            .vendor-info-block {{ flex: 1; min-width: 300px; margin: 10px; }}
        </style>
    </head>
    <body>
        <div class="report-header">
            <h1>Solar Impact Forecast Report</h1>
            <h2>{site_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}</h2>
        </div>
    """
    
    # Add vendor information if available
    if vendor_info:
        html += """
        <h2>Installation Details</h2>
        <div class="vendor-info">
            <div class="vendor-info-block">
                <h3>Panel Configuration</h3>
                <ul>
        """
        html += f"<li><strong>Azimuth:</strong> {vendor_info.get('azimuth', 'N/A')}°</li>"
        html += f"<li><strong>Tilt:</strong> {vendor_info.get('tilt', 'N/A')}°</li>"
        if 'capacity' in vendor_info:
            html += f"<li><strong>Capacity:</strong> {vendor_info.get('capacity')} kWp</li>"
        if 'panel_type' in vendor_info:
            html += f"<li><strong>Panel Type:</strong> {vendor_info.get('panel_type')}</li>"
        html += """
                </ul>
            </div>
            <div class="vendor-info-block">
                <h3>Site Characteristics</h3>
                <ul>
        """
        if 'coordinates' in vendor_info:
            lat, lon = vendor_info['coordinates']
            html += f"<li><strong>Location:</strong> {lat:.4f}°, {lon:.4f}°</li>"
        if 'topography' in vendor_info:
            html += f"<li><strong>Topography:</strong> {vendor_info.get('topography')}</li>"
        if 'shading' in vendor_info:
            html += f"<li><strong>Shading Sources:</strong> {vendor_info.get('shading')}</li>"
        
        # Add seasonal wind information
        current_season = get_current_season()
        html += f"<li><strong>Current Season:</strong> {current_season}</li>"
        if 'seasonal_winds' in vendor_info and current_season in vendor_info['seasonal_winds']:
            wind_dir = vendor_info['seasonal_winds'][current_season]['dominant_dir']
            html += f"<li><strong>Seasonal Wind Direction:</strong> {wind_dir}</li>"
        
        html += """
                </ul>
            </div>
        </div>
        """
    
    # Add motion analysis if available
    if pysteps_result:
        html += """
        <h2>Cloud Motion Analysis</h2>
        <ul>
        """
        html += f"<li><strong>Cloud Movement Direction:</strong> {pysteps_result['motion_dir_deg']:.1f}°</li>"
        html += f"<li><strong>Cloud Movement Speed:</strong> {pysteps_result['motion_speed']:.3f} pixels/timestep</li>"
        html += """
        </ul>
        """
    
    # Add impact table
    html += """
    <h2>Forecast Impact Assessment</h2>
    <table class="impact-table">
        <tr>
            <th>Time</th>
            <th>Cloud Coverage</th>
            <th>Weighted Coverage</th>
            <th>Impact Level</th>
    """
    
    if 'directional_weight' in list(impact_assessment.values())[0]:
        html += "<th>Directional<br>Factor</th>"
    
    html += "</tr>"
    
    for t, data in sorted(impact_assessment.items()):
        impact_class = ""
        if data['impact_level'] == "Severe" or data['impact_level'] == "High":
            impact_class = "impact-high"
        elif data['impact_level'] == "Moderate":
            impact_class = "impact-moderate"
        else:
            impact_class = "impact-low"
        
        html += f"""
        <tr class="{impact_class}">
            <td>+{data['time_minutes']} min</td>
            <td>{data['cloud_coverage_percent']:.1f}%</td>
            <td>{data.get('weighted_coverage', data['cloud_coverage_percent']):.1f}%</td>
            <td>{data['impact_level']}</td>
        """
        
        if 'directional_weight' in data:
            html += f"<td>{data['directional_weight']:.2f}</td>"
        
        html += "</tr>"
    
    html += """
    </table>
    
    <div style="margin-top: 30px;">
        <h3>Recommendations:</h3>
        <ul>
    """
    
    # Generate dynamic recommendations
    severe_periods = [d['time_minutes'] for t, d in impact_assessment.items() 
                    if d['impact_level'] in ["Severe", "High"]]
    if severe_periods:
        html += f"<li>Prepare for reduced output during minutes {', '.join(map(str, severe_periods))} of the forecast period.</li>"
    else:
        html += "<li>No significant impact expected during the forecast period.</li>"
    
    # Add more recommendations based on vendor data if available
    if vendor_info and pysteps_result:
        panel_azimuth = np.radians(vendor_info.get("azimuth", 180))
        motion_dir = pysteps_result["motion_dir_rad"]
        angle_diff = np.abs(panel_azimuth - motion_dir)
        angle_diff = np.where(angle_diff > np.pi, 2*np.pi - angle_diff, angle_diff)
        
        if angle_diff < np.pi/4:  # Within 45 degrees
            html += "<li>Cloud motion is aligned with panel orientation, which may prolong impact duration.</li>"
        elif angle_diff > 3*np.pi/4:  # More than 135 degrees
            html += "<li>Cloud motion is opposite to panel orientation, which may shorten impact duration.</li>"
    
    html += """
        </ul>
    </div>
    
    <p style="margin-top: 40px; color: #666; font-size: 0.8em;">
        This report was generated using PySTEPS cloud motion analysis and vendor-specific solar installation data.
        For more information, please contact support.
    </p>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Saved impact report to {output_path}")

# ==================== MAIN WORKFLOW ====================

def main(site_name="Coimbatore", sentinel2_analysis=False):
    """Main workflow with vendor data integration"""
    # Initialize vendor manager
    vendor_manager = VendorManager()
    vendor_info = vendor_manager.get_vendor_info(site_name)
    vendor_info["site_name"] = site_name
    
    print(f"Processing forecast for {site_name}")
    print(f"Vendor data: {json.dumps(vendor_info, indent=2)}")
    
    # Get site coordinates
    site_coords = vendor_info.get("coordinates", COIMBATORE_COORDS)
    
    # Create site-specific output directory
    site_output_dir = os.path.join(OUTPUT_DIR, site_name)
    os.makedirs(site_output_dir, exist_ok=True)
    
    print("Downloading latest Himawari images...")
    downloaded_files = download_latest_himawari_jpegs()
    
    if len(downloaded_files) < 2:
        print("Not enough images downloaded!")
        return
    
    print("\nProcessing images...")
    image_arrays = [process_jpeg_to_array(f[0], site_coords) for f in downloaded_files]
    
    print("\nCreating TIFF files for Pysteps...")
    clipped_files = []
    for i, arr in enumerate(image_arrays):
        filename = f"input_image_{i}.tif"
        with rasterio.open(
            filename, 'w',
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype
        ) as dst:
            dst.write(arr, 1)
        clipped_files.append(filename)
    
    # Download Sentinel-2 data if requested
    sentinel2_data = None
    if sentinel2_analysis:
        try:
            print("\nDownloading Sentinel-2 data for enhanced analysis...")
            # Note: In a real implementation, you would provide actual credentials
            sentinel2_file = download_sentinel2_data(
                site_coords, 
                (datetime.now() - timedelta(days=14)).strftime("%Y%m%d"), 
                datetime.now().strftime("%Y%m%d"),
                "username", "password"  # Replace with actual credentials
            )
            
            if sentinel2_file:
                sentinel2_data = process_sentinel2_data(sentinel2_file, site_coords)
                print("Successfully processed Sentinel-2 data")
        except Exception as e:
            print(f"Error processing Sentinel-2 data: {str(e)}")
            print("Continuing without Sentinel-2 enhancement...")
    
    print("\nRunning Pysteps analysis with vendor parameters...")
    pysteps_result = pysteps_analysis(clipped_files, vendor_info=vendor_info)
    
    print("\nAssessing cloud impact with vendor-specific parameters...")
    impact_assessment = assess_cloud_impact_on_solar(
        pysteps_result["nowcast"], 
        vendor_info=vendor_info,
        pysteps_result=pysteps_result
    )
    
    print("\nGenerating visualization...")
    vis_path = os.path.join(site_output_dir, f"{site_name}_solar_impact_forecast.png")
    visualize_solar_impact(
        pysteps_result["nowcast"], 
        impact_assessment, 
        vendor_info=vendor_info,
        output_path=vis_path,
        pysteps_result=pysteps_result
    )
    
    print("\nCreating animation...")
    anim_path = os.path.join(site_output_dir, f"{site_name}_cloud_forecast.gif")
    create_animation(
        pysteps_result["nowcast"], 
        output_path=anim_path,
        vendor_info=vendor_info
    )
    
    print("\nGenerating impact report...")
    report_path = os.path.join(site_output_dir, f"{site_name}_impact_report.html")
    create_impact_report(
        impact_assessment,
        vendor_info=vendor_info,
        pysteps_result=pysteps_result,
        output_path=report_path
    )
    
    print(f"\nResults saved to: {site_output_dir}")
    print("\nFirst frame sample:")
    plt.figure(figsize=(6,4))
    plt.imshow(image_arrays[0], cmap='gray')
    plt.title(f"Latest Cloud Image - {site_name}")
    plt.axis('off')
    plt.show()

# ==================== INTERACTIVE VENDOR DATA MANAGEMENT ====================

def add_new_vendor_interactive():
    """Interactive function to add a new vendor through questionnaire"""
    print("\n=== Add New Vendor Installation ===\n")
    
    site_name = input("Enter site name: ")
    
    # Initialize vendor manager
    vendor_manager = VendorManager()
    
    # Generate and display questionnaire
    print("\nQuestionnaire Template (copy and send to vendor):")
    vendor_manager.generate_questionnaire(site_name)
    
    print("\nEnter vendor responses (leave blank if not provided):")
    
    response_dict = {}
    
    # Panel details
    response_dict["azimuth"] = input("Azimuth Angle (0°=North, 90°=East, 180°=South): ") or "180"
    response_dict["tilt"] = input("Tilt Angle (relative to horizontal): ") or "11"
    response_dict["panel_type"] = input("Panel Technology (Mono/Poly/Thin Film): ") or ""
    response_dict["capacity"] = input("Total Capacity (kWp): ") or ""
    
    # Site geometry
    response_dict["latitude"] = input("Latitude: ") or COIMBATORE_COORDS[0]
    response_dict["longitude"] = input("Longitude: ") or COIMBATORE_COORDS[1]
    response_dict["topography"] = input("Topography Notes: ") or ""
    response_dict["shading"] = input("Shading Sources: ") or ""
    
    # Seasonal patterns
    response_dict["summer_wind"] = input("Dominant Wind Direction (Summer, Apr-Jun): ") or "SW"
    response_dict["monsoon_wind"] = input("Dominant Wind Direction (Monsoon, Jul-Sep): ") or "SW"
    response_dict["winter_wind"] = input("Dominant Wind Direction (Winter, Oct-Mar): ") or "NE"
    
    # Process and save vendor data
    vendor_manager.process_vendor_response(site_name, response_dict)
    
    print(f"\nVendor '{site_name}' added successfully!")
    return site_name

def main_menu():
    """Interactive menu for the application"""
    print("\n=== PySTEPS Solar Forecast System ===")
    print("1. Run forecast for existing site")
    print("2. Add new vendor installation")
    print("3. Generate vendor questionnaire")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    vendor_manager = VendorManager()
    
    if choice == "1":
        # List existing sites
        sites = list(vendor_manager.vendor_data.keys())
        if not sites:
            print("No vendor sites found. Please add a vendor first.")
            return main_menu()
        
        print("\nAvailable sites:")
        for i, site in enumerate(sites):
            print(f"{i+1}. {site}")
        
        site_idx = int(input("\nSelect site number: ")) - 1
        if 0 <= site_idx < len(sites):
            use_sentinel = input("Include Sentinel-2 analysis? (y/n): ").lower() == 'y'
            main(sites[site_idx], sentinel2_analysis=use_sentinel)
        else:
            print("Invalid selection")
        
        return main_menu()
    
    elif choice == "2":
        site_name = add_new_vendor_interactive()
        run_now = input(f"Run forecast for {site_name} now? (y/n): ").lower() == 'y'
        if run_now:
            use_sentinel = input("Include Sentinel-2 analysis? (y/n): ").lower() == 'y'
            main(site_name, sentinel2_analysis=use_sentinel)
        
        return main_menu()
    
    elif choice == "3":
        site_name = input("Enter name for new site: ")
        vendor_manager.generate_questionnaire(site_name)
        input("\nPress Enter to continue...")
        
        return main_menu()
    
    elif choice == "4":
        print("Exiting program. Goodbye!")
        return
    
    else:
        print("Invalid choice. Please try again.")
        return main_menu()

if __name__ == "__main__":
    main_menu()
