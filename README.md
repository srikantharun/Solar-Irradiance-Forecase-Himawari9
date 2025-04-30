# Solar-Irradiance-Forecase-Himawari9
A comprehensive system for analyzing Himawari-9 satellite imagery to forecast cloud movement and predict impacts on solar panel irradiance in the Coimbatore region.

## Overview

This system consists of two main components using Collab notebook:

1. **Cell 1: Himawari Cloud Analysis** - Downloads and processes satellite imagery, performs motion estimation, and generates cloud cover forecasts
2. **Cell 2: Solar Irradiance Impact** - Analyzes cloud forecasts to assess potential impacts on solar panel performance and generates alerts

The workflow integrates satellite data processing, optical flow analysis, stochastic nowcasting, and impact assessment to create a complete solar irradiance forecasting system.

## Features

- Real-time processing of Himawari-9 satellite imagery
- Region-specific analysis for the Coimbatore area (11.0168°N, 76.9558°E)
- Lucas-Kanade optical flow for cloud motion estimation
- PySteps STEPS method for stochastic ensemble forecasting
- Automated cloud coverage detection and impact classification
- Impact severity alerts with timing information
- Detailed visualizations of cloud movement and impact levels
- Minute-by-minute solar irradiance impact forecasts

## Requirements

- Python 3.7+
- Required libraries:
  - requests
  - numpy
  - matplotlib
  - pillow (PIL)
  - rasterio
  - pysteps (version 1.17.0)
  - scikit-image (optional, for Otsu thresholding)

## Installation

```bash
pip install requests numpy matplotlib pillow rasterio pysteps scikit-image
```

## Usage

The system is designed to be run in a Jupyter notebook or similar environment with two main cells:

### Cell 1: Himawari Cloud Analysis

This cell downloads the latest Himawari-9 satellite images and processes them to generate cloud movement forecasts for the Coimbatore region.

```python
# Import the required libraries and functions from the code...

# Run the cloud analysis
downloaded_files = download_latest_himawari_jpegs(num_images=3, hours_back=4)
image_arrays = [process_jpeg_to_array(file, crop_to_coimbatore=True) for file in downloaded_files]

# Save as TIF files and perform pysteps analysis
clipped_files = []
for i, img_array in enumerate(image_arrays):
    output_filename = f"coimbatore_image{i+1}_clipped.tif"
    # Save TIF files...
    clipped_files.append(output_filename)

# Run pysteps analysis
motion_field, nowcast = pysteps_analysis(clipped_files, forecast_minutes=60, timestep=10, region_name="Coimbatore")

# Create visualization
create_animation(nowcast, timestep=10, region_name="Coimbatore")
```

### Information solar-forecast-notebook.py

####1. Vendor Information Integration

The enhanced notebook includes a complete vendor management system that:

- Processes vendor questionnaire responses (from the template you provided)
- Stores installation details like panel azimuth, tilt, and site-specific information
- Converts seasonal wind directions to radians for mathematical analysis
- Uses these parameters to customize the optical flow calculations

#### 2. Directionally-Weighted Impact Assessment
I've significantly improved the impact assessment by:

- Calculating angular differences between cloud motion and critical wind directions
- Applying cosine weighting to better estimate impact severity
- Accounting for panel orientation relative to cloud movement
- Adding tilt factor adjustments (higher tilt = less impact)

#### 3. Enhanced Visualization
The visualization components now show:

- Directional vectors for cloud motion, panel orientation, and seasonal wind
- Weighted impact values based on all vendor parameters
- More detailed forecasts with multiple visualization formats

#### 4. Sentinel-2 Integration
The notebook now includes functions to:

- Download recent Sentinel-2 data for the installation site
- Extract terrain and land cover information
- Adjust cloud impact predictions based on local topography

#### 5. Interactive Interface
The system now features an interactive menu for:

- Managing multiple vendor installations
- Generating and processing vendor questionnaires
- Running forecasts for different sites with customizable parameters



## Output Files

The system generates several output files:

- `coimbatore_cloud_mask.png` - Initial cloud detection for the Coimbatore region
- `coimbatore_motion_field.png` - Visualization of cloud movement vectors
- `coimbatore_cloud_cover_animation.mp4` - Animation of forecasted cloud movement
- `coimbatore_solar_impact_forecast.png` - Visualization of solar impact levels over time
- `coimbatore_region.png` - Reference image showing the extracted Coimbatore region

## Customization

### Region Settings

You can modify the bounding box to focus on a different area:

```python
CUSTOM_COORDS = (latitude, longitude)  # Center coordinates
CUSTOM_BBOX = {
    "min_lat": CUSTOM_COORDS[0] - radius,
    "max_lat": CUSTOM_COORDS[0] + radius,
    "min_lon": CUSTOM_COORDS[1] - radius,
    "max_lon": CUSTOM_COORDS[1] + radius
}
```

### Impact Thresholds

Adjust the cloud detection and impact thresholds to match local conditions:

```python
# More sensitive cloud detection
impact_assessment = assess_cloud_impact_on_solar(nowcast, threshold=0.25, impact_threshold=0.2)

# Less sensitive (only alert for significant coverage)
impact_assessment = assess_cloud_impact_on_solar(nowcast, threshold=0.35, impact_threshold=0.3)
```

## Technical Details

### Cloud Detection

The system uses a brightness threshold on visible-band (B03) satellite imagery to detect clouds:

```python
cloud_mask = image_array > threshold  # Default threshold = 0.3
```

### Impact Classification

Cloud coverage is classified into impact levels based on percentage of area covered:

| Coverage Range | Impact Level | Description                                    |
|----------------|--------------|------------------------------------------------|
| ≥75%           | Severe       | Major reduction in solar irradiance expected   |
| 50-74.9%       | High         | Significant reduction in solar irradiance      |
| 25-49.9%       | Moderate     | Moderate reduction in solar irradiance         |
| 10-24.9%       | Low          | Minor reduction in solar irradiance possible   |
| <10%           | Minimal      | No significant impact on solar irradiance      |

### PySteps Configuration

The system is optimized for pysteps version 1.17.0 with these key parameters:

```python
nowcast = nowcast_method(
    data_array,
    motion_field,
    timesteps=n_timesteps,
    mask_method=None,  # Disable precipitation masking for visible-band imagery
    vel_pert_method=None,  # Disable velocity perturbation
    n_cascade_levels=6,
    ar_order=2,
    noise_method="nonparametric"
)
```

## Troubleshooting

### Common Issues

1. **PySteps API Changes**
   - The code is specifically tuned for pysteps 1.17.0
   - Parameter names may differ in other versions

2. **Satellite Image Availability**
   - The system uses publicly available Himawari-9 images from JMA
   - Image availability depends on JMA's update schedule

3. **Low Motion Detection**
   - If no significant cloud movement is detected, the system adds small random perturbations
   - Check the motion magnitude values in the output logs

4. **Animation Errors**
   - If standard animation fails, the system will save individual frames
   - Check the "forecast_frames" directory for these images

### Logs

Key log messages to watch for:

- `Motion magnitude - Mean: X, Max: Y` - Should show non-zero values
- `Cloud cover over Coimbatore region: Z%` - Current cloud coverage percentage
- `Using estimated pixel size: X km` - Should be a reasonable value (typically 0.2-1.0 km)

## Workflow 
![image](https://github.com/user-attachments/assets/c85cd11c-0363-4f3a-9e8d-0f621391e251)

Shown below is process flow diagram

![image](https://github.com/user-attachments/assets/7a36ff6e-f0e3-4f7d-87f7-dc7f21e1590b)


## Further Development

Potential enhancements for future versions:

1. Integration with actual solar panel monitoring systems
2. Incorporation of cloud type classification (thin vs. thick clouds)
3. Machine learning-based cloud detection for improved accuracy
4. Web-based dashboard for real-time monitoring
5. Multi-region analysis for distributed solar installations

## References

- PySteps library: [https://github.com/pySTEPS/pysteps](https://github.com/pySTEPS/pysteps)
- Himawari-9 satellite: [https://www.data.jma.go.jp/mscweb/en/himawari89/](https://www.data.jma.go.jp/mscweb/en/himawari89/)
- Lucas-Kanade optical flow: B.D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," in Proc. DARPA Image Understanding Workshop, 1981, pp. 121–130.

## License

This project is licensed under the MIT License.
