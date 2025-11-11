# Seal Strip Inference Pipeline

A portable, self-contained inference package for seal strip analysis. This package includes all necessary models, utilities, and code to run inference on seal strip images.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Verify the models are in the `models/` directory:
   - `best_od_cropper_epoch90.pt` - OD bounding box regressor
   - `checkpoint_epoch108.pt` - Heatmap-based point predictor
   - `best_strip_classifier.pth` - Binary classifier (OK/Short)

## Usage

### Python API

```python
from inference_pipeline import SealStripInference

# Initialize pipeline
pipeline = SealStripInference(
    models_dir="./models",
    device="auto"  # or "cuda" or "cpu"
)

# Process an image
result = pipeline.process_image("path/to/image.jpg")

# Access results
print(f"Prediction: {result['prediction']}")  # 'OK' or 'Short'
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probability: {result['probability']:.3f}")
print(f"OD BBox: {result['od_bbox']}")  # (x1, y1, x2, y2)

# Get intermediate results
result = pipeline.process_image("image.jpg", return_intermediates=True)
unwrapped_strip = result['unwrapped_strip']  # numpy array (BGR)
od_crop = result['od_crop']  # cropped OD region

# Process multiple images and create visualization grid
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = pipeline.process_batch(image_paths, output_path="validation_grid.png")
# This creates a PNG grid showing all 5 pipeline steps for each image
```

### Command Line

**Single Image:**
```bash
# Basic usage
python inference_pipeline.py path/to/image.jpg

# Specify models directory
python inference_pipeline.py image.jpg --models_dir ./my_models

# Use CPU explicitly
python inference_pipeline.py image.jpg --device cpu

# Save unwrapped strip
python inference_pipeline.py image.jpg --output unwrapped.jpg

# Save intermediate results
python inference_pipeline.py image.jpg --save_intermediates
```

**Multiple Images (with visualization grid):**
```bash
# Process multiple images and create visualization grid
python inference_pipeline.py image1.jpg image2.jpg image3.jpg --output validation_grid.png

# Process all images in a directory
python inference_pipeline.py --image_dir ./images --output validation_grid.png

# Process random N images from directory
python inference_pipeline.py --image_dir ./images --num_images 10 --output validation_grid.png
```

## Pipeline Steps

The inference pipeline performs the following steps:

1. **OD Detection**: Predicts bounding box of the OD region using a 160×160 downsampled image
2. **Cropping & Resizing**: Crops the OD region and resizes to 512×224
3. **Point Prediction**: Predicts 10 edge point pairs using a heatmap-based model
4. **Unwrapping**: Unwraps the cylindrical strip using RANSAC circle fitting
5. **Classification**: Classifies the unwrapped strip as "OK" or "Short"

## Output Format

The `process_image()` method returns a dictionary with:

- `prediction`: `"OK"` or `"Short"` (string)
- `probability`: Probability of being "Short" (float, 0-1)
- `confidence`: Confidence in the prediction (float, 0-1)
- `threshold`: Classification threshold used (float)
- `od_bbox`: Bounding box coordinates `(x1, y1, x2, y2)` (tuple)

If `return_intermediates=True`, additional fields:
- `od_crop`: Cropped OD region (numpy array, BGR format)
- `unwrapped_strip`: Unwrapped strip image (numpy array, BGR format, 512×224)
- `od_resized`: Resized OD region (numpy array, BGR format, 512×224)
- `points_upper`: Upper edge y-coordinates (numpy array)
- `points_lower`: Lower edge y-coordinates (numpy array)
- `point_xs`: X-coordinates for points (list)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full list

## Performance

- **GPU**: ~63ms per image (15-16 images/sec)
- **CPU**: ~95ms per image (10-11 images/sec)

## File Structure

```
inference_export/
├── inference_pipeline.py  # Main inference script
├── geom.py                 # Geometry utilities
├── models/                 # Model files
│   ├── best_od_cropper_epoch90.pt
│   ├── checkpoint_epoch108.pt
│   └── best_strip_classifier.pth
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Notes

- Input images should be 640×640 pixels (other sizes work but may affect accuracy)
- The pipeline automatically handles device selection (GPU if available, else CPU)
- Models are loaded once during initialization for efficient batch processing
- The unwrapping step uses RANSAC circle fitting, which may fail on very noisy images


