#!/usr/bin/env python3
"""
Example usage of the Seal Strip Inference Pipeline
"""

from inference_pipeline import SealStripInference
import cv2
from pathlib import Path

# Initialize pipeline
print("Initializing pipeline...")
pipeline = SealStripInference(
    models_dir="./models",
    device="auto"  # Use GPU if available, else CPU
)

# Process a single image
image_path = "path/to/your/image.jpg"
print(f"\nProcessing: {image_path}")

result = pipeline.process_image(image_path, return_intermediates=True)

# Print results
print(f"\n{'='*60}")
print("Results:")
print(f"{'='*60}")
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"OD BBox: {result['od_bbox']}")

# Save intermediate results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

cv2.imwrite(str(output_dir / "unwrapped_strip.jpg"), result['unwrapped_strip'])
cv2.imwrite(str(output_dir / "od_crop.jpg"), result['od_crop'])

print(f"\n✓ Saved results to: {output_dir}/")

# Process multiple images
print("\n" + "="*60)
print("Processing multiple images...")
print("="*60)

image_paths = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

for img_path in image_paths:
    if Path(img_path).exists():
        result = pipeline.process_image(img_path)
        print(f"{Path(img_path).name}: {result['prediction']} (conf: {result['confidence']:.3f})")
    else:
        print(f"{img_path}: File not found")



