#!/usr/bin/env python3
"""
Standalone Inference Pipeline for Seal Strip Analysis

This is a portable, self-contained inference script that can be used in any project.
It includes all necessary model definitions and processing functions.

Usage:
    from inference_pipeline import SealStripInference
    
    pipeline = SealStripInference(
        models_dir="./models",
        device="auto"  # or "cuda" or "cpu"
    )
    
    result = pipeline.process_image("path/to/image.jpg")
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from typing import Dict, Tuple, Optional
import time

# Import geom utilities (included in same package)
from geom import select_columns_from_base, fit_circle_ransac

# ===== Configuration =====
OD_INPUT_SIZE = 160
OD_PADDING_X = 0.0
OD_PADDING_Y = 0.15
POINT_IMG_W = 512
POINT_IMG_H = 224
NUM_COLS = 10
BAND_PCT = 0.96
OUT_H = 224
OUT_W = 512
CLASSIFIER_INPUT_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===== Model Definitions =====

class LightweightBBRegressor(nn.Module):
    """Lightweight CNN for bounding box regression."""
    def __init__(self, pretrained=True):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(mobilenet.last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class EdgeHeatmapRegressor(nn.Module):
    """Heatmap-based edge point predictor."""
    def __init__(self, num_columns=10, img_h=224, img_w=512, use_simple_head=False):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_columns = num_columns
        self.img_h = img_h
        self.img_w = img_w
        
        # Output: [2, num_columns, img_h] (upper and lower heatmaps)
        if use_simple_head:
            # Simple head for compatibility with old checkpoints
            self.head = nn.Linear(mobilenet.last_channel, 2 * num_columns * img_h)
        else:
            # Sequential head with intermediate layer
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(mobilenet.last_channel, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 2 * num_columns * img_h)
            )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = x.view(-1, 2, self.num_columns, self.img_h)
        return x


class ODClassificationCNN(nn.Module):
    """Binary classifier for OK/Short classification."""
    def __init__(self, num_classes=1, use_simple_head=False):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        if use_simple_head:
            # Simple head for compatibility with old checkpoints
            self.classifier = nn.Linear(mobilenet.last_channel, num_classes)
        else:
            # Sequential head with intermediate layer
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(mobilenet.last_channel, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===== Utility Functions =====

def extract_points_from_heatmap(heatmap: torch.Tensor, num_columns: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract edge points from heatmap predictions.
    
    Args:
        heatmap: Tensor of shape [2, num_columns, img_h] (upper and lower heatmaps)
        num_columns: Number of columns
        img_h: Image height
    
    Returns:
        upper_y, lower_y: Arrays of y-coordinates for each column
    """
    # Apply sigmoid to get probabilities
    heatmap_prob = torch.sigmoid(heatmap)
    
    upper_y = np.full(num_columns, np.nan, dtype=np.float32)
    lower_y = np.full(num_columns, np.nan, dtype=np.float32)
    
    # Extract points using argmax
    for i in range(num_columns):
        upper_hm = heatmap_prob[0, i, :].cpu().numpy()
        lower_hm = heatmap_prob[1, i, :].cpu().numpy()
        
        upper_max_idx = np.argmax(upper_hm)
        lower_max_idx = np.argmax(lower_hm)
        
        if upper_hm[upper_max_idx] > 0.1:  # Minimum confidence threshold
            upper_y[i] = float(upper_max_idx)
        if lower_hm[lower_max_idx] > 0.1:
            lower_y[i] = float(lower_max_idx)
    
    return upper_y, lower_y


def norm_to_pixels(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized bbox (cx, cy, w, h) to pixel coordinates (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    
    return x1, y1, x2, y2


def unwrap_with_band_mask(
    img_bgr: np.ndarray,
    xs: list,
    upper_y: np.ndarray,
    lower_y: np.ndarray,
    out_h: int = 224,
    out_w: int = 512
) -> np.ndarray:
    """
    Unwrap image using band_mask method.
    
    Args:
        img_bgr: Input image (BGR format, resized to model input size)
        xs: List of x-coordinates where points were predicted
        upper_y: Upper edge y-coordinates (same length as xs)
        lower_y: Lower edge y-coordinates (same length as xs)
        out_h: Output strip height
        out_w: Output strip width
    
    Returns:
        Unwrapped strip image (BGR format)
    """
    h, w = img_bgr.shape[:2]
    
    # Fit RANSAC circles
    upper_pts = [(float(xs[i]), float(upper_y[i])) for i in range(len(xs)) if not np.isnan(upper_y[i])]
    lower_pts = [(float(xs[i]), float(lower_y[i])) for i in range(len(xs)) if not np.isnan(lower_y[i])]
    
    fu = fit_circle_ransac(upper_pts)
    fl = fit_circle_ransac(lower_pts)
    if fu is None or fl is None:
        raise RuntimeError('RANSAC circle fit failed')
    
    (cux, cuy), ru, _ = fu
    (clx, cly), rl, _ = fl
    
    # Build band_mask
    x_min, x_max = int(min(xs)), int(max(xs))
    yy, xx = np.mgrid[0:h, 0:w]
    du = np.hypot(xx - cux, yy - cuy)
    dl = np.hypot(xx - clx, yy - cly)
    band_x = (xx >= x_min) & (xx <= x_max)
    band_mask = ((dl <= rl) & (du >= ru) & band_x).astype(np.uint8)
    
    # Find columns with band
    cols_with_band = [x for x in range(x_min, x_max + 1) if np.any(band_mask[:, x] > 0)]
    if len(cols_with_band) < 2:
        raise RuntimeError('Band too thin to unwrap')
    
    x_left = min(cols_with_band)
    x_right = max(cols_with_band)
    
    # Create mapping
    xs_real = np.linspace(float(x_left), float(x_right), out_w, dtype=np.float32)
    frac_h = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
    map_x = np.zeros((out_h, out_w), dtype=np.float32)
    map_y = np.zeros((out_h, out_w), dtype=np.float32)
    
    for j, xr in enumerate(xs_real):
        du_r = np.hypot(xr - cux, yy - cuy)
        dl_r = np.hypot(xr - clx, yy - cly)
        mask_col = ((dl_r <= rl) & (du_r >= ru)).astype(np.uint8)
        col_mask = mask_col[:, int(np.clip(xr, 0, w - 1))]
        if np.sum(col_mask) == 0:
            continue
        y_min = np.argmax(col_mask)
        y_max = len(col_mask) - 1 - np.argmax(col_mask[::-1])
        if y_max <= y_min:
            continue
        for i, frac in enumerate(frac_h):
            y_map = y_min + frac * (y_max - y_min)
            map_y[i, j] = y_map
            map_x[i, j] = xr
    
    # Remap image
    strip = cv2.remap(
        img_bgr, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return strip


# ===== Main Inference Class =====

class SealStripInference:
    """
    Main inference class for seal strip analysis.
    
    Example:
        pipeline = SealStripInference(models_dir="./models")
        result = pipeline.process_image("image.jpg")
    """
    
    def __init__(
        self,
        models_dir: str = "./models",
        od_model_path: Optional[str] = None,
        point_model_path: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            models_dir: Directory containing model files
            od_model_path: Path to OD model (default: models_dir/best_od_cropper_epoch90.pt)
            point_model_path: Path to point predictor model (default: models_dir/checkpoint_epoch108.pt)
            classifier_model_path: Path to classifier model (default: models_dir/best_strip_classifier.pth)
            device: Device to use ("auto", "cuda", or "cpu")
        """
        # Determine device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == "cuda":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
        else:
            self.device = torch.device('cpu')
        
        models_dir = Path(models_dir)
        
        # Set model paths
        if od_model_path is None:
            od_model_path = models_dir / 'best_od_cropper_epoch90.pt'
        if point_model_path is None:
            point_model_path = models_dir / 'checkpoint_epoch108.pt'
        if classifier_model_path is None:
            classifier_model_path = models_dir / 'best_strip_classifier.pth'
        
        # Load models
        print(f"Loading models on {self.device}...")
        self._load_models(od_model_path, point_model_path, classifier_model_path)
        print("✓ All models loaded successfully!")
    
    def _load_models(self, od_path, point_path, classifier_path):
        """Load all models with comprehensive checkpoint handling."""
        # OD Bounding Box Regressor (160x160)
        self.od_model = LightweightBBRegressor(pretrained=False).to(self.device)
        od_checkpoint = torch.load(od_path, map_location=self.device, weights_only=False)
        if isinstance(od_checkpoint, dict):
            if 'model_state_dict' in od_checkpoint:
                self.od_model.load_state_dict(od_checkpoint['model_state_dict'])
            elif 'model' in od_checkpoint:
                self.od_model.load_state_dict(od_checkpoint['model'])
            elif 'state_dict' in od_checkpoint:
                self.od_model.load_state_dict(od_checkpoint['state_dict'])
            else:
                # Try direct loading
                self.od_model.load_state_dict(od_checkpoint)
        else:
            # Direct state dict
            self.od_model.load_state_dict(od_checkpoint)
        self.od_model.eval()
        print(f"✓ OD model loaded from {od_path}")
        
        # Heatmap-based Point Predictor (10 columns)
        # First, check checkpoint structure to determine which model to use
        point_state = torch.load(point_path, map_location=self.device, weights_only=False)
        
        # Extract state dict from checkpoint
        if isinstance(point_state, dict):
            if 'model' in point_state:
                state_dict = point_state['model']
            elif 'model_state_dict' in point_state:
                state_dict = point_state['model_state_dict']
            elif 'state_dict' in point_state:
                state_dict = point_state['state_dict']
            else:
                state_dict = point_state
        else:
            state_dict = point_state
        
        # Check if checkpoint uses simple head (old format) or Sequential head (new format)
        use_simple_head = 'head.weight' in state_dict and 'head.1.weight' not in state_dict
        
        # Create model with appropriate head structure
        self.point_model = EdgeHeatmapRegressor(
            num_columns=NUM_COLS, 
            img_h=POINT_IMG_H, 
            img_w=POINT_IMG_W,
            use_simple_head=use_simple_head
        ).to(self.device)
        
        # Load state dict
        missing_keys, unexpected_keys = self.point_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            non_head_missing = [k for k in missing_keys if not k.startswith('head.')]
            if non_head_missing:
                print(f"Warning: Missing non-head keys (using initialized values): {non_head_missing[:3]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected_keys[:3]}...")
        if use_simple_head:
            print("Note: Using simple head structure (old checkpoint format)")
        
        self.point_model.eval()
        print(f"✓ Heatmap point predictor loaded from {point_path}")
        
        # Classifier
        # First, check checkpoint structure to determine which model to use
        classifier_ckpt = torch.load(classifier_path, map_location=self.device, weights_only=False)
        
        # Extract state dict from checkpoint
        if isinstance(classifier_ckpt, dict):
            if 'state_dict' in classifier_ckpt:
                classifier_state = classifier_ckpt['state_dict']
            elif 'model_state_dict' in classifier_ckpt:
                classifier_state = classifier_ckpt['model_state_dict']
            elif 'model' in classifier_ckpt:
                classifier_state = classifier_ckpt['model']
            else:
                classifier_state = classifier_ckpt
        else:
            classifier_state = classifier_ckpt
        
        # Check if checkpoint uses simple head (old format) or Sequential head (new format)
        # Old format has 'classifier.1.weight' with shape [1, 1280], new format has 'classifier.1.weight' with shape [128, 1280]
        use_simple_head = False
        remap_keys = False
        if 'classifier.1.weight' in classifier_state:
            # Check shape - old format has [1, 1280], new format has [128, 1280]
            if classifier_state['classifier.1.weight'].shape[0] == 1:
                use_simple_head = True
                remap_keys = True  # Need to remap classifier.1.* -> classifier.*
        elif 'classifier.weight' in classifier_state:
            # Direct classifier.weight (very old format)
            use_simple_head = True
        
        # Create model with appropriate head structure
        self.classifier = ODClassificationCNN(num_classes=1, use_simple_head=use_simple_head).to(self.device)
        
        # Remap keys if needed (classifier.1.* -> classifier.* for simple head)
        if remap_keys:
            remapped_state = {}
            for key, value in classifier_state.items():
                if key == 'classifier.1.weight':
                    remapped_state['classifier.weight'] = value
                elif key == 'classifier.1.bias':
                    remapped_state['classifier.bias'] = value
                elif not key.startswith('classifier.1.'):
                    # Keep all other keys as-is
                    remapped_state[key] = value
            classifier_state = remapped_state
        
        # Load state dict
        missing_keys, unexpected_keys = self.classifier.load_state_dict(classifier_state, strict=False)
        if missing_keys:
            non_classifier_missing = [k for k in missing_keys if not k.startswith('classifier.')]
            if non_classifier_missing:
                print(f"Warning: Missing non-classifier keys (using initialized values): {non_classifier_missing[:3]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in classifier checkpoint (ignored): {unexpected_keys[:3]}...")
        if use_simple_head:
            print("Note: Using simple classifier head structure (old checkpoint format)")
        
        self.classifier.eval()
        self.threshold = classifier_ckpt.get('threshold', 0.35) if isinstance(classifier_ckpt, dict) else 0.35
        print(f"✓ Classifier loaded from {classifier_path} (threshold: {self.threshold:.3f})")
    
    def process_image(
        self,
        img_path: str,
        return_intermediates: bool = False
    ) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            img_path: Path to input image
            return_intermediates: If True, return intermediate results (od_crop, unwrapped_strip, etc.)
        
        Returns:
            Dictionary with:
                - 'prediction': 'OK' or 'Short'
                - 'probability': float (probability of being "Short")
                - 'confidence': float (confidence in prediction)
                - 'threshold': float (classification threshold)
                - 'od_bbox': (x1, y1, x2, y2) bounding box coordinates
                - Additional fields if return_intermediates=True:
                    - 'od_crop': cropped OD region
                    - 'unwrapped_strip': unwrapped strip image
                    - 'points_upper': upper edge points
                    - 'points_lower': lower edge points
        """
        # Load image
        img_bgr_orig = cv2.imread(img_path)
        if img_bgr_orig is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        orig_h, orig_w = img_bgr_orig.shape[:2]
        
        # Step 1: Predict OD bounding box (160x160)
        img_small = cv2.resize(img_bgr_orig, (OD_INPUT_SIZE, OD_INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        img_rgb_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_pil_small = Image.fromarray(img_rgb_small)
        
        transform_od = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        tensor_od = transform_od(img_pil_small).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_bbox = self.od_model(tensor_od).cpu().numpy()[0]
        
        # Apply sigmoid to constrain outputs to [0, 1] range
        cx, cy, w, h = 1 / (1 + np.exp(-pred_bbox))  # Sigmoid
        
        # Ensure minimum size
        w = max(w, 0.1)
        h = max(h, 0.1)
        
        # Ensure width and height don't exceed image bounds
        w = min(w, 2.0 * min(cx, 1.0 - cx))
        h = min(h, 2.0 * min(cy, 1.0 - cy))
        
        x1, y1, x2, y2 = norm_to_pixels(cx, cy, w, h, orig_w, orig_h)
        
        # Ensure valid bbox
        if x1 >= x2:
            x1, x2 = max(0, x2 - 10), min(orig_w, x1 + 10)
        if y1 >= y2:
            y1, y2 = max(0, y2 - 10), min(orig_h, y1 + 10)
        
        if x2 - x1 < 10:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - 5)
            x2 = min(orig_w, center_x + 5)
        if y2 - y1 < 10:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - 5)
            y2 = min(orig_h, center_y + 5)
        
        # Add asymmetric padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = int(bbox_w * OD_PADDING_X)
        pad_y = int(bbox_h * OD_PADDING_Y)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(orig_w, x2 + pad_x)
        y2 = min(orig_h, y2 + pad_y)
        
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Step 2: Crop OD region
        if x1 >= x2 or y1 >= y2 or x2 - x1 < 1 or y2 - y1 < 1:
            raise ValueError(f"Invalid bbox coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        
        od_crop = img_bgr_orig[y1:y2, x1:x2]
        if od_crop.size == 0:
            raise ValueError("Empty OD crop!")
        
        # Step 3: Resize for point prediction
        od_resized = cv2.resize(od_crop, (POINT_IMG_W, POINT_IMG_H), interpolation=cv2.INTER_LINEAR)
        
        # Step 4: Predict edge points from heatmaps
        xs_all = select_columns_from_base(
            POINT_IMG_W, base_num=24, use_num=20, band_pct=BAND_PCT, drop_ends=2
        )
        xs = xs_all[::2]  # Take every 2nd column for 10 columns
        
        img_rgb = cv2.cvtColor(od_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor_point = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            heatmaps_pred = self.point_model(tensor_point)
            heatmaps_pred = heatmaps_pred.squeeze(0)
        
        # Extract points from heatmaps
        upper_y, lower_y = extract_points_from_heatmap(heatmaps_pred, NUM_COLS, POINT_IMG_H)
        
        # Step 5: Unwrap strip
        try:
            unwrapped_strip = unwrap_with_band_mask(
                od_resized, xs, upper_y, lower_y, OUT_H, OUT_W
            )
        except Exception as e:
            raise RuntimeError(f"Unwrapping failed: {e}")
        
        # Step 6: Classify unwrapped strip
        strip_rgb = cv2.cvtColor(unwrapped_strip, cv2.COLOR_BGR2RGB)
        strip_pil = Image.fromarray(strip_rgb)
        
        transform_classifier = transforms.Compose([
            transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        tensor_classifier = transform_classifier(strip_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logit = self.classifier(tensor_classifier).item()
            prob = 1 / (1 + np.exp(-logit))  # Sigmoid
        
        prediction = "Short" if prob >= self.threshold else "OK"
        confidence = prob if prob >= self.threshold else 1 - prob
        
        # Create visualization image (with bbox overlay)
        vis_img = img_bgr_orig.copy()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label_text = f"{prediction} ({prob:.3f})"
        cv2.putText(vis_img, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Build result
        result = {
            'prediction': prediction,
            'probability': float(prob),
            'confidence': float(confidence),
            'threshold': float(self.threshold),
            'od_bbox': (x1, y1, x2, y2),
            'visualization': vis_img
        }
        
        if return_intermediates:
            result.update({
                'od_crop': od_crop,
                'unwrapped_strip': unwrapped_strip,
                'od_resized': od_resized,
                'points_upper': upper_y,
                'points_lower': lower_y,
                'point_xs': xs
            })
        
        return result
    
    def process_batch(
        self,
        image_paths: list,
        output_path: Optional[str] = None
    ) -> list:
        """
        Process multiple images and optionally create visualization grid.
        
        Args:
            image_paths: List of paths to input images
            output_path: If provided, save visualization grid to this path
        
        Returns:
            List of result dictionaries (one per image)
        """
        results = []
        
        for img_path in image_paths:
            try:
                result = self.process_image(img_path, return_intermediates=True)
                result['img_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if output_path and results:
            self.create_visualization_grid(results, output_path)
        
        return results
    
    def create_visualization_grid(self, results_list: list, output_path: str):
        """
        Create visualization grid showing all pipeline steps for multiple images.
        
        Args:
            results_list: List of result dictionaries from process_image()
            output_path: Path to save the visualization PNG
        """
        num_images = len(results_list)
        
        # Steps: 1) Original + bbox, 2) OD crop, 3) OD with points, 4) Unwrapped, 5) Classification
        cols = 5
        rows = num_images
        
        cell_w = 300
        cell_h = 300
        spacing = 10
        label_h = 30
        
        canvas_w = cols * (cell_w + spacing) + spacing
        canvas_h = rows * (cell_h + label_h + spacing) + spacing
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        step_labels = [
            "1. Original + OD BBox",
            "2. OD Crop",
            "3. OD with Points",
            "4. Unwrapped Strip",
            "5. Classification"
        ]
        
        for row_idx, result in enumerate(results_list):
            y_offset = row_idx * (cell_h + label_h + spacing) + spacing
            
            # Step 1: Original image with bounding box
            orig_img = result['visualization'].copy()
            orig_img = cv2.resize(orig_img, (cell_w, cell_h))
            x_offset = spacing
            canvas[y_offset + label_h:y_offset + label_h + cell_h, x_offset:x_offset + cell_w] = orig_img
            cv2.putText(canvas, step_labels[0], (x_offset, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Step 2: OD crop (maintain aspect ratio)
            od_crop = result['od_crop'].copy()
            crop_h, crop_w = od_crop.shape[:2]
            scale = min(cell_w / crop_w, cell_h / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            od_crop_resized = cv2.resize(od_crop, (new_w, new_h))
            x_offset = spacing + (cell_w + spacing)
            y_center = y_offset + label_h + cell_h // 2
            x_center = x_offset + cell_w // 2
            y_start = y_center - new_h // 2
            x_start = x_center - new_w // 2
            canvas[y_start:y_start + new_h, x_start:x_start + new_w] = od_crop_resized
            cv2.putText(canvas, step_labels[1], (x_offset, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Step 3: OD with predicted points
            od_with_points = result['od_resized'].copy()
            od_h, od_w = od_with_points.shape[:2]
            
            # Draw points
            if 'points_upper' in result and 'points_lower' in result and 'point_xs' in result:
                upper_points = result['points_upper']
                lower_points = result['points_lower']
                xs = result['point_xs']
                
                if hasattr(upper_points, '__len__') and len(upper_points) > 0:
                    for i, (x, y_upper, y_lower) in enumerate(zip(xs, upper_points, lower_points)):
                        if np.isnan(y_upper) or np.isnan(y_lower):
                            continue
                        x_int = int(round(x))
                        y_upper_int = int(round(y_upper))
                        y_lower_int = int(round(y_lower))
                        
                        if 0 <= x_int < od_w and 0 <= y_upper_int < od_h and 0 <= y_lower_int < od_h:
                            cv2.circle(od_with_points, (x_int, y_upper_int), 4, (0, 255, 0), -1)
                            cv2.circle(od_with_points, (x_int, y_lower_int), 4, (0, 0, 255), -1)
                            cv2.line(od_with_points, (x_int, y_upper_int), (x_int, y_lower_int), (255, 0, 0), 2)
            
            # Resize maintaining aspect ratio
            points_h, points_w = od_with_points.shape[:2]
            scale = min(cell_w / points_w, cell_h / points_h)
            new_w = int(points_w * scale)
            new_h = int(points_h * scale)
            od_with_points_resized = cv2.resize(od_with_points, (new_w, new_h))
            x_offset = spacing + 2 * (cell_w + spacing)
            y_center = y_offset + label_h + cell_h // 2
            x_center = x_offset + cell_w // 2
            y_start = y_center - new_h // 2
            x_start = x_center - new_w // 2
            canvas[y_start:y_start + new_h, x_start:x_start + new_w] = od_with_points_resized
            cv2.putText(canvas, step_labels[2], (x_offset, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Step 4: Unwrapped strip
            unwrapped = result['unwrapped_strip'].copy()
            unwrapped_h, unwrapped_w = unwrapped.shape[:2]
            scale = min(cell_w / unwrapped_w, cell_h / unwrapped_h)
            new_w = int(unwrapped_w * scale)
            new_h = int(unwrapped_h * scale)
            unwrapped_resized = cv2.resize(unwrapped, (new_w, new_h))
            
            x_offset = spacing + 3 * (cell_w + spacing)
            y_center = y_offset + label_h + cell_h // 2
            x_center = x_offset + cell_w // 2
            y_start = y_center - new_h // 2
            x_start = x_center - new_w // 2
            canvas[y_start:y_start + new_h, x_start:x_start + new_w] = unwrapped_resized
            cv2.putText(canvas, step_labels[3], (x_offset, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Step 5: Classification result
            pred = result['prediction']
            prob = result['probability']
            color = (0, 255, 0) if pred == 'OK' else (0, 0, 255)
            
            text_img = np.ones((cell_h, cell_w, 3), dtype=np.uint8) * 255
            text = f"{pred}"
            text2 = f"Prob: {prob:.3f}"
            text3 = f"Conf: {result.get('confidence', 0):.3f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            (text_w2, text_h2), _ = cv2.getTextSize(text2, font, 0.7, 1)
            
            x_text = (cell_w - text_w) // 2
            y_text = cell_h // 2 - 20
            x_text2 = (cell_w - text_w2) // 2
            y_text2 = cell_h // 2 + 20
            
            cv2.putText(text_img, text, (x_text, y_text), font, font_scale, color, thickness)
            cv2.putText(text_img, text2, (x_text2, y_text2), font, 0.7, (0, 0, 0), 1)
            cv2.putText(text_img, text3, (x_text2, y_text2 + 25), font, 0.6, (0, 0, 0), 1)
            
            x_offset = spacing + 4 * (cell_w + spacing)
            canvas[y_offset + label_h:y_offset + label_h + cell_h, x_offset:x_offset + cell_w] = text_img
            cv2.putText(canvas, step_labels[4], (x_offset, y_offset + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(str(output_path), canvas)
        print(f"✓ Saved visualization grid to: {output_path}")


# ===== Command-line interface =====

if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description='Seal Strip Inference Pipeline')
    parser.add_argument('image', type=str, nargs='*', help='Path(s) to input image(s) or directory')
    parser.add_argument('--models_dir', type=str, default='./models',
                       help='Directory containing model files')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                       help='Save unwrapped strip (single image) or visualization grid (multiple images)')
    parser.add_argument('--save_intermediates', action='store_true',
                       help='Save intermediate results')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Process all images in directory')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Randomly select N images from directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for image selection')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SealStripInference(
        models_dir=args.models_dir,
        device=args.device
    )
    
    # Collect image paths
    image_paths = []
    
    if args.image_dir:
        # Process directory
        image_dir = Path(args.image_dir)
        for img_path in image_dir.glob('*.jpg'):
            image_paths.append(str(img_path))
        for img_path in image_dir.glob('*.png'):
            image_paths.append(str(img_path))
        
        if args.num_images and len(image_paths) > args.num_images:
            random.seed(args.seed)
            image_paths = random.sample(image_paths, args.num_images)
    
    elif args.image:
        # Process provided image(s)
        for img in args.image:
            if Path(img).is_dir():
                for img_path in Path(img).glob('*.jpg'):
                    image_paths.append(str(img_path))
                for img_path in Path(img).glob('*.png'):
                    image_paths.append(str(img_path))
            else:
                image_paths.append(img)
    
    if not image_paths:
        print("Error: No images found to process")
        sys.exit(1)
    
    print(f"Processing {len(image_paths)} image(s)...")
    
    # Process images
    if len(image_paths) == 1:
        # Single image processing
        result = pipeline.process_image(image_paths[0], return_intermediates=args.save_intermediates or args.output is not None)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Results for: {image_paths[0]}")
        print(f"{'='*60}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Threshold: {result['threshold']:.3f}")
        print(f"OD BBox: {result['od_bbox']}")
        
        # Save outputs if requested
        if args.output:
            if 'unwrapped_strip' in result:
                cv2.imwrite(args.output, result['unwrapped_strip'])
                print(f"\n✓ Saved unwrapped strip to: {args.output}")
            else:
                print("\n⚠ Warning: unwrapped_strip not available")
        
        if args.save_intermediates:
            output_dir = Path(image_paths[0]).parent / "intermediates"
            output_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(output_dir / "od_crop.jpg"), result['od_crop'])
            cv2.imwrite(str(output_dir / "unwrapped_strip.jpg"), result['unwrapped_strip'])
            print(f"\n✓ Saved intermediate results to: {output_dir}")
    
    else:
        # Batch processing with visualization grid
        results = pipeline.process_batch(image_paths, output_path=args.output)
        
        print(f"\n{'='*60}")
        print(f"Processed {len(results)} images")
        print(f"{'='*60}")
        
        ok_count = sum(1 for r in results if r['prediction'] == 'OK')
        short_count = len(results) - ok_count
        print(f"OK: {ok_count}/{len(results)}")
        print(f"Short: {short_count}/{len(results)}")
        
        if args.output:
            print(f"\n✓ Visualization grid saved to: {args.output}")


