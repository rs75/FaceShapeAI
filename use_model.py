"""
Script to use the trained Face Shape Classification Model to predict the shape of a face in an image.
Usage: python use_model.py <path_to_image>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse

# Paths to model files (assumed to be in the same directory as this script)
PROJECT_DIR = Path(__file__).parent
MODEL_FILE = PROJECT_DIR / 'face_shape_model.pkl'
LABEL_ENCODER_FILE = PROJECT_DIR / 'label_encoder.pkl'


Keypoint = Dict[str, float]
NormalizedLandmark = Tuple[float, float, float]


def normalize_landmarks(keypoints: Sequence[Keypoint], width: int, height: int) -> List[NormalizedLandmark]:
    """
    Normalize keypoints to be centered, roll-corrected, and scaled.
    Retains 3D coordinates (Z) but aligns to the 2D plane based on eyes.
    
    Returns list of tuples: [(x, y, z), ...]
    
    (Copied from create.py to ensure consistent preprocessing)
    """
    if not keypoints:
        return []

    # Convert to numpy array (N, 3)
    landmarks = np.array([[kp["x"], kp["y"], kp["z"]] for kp in keypoints])
    
    # Denormalize x, y, z to pixel/aspect-correct coordinates
    # MediaPipe Z is roughly same scale as X (relative to image width)
    landmarks[:, 0] *= width
    landmarks[:, 1] *= height
    landmarks[:, 2] *= width 

    # Indices for irises (refine_landmarks=True gives 478 points)
    # 468: Left Iris Center (Subject's Left, Image Right)
    # 473: Right Iris Center (Subject's Right, Image Left)
    left_iris_idx = 468
    right_iris_idx = 473

    if len(landmarks) > right_iris_idx:
        left_iris = landmarks[left_iris_idx]
        right_iris = landmarks[right_iris_idx]
    else:
        # Fallback to eye corners if iris landmarks missing
        p1 = landmarks[33]  # Left eye outer
        p2 = landmarks[133] # Left eye inner
        left_iris = (p1 + p2) / 2
        p3 = landmarks[362] # Right eye inner
        p4 = landmarks[263] # Right eye outer
        right_iris = (p3 + p4) / 2

    # 1. Centering: Move midpoint of eyes to origin
    eye_center = (left_iris + right_iris) / 2.0
    landmarks -= eye_center

    # 2. Rotation (Roll Correction)
    delta = left_iris - right_iris
    dX, dY = delta[0], delta[1]
    
    # Calculate angle of this vector relative to horizontal
    angle = np.arctan2(dY, dX)
    
    # Rotate by -angle to align with X-axis
    c, s = np.cos(-angle), np.sin(-angle)
    
    # Rotation matrix around Z axis
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    landmarks = landmarks.dot(R.T)

    # 3. Scaling: Scale such that inter-ocular distance is 1.0
    dist = np.sqrt(dX**2 + dY**2)
    if dist > 0:
        scale = 1.0 / dist
        landmarks *= scale

    # Convert to list of tuples
    return [(round(float(l[0]), 5), round(float(l[1]), 5), round(float(l[2]), 5)) 
            for l in landmarks]


def create_face_mesh(image_path: Union[str, Path]) -> Tuple[Optional[List[Keypoint]], Optional[np.ndarray]]:
    """
    Process image to get face mesh data using MediaPipe
    Returns: keypoints, img_bgr or None if failed
    
    (Copied from create.py to ensure consistent preprocessing)
    """
    max_width_or_height = 512

    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize face mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        
        # Read image from file
        img_bgr = cv2.imread(str(image_path))
        
        if img_bgr is None:
            print(f"Error: Could not read image: {image_path}")
            return None, None

        # Downscale large images to speed up inference (keep aspect ratio)
        h, w = img_bgr.shape[:2]
        longest = max(h, w)
        if longest > max_width_or_height:
            scale = max_width_or_height / float(longest)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            print(f"Error: No face detected in: {image_path}")
            return None, None

        keypoints = []
        for landmark in results.multi_face_landmarks[0].landmark:
            keypoints.append({
                "x": round(landmark.x, 5),
                "y": round(landmark.y, 5),
                "z": round(landmark.z, 5)
            })
        return keypoints, img_bgr


def load_model_resources() -> Tuple[Any, Any]:
    """Load the trained model and label encoder."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Please run create_model.py first.")
    
    if not LABEL_ENCODER_FILE.exists():
        raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_FILE}. Please run create_model.py first.")

    print(f"Loading model from {MODEL_FILE}...")
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Loading label encoder from {LABEL_ENCODER_FILE}...")
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
        
    return model, label_encoder


def predict_face_shape(image_path: Union[str, Path]) -> Optional[str]:
    """
    Main function to predict face shape for a given image.
    """
    # 1. Load Model
    try:
        model, label_encoder = load_model_resources()
    except Exception as e:
        print(f"Failed to load model resources: {e}")
        return None

    # 2. Process Image (Extract Landmarks)
    print(f"Processing image: {image_path}")
    keypoints, img_bgr = create_face_mesh(image_path)
    
    if keypoints is None:
        print("Could not extract landmarks. Exiting.")
        return None

    # 3. Normalize Landmarks
    h, w = img_bgr.shape[:2]
    normalized_kpts = normalize_landmarks(keypoints, w, h)
    
    # 4. Prepare Features (Flatten and drop Z)
    # The model expects a flattened array of [x1, y1, x2, y2, ...]
    flattened_features: List[float] = []
    for kp in normalized_kpts:
        flattened_features.extend([kp[0], kp[1]])  # x, y only
    
    # Reshape for sklearn (1 sample, N features)
    features_array = np.array([flattened_features])
    
    # 5. Predict
    print("Running prediction...")
    # Get probabilities
    probas = model.predict_proba(features_array)[0]
    # Get prediction
    prediction_idx = model.predict(features_array)[0]
    predicted_label = label_encoder.inverse_transform([prediction_idx])[0]
    
    # 6. Show Results
    print("\n" + "="*30)
    print(f"PREDICTED FACE SHAPE: {predicted_label.upper()}")
    print("="*30)
    
    print("\nConfidence Scores:")
    # Sort probabilities
    class_indices = np.argsort(probas)[::-1]
    for i in class_indices:
        class_name = label_encoder.classes_[i]
        score = probas[i]
        print(f"  {class_name}: {score:.4f}")

    return predicted_label


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Note: Default behavior remains to run against `sample_image.jpg` when no args are provided.
    """
    parser = argparse.ArgumentParser(description="Predict face shape from an image using a trained sklearn model.")
    parser.add_argument(
        "image",
        nargs="?",
        default="sample_image.jpg",
        help="Path to the input image (default: sample_image.jpg).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    predict_face_shape(args.image)
