# Debug script for WGG (geometric verification)
import cv2
import numpy as np
from check_geometric_consistency import extract_sift_features, check_geometric_consistency

# Image paths
image1_path = r"f:\E\SourceAntiDupl\TestCropHard\emma stone katharine mcphee kat dennings anna faris rumer willis the house bunny.jpg"
image2_path = r"f:\E\SourceAntiDupl\TestCropHard\emma stone.jpg"

print(f"Image 1: {image1_path}")
print(f"Image 2: {image2_path}")
print()

# Load images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

if img1 is None:
    print(f"ERROR: Could not load image 1: {image1_path}")
    exit(1)
    
if img2 is None:
    print(f"ERROR: Could not load image 2: {image2_path}")
    exit(1)

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")
print()

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print(f"Image 1 size: {gray1.shape[1]}x{gray1.shape[0]}")
print(f"Image 2 size: {gray2.shape[1]}x{gray2.shape[0]}")
print()

# Extract SIFT features
kp1, des1 = extract_sift_features(gray1)
kp2, des2 = extract_sift_features(gray2)

print(f"Keypoints in image 1: {len(kp1)}")
print(f"Keypoints in image 2: {len(kp2)}")
print(f"Descriptors in image 1: {des1.shape if des1 is not None else None}")
print(f"Descriptors in image 2: {des2.shape if des2 is not None else None}")
print()

if des1 is None or des2 is None:
    print("ERROR: No descriptors found in one of the images!")
    exit(1)

# Find matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

print(f"Total matches (k=2): {len(matches)}")
print()

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"Good matches after ratio test: {len(good_matches)}")
print()

if len(good_matches) < 10:
    print("FAIL: Too few good matches (<10)")
    exit(1)

# Analyze geometric consistency
is_valid, avg_angle, avg_scale, a_votes, s_votes = check_geometric_consistency(kp1, des1, kp2, des2, 0.3)

print(f"WGC Result: {is_valid}")
print(f"Average angle: {avg_angle:.2f} degrees ({a_votes} votes)")
print(f"Average scale: {avg_scale:.4f} ({s_votes} votes)")
print()

# Show some match details
print("First 20 matches:")
for i, m in enumerate(good_matches[:20]):
    pt1 = kp1[m.queryIdx]
    pt2 = kp2[m.trainIdx]
    angle_diff = (pt1.angle - pt2.angle) % 360
    scale_ratio = pt1.size / pt2.size if pt2.size > 0 else 0
    print(f"  Match {i+1}: img1_angle={pt1.angle:.1f}, img2_angle={pt2.angle:.1f}, diff={angle_diff:.1f}, ratio={scale_ratio:.3f}")

print()

# Analyze distribution
angles = []
scales = []
for m in good_matches:
    pt1 = kp1[m.queryIdx]
    pt2 = kp2[m.trainIdx]
    angle_diff = (pt1.angle - pt2.angle) % 360
    angles.append(angle_diff)
    if pt2.size > 0:
        scales.append(pt1.size / pt2.size)

print(f"Angle statistics:")
print(f"  Min: {min(angles):.2f}")
print(f"  Max: {max(angles):.2f}")
print(f"  Mean: {np.mean(angles):.2f}")
print(f"  Std: {np.std(angles):.2f}")
print()

print(f"Scale ratio statistics:")
print(f"  Min: {min(scales):.4f}")
print(f"  Max: {max(scales):.4f}")
print(f"  Mean: {np.mean(scales):.4f}")
print(f"  Std: {np.std(scales):.4f}")
print()

# Histogram analysis
hist_angles, bins_angles = np.histogram(angles, bins=24, range=(0, 360))
max_angle_votes = np.max(hist_angles)
print(f"Most common angle bin: {max_angle_votes} votes (need >{len(good_matches)*0.5:.1f})")

log_scales = np.log2(scales)
hist_scales, bins_scales = np.histogram(log_scales, bins=20, range=(-3, 3))
max_scale_votes = np.max(hist_scales)
print(f"Most common scale bin: {max_scale_votes} votes (need >{len(good_matches)*0.5:.1f})")