import numpy as np
import cv2

### 1. Filters
## (1)
# ── Original 5×5 image ──────────────────────────────────────────────
image = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0, 10, 10, 10,  0],
    [ 0, 10, 10, 10,  0],
    [ 0, 10,  0, 10,  0],
    [10,  0,  0,  0,  0]
], dtype=np.float64)

# ── Kernels ──────────────────────────────────────────────────────────
h0 = np.ones((3, 3)) / 9          # Mean
h1 = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16  # Gaussian

# Mean filter — use box filter (equivalent to mean)
out_mean = cv2.boxFilter(image, ddepth=-1, ksize=(3,3), normalize=True)

# Gaussian filter
h1 = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float64) / 16
out_gauss = cv2.filter2D(image, -1, h1)

# Median filter — cv2.medianBlur requires uint8
out_median = cv2.medianBlur(image.astype(np.uint8), ksize=3)

# Round to uint8
out_mean   = np.round(out_mean).astype(np.uint8)
out_gauss  = np.round(out_gauss).astype(np.uint8)

print("Original image:")
print(image.astype(np.uint8))
print("\nMean filter output:")
print(out_mean)
print("\nGaussian filter output:")
print(out_gauss)
print("\nMedian filter output:")
print(out_median)

## (2) What's the size of the resulting image after convolution? What happens if a larger kernel size is used? How can we maintain the size?
'''
The input image is 5x5 and the filter is 3x3. 
Then the output image must be 3x3 since using a 3x3 kernal will lose a border of 1px.
If we used a larger kernel the output image would be smaller.
To maintain output size we can pad the border with pixels.
'''

## (3) Which filter performs best for removing salt-and-pepper noise? Which performs best for thermal noise?
'''
Median filtering performs best when removing salt-and-pepper noise because it completely removes outliers which salt-and-pepper noise introduces.
Gaussian filtering performs best when removing thermal noise because it smoothes out the random variations in pixel by taking weighted averages of surrounding patches.
'''

### 2.Edge Detection
## (1)
# ── Kernels ──────────────────────────────────────────────────────────
hx = np.array([
    [0,    0, 0],
    [-0.5, 0, 0.5],
    [0,    0, 0]
])

hy = np.array([
    [0, -0.5, 0],
    [0,  0,   0],
    [0,  0.5, 0]
])
# cv2.filter2D(image, depth, kernel)
# depth=-1 means output has same dtype as input
horizontal_edges = cv2.filter2D(image, -1, hy)
vertical_edges   = cv2.filter2D(image, -1, hx)

print("Horizontal edges:\n", horizontal_edges)
print("\nVertical edges:\n", vertical_edges)

## (2) How to comput a directional derivative in a specific direction?
# ── Step 1: Apply Sobel in X and Y directions ────────────────────────
Gx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)  # vertical edges
Gy = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)  # horizontal edges

# ── Step 2: Magnitude ────────────────────────────────────────────────
magnitude = np.sqrt(Gx**2 + Gy**2)

# ── Step 3: Orientation (in degrees) ────────────────────────────────
orientation = np.degrees(np.arctan2(Gy, Gx))

# ── Round to hundredth place ─────────────────────────────────────────
magnitude   = np.round(magnitude,   2)
orientation = np.round(orientation, 2)

print("Gx (vertical edges):")
print(Gx)
print("\nGy (horizontal edges):")
print(Gy)
print("\nMagnitude:")
print(magnitude)
print("\nOrientation (degrees):")
print(orientation)

### 3 Corner Matching
# ── Patches ──────────────────────────────────────────────────────────
P11 = np.array([[0,  0,  0],
                [0,  10, 10],
                [0,  10, 10]])

P12 = np.array([[10, 20, 20],
                [10, 20, 20],
                [10, 10, 10]])

P21 = np.array([[10, 10, 10],
                [10, 20, 20],
                [10, 20, 20]])

P22 = np.array([[0,  10, 10],
                [0,  10, 10],
                [0,  0,  0]])

def ssd(patch1, patch2):
    return np.sum((patch1 - patch2) ** 2)

def mean_subtract(patch):
    return patch - np.mean(patch)

# ── Part 1: Calculate all SSDs ───────────────────────────────────────
print("=" * 40)
print("PART 1 — Raw SSD")
print("=" * 40)
print(f"SSD(P11, P21) = {ssd(P11, P21)}")
print(f"SSD(P11, P22) = {ssd(P11, P22)}")
print(f"SSD(P12, P21) = {ssd(P12, P21)}")
print(f"SSD(P12, P22) = {ssd(P12, P22)}")

# ── Part 2: Find best match ──────────────────────────────────────────
print("\n" + "=" * 40)
print("PART 2 — Best Matches")
print("=" * 40)

p11_best = "P21" if ssd(P11, P21) < ssd(P11, P22) else "P22"
p12_best = "P21" if ssd(P12, P21) < ssd(P12, P22) else "P22"
print(f"P11 best match: {p11_best} (lower SSD wins)")
print(f"P12 best match: {p12_best} (lower SSD wins)")

# ── Part 3: Mean-subtracted SSD ──────────────────────────────────────
print("\n" + "=" * 40)
print("PART 3 — Mean-Subtracted SSD")
print("=" * 40)

P11m, P12m = mean_subtract(P11), mean_subtract(P12)
P21m, P22m = mean_subtract(P21), mean_subtract(P22)

print(f"SSD(P11, P21) = {ssd(P11m, P21m):.2f}")
print(f"SSD(P11, P22) = {ssd(P11m, P22m):.2f}")
print(f"SSD(P12, P21) = {ssd(P12m, P21m):.2f}")
print(f"SSD(P12, P22) = {ssd(P12m, P22m):.2f}")

p11_best_m = "P21" if ssd(P11m, P21m) < ssd(P11m, P22m) else "P22"
p12_best_m = "P21" if ssd(P12m, P21m) < ssd(P12m, P22m) else "P22"
print(f"\nP11 best match: {p11_best_m}")
print(f"P12 best match: {p12_best_m}")

# ── Did the matches change? ───────────────────────────────────────────
print("\n" + "=" * 40)
print("Did matches change after mean subtraction?")
print("=" * 40)
print(f"P11: {p11_best} → {p11_best_m} ({'CHANGED' if p11_best != p11_best_m else 'SAME'})")
print(f"P12: {p12_best} → {p12_best_m} ({'CHANGED' if p12_best != p12_best_m else 'SAME'})")