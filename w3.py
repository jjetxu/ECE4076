### 1 
'''
(1) The simplest way is to compare each pixel before and after using the sum of squared differences
(SSD). When shifting the window W by small motion (u, v) cross the grayscale image I.



'''

import numpy as np

# Full 7x7 image from Figure 3
image = np.array([
    [85, 78, 75, 72, 71, 70, 72],
    [83, 80, 76, 71, 72, 69, 67],
    [82, 79, 54, 52, 53, 61, 61],
    [77, 68, 51, 50, 51, 49, 51],
    [75, 62, 53, 52, 49, 48, 49],
    [74, 73, 65, 63, 55, 48, 44],
    [74, 72, 71, 68, 55, 46, 40],
], dtype=np.float64)

# The central 3x3 patch is at rows 2-4, cols 2-4
# With 1-pixel padding, we extract the 5x5 region around it
patch = image[1:6, 1:6]  # rows 1-5, cols 1-5
print("5x5 padded region:\n", patch)

# Compute gradients using h_x = [-1, 0, 1] and h_y = [-1, 0, 1]^T
# For each pixel (i,j) in the central 3x3 (indices 1-3 within the 5x5):
rows, cols = 3, 3
Ix = np.zeros((rows, cols))
Iy = np.zeros((rows, cols))

for i in range(rows):
    for j in range(cols):
        # i+1, j+1 are the coordinates in the 5x5 patch
        pi, pj = i + 1, j + 1
        Ix[i, j] = patch[pi, pj + 1] - patch[pi, pj - 1]  # h_x = [-1, 0, 1]
        Iy[i, j] = patch[pi + 1, pj] - patch[pi - 1, pj]  # h_y = [-1, 0, 1]^T

print("\nIx:\n", Ix)
print("\nIy:\n", Iy)

# Structure tensor M = sum over window of [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]]
m11 = np.sum(Ix ** 2)
m12 = np.sum(Ix * Iy)
m22 = np.sum(Iy ** 2)

M = np.array([[m11, m12],
              [m12, m22]])
print("\nStructure Tensor M:\n", M)

# Eigenvalues
eigenvalues = np.linalg.eigvalsh(M)
print("\nEigenvalues:", np.sort(eigenvalues)[::-1])

### 2
'''
Scale-invariant feature transform (SIFT). The SIFT detector is designed to identify and describe
local features in images. The key aspects of SIFT include the detection of keypoints and computation
of their descriptors which are invariant to scale, and rotation, and partially invariant to changes in
illumination and viewpoint.
Please elaborate on the fundamental concept and the procedural steps involved in the SIFT algorithm.

Answer:
SIFT finds keypoints that are stable across changes in scale, rotation and lighting, then describes them in a way that allows reliable matching between images.

The steps are:
    1. Extract scale and rotation normalised image
    2. cover with a 16x16 grid
    3. Calculate gradient in each cell
    4. Make a histogram with 8 bins for the directions of each cell
'''

### 3
