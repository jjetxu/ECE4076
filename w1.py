import numpy as np

### 1. Grayscale Image
greyscaleImg = np.array([117, 97, 83, 79, 71, 113, 90, 80, 76, 68, 93, 86, 78, 65, 60, 89, 84, 74, 64, 59, 83, 79, 68, 62, 58])

greyscaleIntensity = greyscaleImg / 255

### 2. Color Image
red = np.array([117, 161, 138, 139, 117, 161, 152, 125, 134, 106, 146, 142, 123, 118, 101, 133, 140, 119, 110, 96, 138, 122, 112, 101, 94])
blue = np.array([97, 86, 67, 72, 59, 86, 80, 57, 72, 51, 75, 75, 61, 60, 48, 67, 77, 60, 56, 47, 76, 62, 55, 48, 47])
green = np.array([24, 21, 13, 27, 22, 18, 21, 8, 31, 20, 11, 20, 14, 22, 17, 6, 24, 16, 20, 17, 19, 12, 12, 14, 19])

rgbIntensity = (0.21*red + 0.72*green + 0.07*blue) / 255


### 3. Color Space Conversion
# normalize rgb values
red_ = red/255
blue_ = blue/255
green_ = green/255

# compute Cmax, Cmin, delta
cmax  = np.maximum(np.maximum(red_, green_), blue_)
cmin  = np.minimum(np.minimum(red_, green_), blue_)
delta = cmax - cmin

# hue calculation
hue = np.zeros_like(delta)
mask1 = (delta != 0) & (cmax == red_)
mask2 = (delta != 0) & (cmax == green_)
mask3 = (delta != 0) & (cmax == blue_)

hue[mask1] = 60 * (((green_[mask1] - blue_[mask1]) / delta[mask1]) % 6)
hue[mask2] = 60 * (((blue_[mask2] - red_[mask2])   / delta[mask2]) + 2)
hue[mask3] = 60 * (((red_[mask3] - green_[mask3])   / delta[mask3]) + 4)

# saturation computation
saturation = np.where(cmax == 0, 0, delta / cmax)

# value computation
value = cmax


### 4. Optical Illusion
# Explain why the human visual system would be influenced by the context within which an object is viewed.
'''
The human visual system relies on eyes as the sensors, but all data is processed by the human brain. 
Our brain recognizes patterns and can naturally guess the contexts. 
Given figure 4, we see the checkerboard pattern and might determine tile B to be the same color as the other light greys.
We recognize the shading is from the green cylinder, making tile B slightly darker, but in a real world scenario we believe tile B to be lighter than A.
The brain evolved to perceive reflectance (what colour something truly is) rather than luminance (how much light reaches our eyes).
'''

### 5. Surface Reflectance
# R = L - 2(L ⋅ N)N
''' 
R is L flipped across N, so take L and subtract the perpendicular to the normal component twice.
'''
