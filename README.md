# Crack-Detection-Crack-Segmentation-Using-ComputerVision

## Crack Segmentation Experiments

This section summarizes the different crack segmentation methods I tested and their effectiveness depending on the type of crack (thick vs. thin).

---

### 1. U-Net with OpenCV Preprocessing

I began by using a standard U-Net architecture for semantic segmentation, combined with basic OpenCV preprocessing steps such as grayscale conversion, Gaussian blur, and edge detection.

**Observations:**
- The model did not generalize well.
- Performance was poor on both small and large cracks, especially under varying lighting conditions.
- Noise and background textures negatively affected the segmentation accuracy.

**Conclusion:**  
This approach lacked robustness and failed to reliably detect cracks of different sizes and contrasts.

---

### 2. OTSU Thresholding

To address the challenge of segmenting very fine, hairline cracks, I used OTSU's method for automatic thresholding based on image histograms.

**Observations:**
- Performed well on high-resolution images containing thin, low-contrast cracks.
- Worked best when the image had a clear foreground-background intensity separation.
- Required proper preprocessing to handle lighting variation.

**Conclusion:**  
OTSU was effective for detecting subtle cracks that other models missed, but it struggled in inconsistent lighting conditions or when cracks were not clearly separated in grayscale intensity.

---

### 3. Sobel Edge Detection

I also tested Sobel filters to extract edges based on image gradients (horizontal and vertical).

**Observations:**
- Performed reasonably well on images with clear crack boundaries.
- Limited performance in noisy or complex backgrounds.
- Often produced fragmented or incomplete crack outlines.

**Conclusion:**  
Useful for initial edge detection but insufficient on its own for accurate segmentation.

---
## 4. Model training using Dynamic Sneak Convolution (DSC)

Dynamic Snake Convolution (DSC) is a technique designed to improve how convolutional neural networks (CNNs) detect and follow object boundaries — like cracks in concrete.

Unlike standard convolution, which uses fixed grid patterns (e.g., 3×3 kernels), DSC allows the sampling points of the convolution to move dynamically. This flexibility helps the network "adapt" to curved or irregular shapes in the image, similar to how a snake moves along a path.

DSC is especially useful in tasks like crack detection, edge segmentation, and irregular object boundaries, where traditional kernels might miss or blur thin or curved details.

https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-024-81119-1/MediaObjects/41598_2024_81119_Fig2_HTML.png?as=webp

### Summary

| Method              | Best for            | Performance | Notes                            |
|---------------------|---------------------|-------------|----------------------------------|
| U-Net + OpenCV      | General segmentation| Low         | Sensitive to noise and lighting  |
| OTSU Thresholding   | Thin/hairline cracks| Medium-High | Requires good contrast/preprocessing |
| Sobel Filtering     | Basic edge detection| Low-Medium  | Not reliable in complex images   |

---

### Next Steps

- Explore other deeplearning models (SAM,...)
- Improve preprocessing to normalize lighting conditions.
- Test newer segmentation models for better generalization.
