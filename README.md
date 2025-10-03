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

### Summary

| Method              | Best for            | Performance | Notes                            |
|---------------------|---------------------|-------------|----------------------------------|
| U-Net + OpenCV      | General segmentation| Low         | Sensitive to noise and lighting  |
| OTSU Thresholding   | Thin/hairline cracks| Medium-High | Requires good contrast/preprocessing |
| Sobel Filtering     | Basic edge detection| Low-Medium  | Not reliable in complex images   |

---

### Next Steps

- Explore hybrid approaches that combine OTSU with other edge-based techniques.
- Improve preprocessing to normalize lighting conditions.
- Test newer segmentation models for better generalization.
