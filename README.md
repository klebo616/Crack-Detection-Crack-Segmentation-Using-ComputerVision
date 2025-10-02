# Crack-Detection-Crack-Segmentation-Using-ComputerVision

This GitHub repository has the scope to follow the steps of a successful crack detection with multiple, mainly un-pretrained, models.

**Focusing on now → UNet architecture (DoubleConvolution, UpSampling, DownSampling):**
The trained UNet model from scratch includes generated binary Masks (ground truth). Segmentation model used was OpenCV  
Problems:
- Thin *haircracks* are not being segmented correctly  
- Very sensitive to noise

