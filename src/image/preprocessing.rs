use crate::Result;
use ndarray::Array3;

pub struct ImagePreprocessor;

impl ImagePreprocessor {
    /// OCR通用预处理流水线
    pub fn preprocess_for_ocr(image: Array3<f32>) -> Result<Array3<f32>> {
        let mut processed = image;
        
        // 1. 去噪
        processed = Self::denoise(processed)?;
        
        // 2. 对比度增强
        processed = Self::enhance_contrast(processed)?;
        
        // 3. 二值化（可选，根据需要）
        // processed = Self::binarize(processed)?;
        
        Ok(processed)
    }

    /// 图像去噪（简化版高斯滤波）
    fn denoise(image: Array3<f32>) -> Result<Array3<f32>> {
        // 简化的去噪处理
        // 在实际应用中应该使用更复杂的去噪算法
        let (height, width, channels) = image.dim();
        let mut denoised = image.clone();
        
        // 简单的3x3均值滤波
        for c in 0..channels {
            for h in 1..height-1 {
                for w in 1..width-1 {
                    let mut sum = 0.0;
                    let mut count = 0;
                    
                    for dh in -1i32..=1 {
                        for dw in -1i32..=1 {
                            let nh = (h as i32 + dh) as usize;
                            let nw = (w as i32 + dw) as usize;
                            sum += image[[nh, nw, c]];
                            count += 1;
                        }
                    }
                    
                    denoised[[h, w, c]] = sum / count as f32;
                }
            }
        }
        
        Ok(denoised)
    }

    /// 对比度增强
    fn enhance_contrast(mut image: Array3<f32>) -> Result<Array3<f32>> {
        let (height, width, channels) = image.dim();
        
        for c in 0..channels {
            // 计算每个通道的最小和最大值
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            
            for h in 0..height {
                for w in 0..width {
                    let val = image[[h, w, c]];
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
            
            // 线性拉伸
            let range = max_val - min_val;
            if range > 0.01 { // 避免除零
                for h in 0..height {
                    for w in 0..width {
                        let val = image[[h, w, c]];
                        image[[h, w, c]] = ((val - min_val) / range * 255.0).clamp(0.0, 255.0);
                    }
                }
            }
        }
        
        Ok(image)
    }

    /// 二值化处理（Otsu算法）
    pub fn binarize(image: Array3<f32>) -> Result<Array3<f32>> {
        let (height, width, _) = image.dim();
        
        // 转换为灰度图
        let mut gray = Array3::<f32>::zeros((height, width, 1));
        for h in 0..height {
            for w in 0..width {
                let gray_val = (image[[h, w, 0]] * 0.299 + 
                              image[[h, w, 1]] * 0.587 + 
                              image[[h, w, 2]] * 0.114);
                gray[[h, w, 0]] = gray_val;
            }
        }
        
        // 简化的Otsu阈值算法
        let threshold = Self::otsu_threshold(&gray)?;
        
        // 应用阈值
        let mut binary = Array3::<f32>::zeros((height, width, 3));
        for h in 0..height {
            for w in 0..width {
                let val = if gray[[h, w, 0]] > threshold { 255.0 } else { 0.0 };
                binary[[h, w, 0]] = val;
                binary[[h, w, 1]] = val;
                binary[[h, w, 2]] = val;
            }
        }
        
        Ok(binary)
    }

    /// 简化的Otsu阈值计算
    fn otsu_threshold(gray: &Array3<f32>) -> Result<f32> {
        let (height, width, _) = gray.dim();
        let mut histogram = vec![0; 256];
        
        // 计算直方图
        for h in 0..height {
            for w in 0..width {
                let intensity = (gray[[h, w, 0]].clamp(0.0, 255.0) as usize).min(255);
                histogram[intensity] += 1;
            }
        }
        
        let total_pixels = (height * width) as f32;
        let mut best_threshold = 128.0;
        let mut max_variance = 0.0;
        
        // 寻找最佳阈值
        for t in 0..256 {
            let (w1, w2, sum1, sum2) = Self::calculate_class_stats(&histogram, t, total_pixels);
            
            if w1 > 0.0 && w2 > 0.0 {
                let mean1 = sum1 / w1;
                let mean2 = sum2 / w2;
                let between_class_variance = w1 * w2 * (mean1 - mean2).powi(2);
                
                if between_class_variance > max_variance {
                    max_variance = between_class_variance;
                    best_threshold = t as f32;
                }
            }
        }
        
        Ok(best_threshold)
    }

    fn calculate_class_stats(histogram: &[i32], threshold: usize, total: f32) -> (f32, f32, f32, f32) {
        let mut w1 = 0.0;
        let mut sum1 = 0.0;
        
        for i in 0..=threshold {
            w1 += histogram[i] as f32;
            sum1 += (i as f32) * (histogram[i] as f32);
        }
        
        let w2 = total - w1;
        let mut sum2 = 0.0;
        
        for i in (threshold + 1)..256 {
            sum2 += (i as f32) * (histogram[i] as f32);
        }
        
        (w1 / total, w2 / total, sum1, sum2)
    }

    /// 图像旋转（基于角度分类结果）
    pub fn rotate_image(image: Array3<f32>, angle_degrees: f32) -> Result<Array3<f32>> {
        if (angle_degrees.abs() < 0.1) {
            return Ok(image); // 不需要旋转
        }
        
        // 简化的旋转实现（只支持90度倍数）
        let normalized_angle = ((angle_degrees / 90.0).round() as i32 * 90) % 360;
        
        match normalized_angle {
            0 => Ok(image),
            90 | -270 => Ok(Self::rotate_90(&image)?),
            180 | -180 => Ok(Self::rotate_180(&image)?),
            270 | -90 => Ok(Self::rotate_270(&image)?),
            _ => Ok(image), // 不支持任意角度旋转，返回原图
        }
    }

    fn rotate_90(image: &Array3<f32>) -> Result<Array3<f32>> {
        let (height, width, channels) = image.dim();
        let mut rotated = Array3::<f32>::zeros((width, height, channels));
        
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    rotated[[width - 1 - w, h, c]] = image[[h, w, c]];
                }
            }
        }
        
        Ok(rotated)
    }

    fn rotate_180(image: &Array3<f32>) -> Result<Array3<f32>> {
        let (height, width, channels) = image.dim();
        let mut rotated = Array3::<f32>::zeros((height, width, channels));
        
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    rotated[[height - 1 - h, width - 1 - w, c]] = image[[h, w, c]];
                }
            }
        }
        
        Ok(rotated)
    }

    fn rotate_270(image: &Array3<f32>) -> Result<Array3<f32>> {
        let (height, width, channels) = image.dim();
        let mut rotated = Array3::<f32>::zeros((width, height, channels));
        
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    rotated[[w, height - 1 - h, c]] = image[[h, w, c]];
                }
            }
        }
        
        Ok(rotated)
    }
}