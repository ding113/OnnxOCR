use crate::Result;
use ndarray::Array3;

/// 图像变换工具集
pub struct ImageTransforms;

impl ImageTransforms {
    /// 图像缩放（保持宽高比）
    pub fn resize_with_aspect_ratio(
        image: &Array3<f32>,
        target_width: usize,
        target_height: usize,
    ) -> Result<Array3<f32>> {
        let (orig_h, orig_w, channels) = image.dim();
        
        // 计算缩放比例
        let scale_w = target_width as f32 / orig_w as f32;
        let scale_h = target_height as f32 / orig_h as f32;
        let scale = scale_w.min(scale_h);
        
        let new_w = (orig_w as f32 * scale) as usize;
        let new_h = (orig_h as f32 * scale) as usize;
        
        // 创建目标图像，用白色填充
        let mut resized = Array3::<f32>::from_elem((target_height, target_width, channels), 255.0);
        
        // 计算padding偏移
        let offset_x = (target_width - new_w) / 2;
        let offset_y = (target_height - new_h) / 2;
        
        // 双线性插值缩放
        for h in 0..new_h {
            for w in 0..new_w {
                let src_h = h as f32 / scale;
                let src_w = w as f32 / scale;
                
                let h1 = src_h.floor() as usize;
                let h2 = (h1 + 1).min(orig_h - 1);
                let w1 = src_w.floor() as usize;
                let w2 = (w1 + 1).min(orig_w - 1);
                
                let dh = src_h - h1 as f32;
                let dw = src_w - w1 as f32;
                
                for c in 0..channels {
                    let v11 = image[[h1, w1, c]];
                    let v12 = image[[h1, w2, c]];
                    let v21 = image[[h2, w1, c]];
                    let v22 = image[[h2, w2, c]];
                    
                    let interpolated = v11 * (1.0 - dh) * (1.0 - dw)
                        + v12 * (1.0 - dh) * dw
                        + v21 * dh * (1.0 - dw)
                        + v22 * dh * dw;
                    
                    resized[[h + offset_y, w + offset_x, c]] = interpolated;
                }
            }
        }
        
        Ok(resized)
    }
    
    /// 图像裁剪（从四边形区域）
    pub fn crop_polygon(
        image: &Array3<f32>,
        points: &[[f32; 2]; 4],
    ) -> Result<Array3<f32>> {
        // 计算边界框
        let min_x = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min) as usize;
        let max_x = points.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max) as usize;
        let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min) as usize;
        let max_y = points.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max) as usize;
        
        let (orig_h, orig_w, channels) = image.dim();
        
        // 边界检查
        let min_x = min_x.min(orig_w - 1);
        let max_x = max_x.min(orig_w - 1);
        let min_y = min_y.min(orig_h - 1);
        let max_y = max_y.min(orig_h - 1);
        
        let crop_width = max_x - min_x + 1;
        let crop_height = max_y - min_y + 1;
        
        if crop_width == 0 || crop_height == 0 {
            return Err(crate::utils::error::OcrError::InvalidInput(
                "Invalid crop region".to_string()
            ));
        }
        
        let mut cropped = Array3::<f32>::zeros((crop_height, crop_width, channels));
        
        for h in 0..crop_height {
            for w in 0..crop_width {
                let src_h = min_y + h;
                let src_w = min_x + w;
                
                for c in 0..channels {
                    cropped[[h, w, c]] = image[[src_h, src_w, c]];
                }
            }
        }
        
        Ok(cropped)
    }
    
    /// 透视变换校正
    pub fn perspective_transform(
        image: &Array3<f32>,
        src_points: &[[f32; 2]; 4],
        dst_width: usize,
        dst_height: usize,
    ) -> Result<Array3<f32>> {
        // 目标点（矩形四个角点）
        let dst_points = [
            [0.0, 0.0],
            [dst_width as f32, 0.0],
            [dst_width as f32, dst_height as f32],
            [0.0, dst_height as f32],
        ];
        
        // 计算透视变换矩阵
        let transform_matrix = Self::compute_perspective_matrix(src_points, &dst_points)?;
        
        // 应用变换
        Self::apply_perspective_transform(image, &transform_matrix, dst_width, dst_height)
    }
    
    /// 计算透视变换矩阵（简化版本）
    fn compute_perspective_matrix(
        src: &[[f32; 2]; 4],
        dst: &[[f32; 2]; 4],
    ) -> Result<[[f32; 3]; 3]> {
        // 这是一个简化的透视变换矩阵计算
        // 在生产环境中应该使用更精确的算法
        
        // 计算平移和缩放
        let src_center_x = (src[0][0] + src[1][0] + src[2][0] + src[3][0]) / 4.0;
        let src_center_y = (src[0][1] + src[1][1] + src[2][1] + src[3][1]) / 4.0;
        let dst_center_x = (dst[0][0] + dst[1][0] + dst[2][0] + dst[3][0]) / 4.0;
        let dst_center_y = (dst[0][1] + dst[1][1] + dst[2][1] + dst[3][1]) / 4.0;
        
        let src_width = (src[1][0] - src[0][0] + src[2][0] - src[3][0]) / 2.0;
        let src_height = (src[3][1] - src[0][1] + src[2][1] - src[1][1]) / 2.0;
        let dst_width = (dst[1][0] - dst[0][0] + dst[2][0] - dst[3][0]) / 2.0;
        let dst_height = (dst[3][1] - dst[0][1] + dst[2][1] - dst[1][1]) / 2.0;
        
        let scale_x = if src_width.abs() > 0.001 { dst_width / src_width } else { 1.0 };
        let scale_y = if src_height.abs() > 0.001 { dst_height / src_height } else { 1.0 };
        
        let tx = dst_center_x - src_center_x * scale_x;
        let ty = dst_center_y - src_center_y * scale_y;
        
        Ok([
            [scale_x, 0.0, tx],
            [0.0, scale_y, ty],
            [0.0, 0.0, 1.0],
        ])
    }
    
    /// 应用透视变换
    fn apply_perspective_transform(
        image: &Array3<f32>,
        matrix: &[[f32; 3]; 3],
        dst_width: usize,
        dst_height: usize,
    ) -> Result<Array3<f32>> {
        let (orig_h, orig_w, channels) = image.dim();
        let mut transformed = Array3::<f32>::from_elem((dst_height, dst_width, channels), 255.0);
        
        for dst_y in 0..dst_height {
            for dst_x in 0..dst_width {
                // 反向变换：从目标坐标映射到源坐标
                let src_x = matrix[0][0] * dst_x as f32 + matrix[0][1] * dst_y as f32 + matrix[0][2];
                let src_y = matrix[1][0] * dst_x as f32 + matrix[1][1] * dst_y as f32 + matrix[1][2];
                
                let src_x_int = src_x as i32;
                let src_y_int = src_y as i32;
                
                if src_x_int >= 0 && src_x_int < orig_w as i32 && 
                   src_y_int >= 0 && src_y_int < orig_h as i32 {
                    for c in 0..channels {
                        transformed[[dst_y, dst_x, c]] = image[[src_y_int as usize, src_x_int as usize, c]];
                    }
                }
            }
        }
        
        Ok(transformed)
    }
    
    /// 图像亮度调整
    pub fn adjust_brightness(image: &Array3<f32>, brightness: f32) -> Result<Array3<f32>> {
        let mut adjusted = image.clone();
        let (height, width, channels) = image.dim();
        
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    adjusted[[h, w, c]] = (image[[h, w, c]] + brightness).clamp(0.0, 255.0);
                }
            }
        }
        
        Ok(adjusted)
    }
    
    /// 图像对比度调整
    pub fn adjust_contrast(image: &Array3<f32>, contrast: f32) -> Result<Array3<f32>> {
        let mut adjusted = image.clone();
        let (height, width, channels) = image.dim();
        
        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    adjusted[[h, w, c]] = ((image[[h, w, c]] - 128.0) * contrast + 128.0).clamp(0.0, 255.0);
                }
            }
        }
        
        Ok(adjusted)
    }
}