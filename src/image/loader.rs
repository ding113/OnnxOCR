use crate::utils::error::OcrError;
use crate::Result;
use axum::body::Bytes;
use base64::Engine;
use image::{DynamicImage, GenericImageView, ImageFormat};
use ndarray::Array3;
use tokio::io::AsyncRead;

pub struct ImageLoader;

impl ImageLoader {
    /// 从base64字符串加载图像
    pub fn from_base64(base64_data: &str) -> Result<DynamicImage> {
        // 检测并移除可能的数据URL前缀 (data:image/xxx;base64,)
        let base64_clean = if base64_data.starts_with("data:") {
            base64_data.split(',').nth(1).unwrap_or(base64_data)
        } else {
            base64_data
        };

        // 解码base64
        let image_bytes = base64::engine::general_purpose::STANDARD
            .decode(base64_clean)
            .map_err(|e| OcrError::Base64(e))?;

        // 检查文件大小
        if image_bytes.len() > 50 * 1024 * 1024 { // 50MB限制
            return Err(OcrError::FileTooLarge(image_bytes.len(), 50 * 1024 * 1024));
        }

        // 解码图像
        let image = image::load_from_memory(&image_bytes)
            .map_err(|e| OcrError::ImageDecode(e))?;

        Ok(image)
    }

    /// 从字节流加载图像
    pub fn from_bytes(bytes: Bytes) -> Result<DynamicImage> {
        // 检查文件大小
        if bytes.len() > 50 * 1024 * 1024 { // 50MB限制
            return Err(OcrError::FileTooLarge(bytes.len(), 50 * 1024 * 1024));
        }

        let image = image::load_from_memory(&bytes)
            .map_err(|e| OcrError::ImageDecode(e))?;

        Ok(image)
    }

    /// 从文件路径加载图像
    pub fn from_path(path: &str) -> Result<DynamicImage> {
        let image = image::open(path)
            .map_err(|e| OcrError::ImageDecode(e))?;

        Ok(image)
    }

    /// 异步从流加载图像（用于multipart上传）
    pub async fn from_stream<R>(mut reader: R) -> Result<DynamicImage> 
    where 
        R: AsyncRead + Unpin,
    {
        use tokio::io::AsyncReadExt;
        
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await
            .map_err(|e| OcrError::Io(e))?;

        // 检查文件大小
        if buffer.len() > 50 * 1024 * 1024 { // 50MB限制
            return Err(OcrError::FileTooLarge(buffer.len(), 50 * 1024 * 1024));
        }

        let image = image::load_from_memory(&buffer)
            .map_err(|e| OcrError::ImageDecode(e))?;

        Ok(image)
    }

    /// 检测图像格式
    pub fn detect_format(bytes: &[u8]) -> Option<ImageFormat> {
        image::guess_format(bytes).ok()
    }

    /// 验证图像格式是否支持
    pub fn is_supported_format(format: ImageFormat) -> bool {
        matches!(format, 
            ImageFormat::Png | 
            ImageFormat::Jpeg | 
            ImageFormat::Bmp | 
            ImageFormat::Tiff | 
            ImageFormat::WebP
        )
    }

    /// 转换DynamicImage为ndarray::Array3<f32> (HWC格式)
    pub fn to_array3(image: &DynamicImage) -> Array3<f32> {
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        let raw_data = rgb_image.into_raw();
        
        let mut array = Array3::<f32>::zeros((height as usize, width as usize, 3));
        
        for (i, pixel_value) in raw_data.iter().enumerate() {
            let h = (i / 3) / width as usize;
            let w = (i / 3) % width as usize;
            let c = i % 3;
            array[[h, w, c]] = *pixel_value as f32;
        }
        
        array
    }

    /// 验证图像尺寸
    pub fn validate_dimensions(image: &DynamicImage) -> Result<()> {
        let (width, height) = image.dimensions();
        
        // 检查最小尺寸
        if width < 16 || height < 16 {
            return Err(OcrError::InvalidInput(
                format!("Image too small: {}x{}, minimum 16x16", width, height)
            ));
        }
        
        // 检查最大尺寸
        if width > 8192 || height > 8192 {
            return Err(OcrError::InvalidInput(
                format!("Image too large: {}x{}, maximum 8192x8192", width, height)
            ));
        }
        
        Ok(())
    }

    /// 预处理图像，标准化颜色空间和尺寸
    pub fn preprocess(image: DynamicImage) -> Result<Array3<f32>> {
        Self::validate_dimensions(&image)?;
        
        // 转换为RGB
        let rgb_image = image.to_rgb8();
        let array = Self::to_array3(&DynamicImage::ImageRgb8(rgb_image));
        
        Ok(array)
    }
}