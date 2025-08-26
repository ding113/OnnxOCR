use crate::{
    image::{ImageLoader, ImagePreprocessor, ResultFormatter},
    models::{get_classifier, get_detector, get_recognizer},
    ocr::{OcrOptions, OcrResult, OcrStats, OcrStatus, OcrStage},
    Result,
};
use ndarray::Array3;
use std::time::Instant;
use tokio::sync::mpsc;

/// OCR处理流水线
pub struct OcrPipeline;

impl OcrPipeline {
    /// 处理base64图像
    pub async fn process_base64(
        base64_data: &str,
        options: OcrOptions,
        status_tx: Option<mpsc::UnboundedSender<OcrStatus>>,
    ) -> Result<OcrResult> {
        let start_time = Instant::now();
        
        // 发送预处理状态
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Preprocessing, 
                0.1, 
                "Loading image from base64"
            ));
        }

        // 加载图像
        let image = ImageLoader::from_base64(base64_data)?;
        let image_array = ImageLoader::preprocess(image)?;
        
        // 执行OCR流水线
        Self::process_image_array(image_array, options, status_tx, start_time).await
    }

    /// 处理字节流图像
    pub async fn process_bytes(
        bytes: axum::body::Bytes,
        options: OcrOptions,
        status_tx: Option<mpsc::UnboundedSender<OcrStatus>>,
    ) -> Result<OcrResult> {
        let start_time = Instant::now();
        
        // 发送预处理状态
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Preprocessing,
                0.1,
                "Loading image from stream"
            ));
        }

        // 加载图像
        let image = ImageLoader::from_bytes(bytes)?;
        let image_array = ImageLoader::preprocess(image)?;
        
        // 执行OCR流水线
        Self::process_image_array(image_array, options, status_tx, start_time).await
    }

    /// 核心图像处理流水线
    async fn process_image_array(
        image: Array3<f32>,
        options: OcrOptions,
        status_tx: Option<mpsc::UnboundedSender<OcrStatus>>,
        start_time: Instant,
    ) -> Result<OcrResult> {
        // 图像预处理
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Preprocessing,
                0.2,
                "Preprocessing image"
            ));
        }

        let preprocessed_image = ImagePreprocessor::preprocess_for_ocr(image)?;
        let preprocessing_time = start_time.elapsed();

        // 文字检测
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Detection,
                0.3,
                "Detecting text regions"
            ));
        }

        let detector = get_detector()?;
        let detection_start = Instant::now();
        let text_boxes = detector.detect(&preprocessed_image)?;
        let detection_time = detection_start.elapsed();

        if text_boxes.is_empty() {
            return Ok(Self::create_empty_result(start_time.elapsed(), options.output_format));
        }

        // 发送检测完成状态
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Detection,
                0.4,
                &format!("Detected {} text regions", text_boxes.len())
            ).with_boxes(0, text_boxes.len()));
        }

        // 文字区域裁剪
        let cropped_images = Self::crop_text_regions(&preprocessed_image, &text_boxes)?;
        
        // 角度分类（可选）
        let classification_start = Instant::now();
        let classified_images = if options.use_angle_cls.unwrap_or(true) {
            if let Some(ref tx) = status_tx {
                let _ = tx.send(OcrStatus::new(
                    OcrStage::Classification,
                    0.5,
                    "Classifying text angles"
                ));
            }

            match get_classifier()? {
                Some(classifier) => {
                    classifier.classify(cropped_images)?
                        .into_iter()
                        .map(|(img, _)| img) // 忽略角度信息，使用校正后的图像
                        .collect()
                }
                None => {
                    tracing::warn!("Angle classifier not available, skipping classification");
                    cropped_images
                }
            }
        } else {
            cropped_images
        };
        let classification_time = classification_start.elapsed();

        // 文字识别
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Recognition,
                0.7,
                "Recognizing text content"
            ));
        }

        let recognizer = get_recognizer()?;
        let recognition_start = Instant::now();
        
        // 批处理识别以提高性能
        let recognition_results = Self::batch_recognize(
            &recognizer, 
            classified_images,
            &status_tx
        ).await?;
        
        let recognition_time = recognition_start.elapsed();

        // 后处理
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Postprocessing,
                0.9,
                "Formatting results"
            ));
        }

        let postprocess_start = Instant::now();
        let min_confidence = options.min_confidence.unwrap_or(0.5);
        let total_time = start_time.elapsed();

        let result = ResultFormatter::format_result(
            text_boxes,
            recognition_results,
            total_time.as_secs_f32(),
            options.output_format,
            min_confidence,
        )?;

        let postprocess_time = postprocess_start.elapsed();

        // 发送完成状态
        if let Some(ref tx) = status_tx {
            let _ = tx.send(OcrStatus::new(
                OcrStage::Completed,
                1.0,
                &format!("OCR completed: {} texts recognized", result.results.len())
            ));
        }

        tracing::info!(
            "OCR completed: detected={}, recognized={}, total_time={:.3}s",
            text_boxes.len(),
            result.results.len(),
            total_time.as_secs_f32()
        );

        Ok(result)
    }

    /// 从文本框坐标裁剪图像区域
    fn crop_text_regions(
        image: &Array3<f32>,
        text_boxes: &[Vec<[f32; 2]>],
    ) -> Result<Vec<Array3<f32>>> {
        use crate::image::ImageTransforms;
        let mut cropped_images = Vec::with_capacity(text_boxes.len());

        for bbox in text_boxes {
            if bbox.len() != 4 {
                tracing::warn!("Invalid bounding box with {} points, expected 4", bbox.len());
                continue;
            }

            // 转换为四点数组
            let points = [
                bbox[0], bbox[1], bbox[2], bbox[3]
            ];

            // 裁剪文本区域
            match ImageTransforms::crop_polygon(image, &points) {
                Ok(cropped) => cropped_images.push(cropped),
                Err(e) => {
                    tracing::warn!("Failed to crop text region: {}", e);
                    // 可以选择跳过或使用原图的一个小区域
                }
            }
        }

        Ok(cropped_images)
    }

    /// 批处理文字识别
    async fn batch_recognize(
        recognizer: &crate::models::Recognizer,
        images: Vec<Array3<f32>>,
        status_tx: &Option<mpsc::UnboundedSender<OcrStatus>>,
    ) -> Result<Vec<(String, f32)>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = 6; // 批处理大小，可以根据内存和性能调整
        let mut all_results = Vec::with_capacity(images.len());
        
        for (i, batch) in images.chunks(batch_size).enumerate() {
            // 更新进度
            if let Some(ref tx) = status_tx {
                let progress = 0.7 + 0.2 * (i * batch_size) as f32 / images.len() as f32;
                let _ = tx.send(OcrStatus::new(
                    OcrStage::Recognition,
                    progress,
                    &format!("Processing batch {}/{}", i + 1, (images.len() + batch_size - 1) / batch_size)
                ).with_boxes(i * batch_size, images.len()));
            }

            // 批处理识别
            let batch_results = recognizer.recognize(batch.to_vec())?;
            all_results.extend(batch_results);
        }

        Ok(all_results)
    }

    /// 创建空结果（没有检测到文字时）
    fn create_empty_result(processing_time: std::time::Duration, output_format: Option<String>) -> OcrResult {
        OcrResult {
            processing_time: processing_time.as_secs_f32(),
            results: Vec::new(),
            output_format,
            model_info: Some(crate::image::postprocessing::ModelInfo {
                detector_version: "PPOCRv5".to_string(),
                recognizer_version: "PPOCRv5-SVTR_LCNet".to_string(),
                classifier_enabled: true,
            }),
        }
    }

    /// 获取处理统计信息
    pub fn create_stats(
        total_time: std::time::Duration,
        detection_time: std::time::Duration,
        classification_time: std::time::Duration,
        recognition_time: std::time::Duration,
        postprocess_time: std::time::Duration,
        detected_boxes: usize,
        recognized_texts: usize,
        average_confidence: f32,
    ) -> OcrStats {
        OcrStats {
            total_time_ms: total_time.as_millis() as u64,
            detection_time_ms: detection_time.as_millis() as u64,
            classification_time_ms: classification_time.as_millis() as u64,
            recognition_time_ms: recognition_time.as_millis() as u64,
            postprocess_time_ms: postprocess_time.as_millis() as u64,
            detected_boxes,
            recognized_texts,
            average_confidence,
        }
    }
}

// 异步trait实现，支持流式处理
#[async_trait::async_trait]
pub trait AsyncOcrProcessor {
    async fn process(&self, image: Array3<f32>) -> Result<OcrResult>;
}

#[async_trait::async_trait]
impl AsyncOcrProcessor for OcrPipeline {
    async fn process(&self, image: Array3<f32>) -> Result<OcrResult> {
        let options = OcrOptions::default();
        Self::process_image_array(image, options, None, Instant::now()).await
    }
}