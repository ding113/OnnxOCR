use crate::{
    ocr::{OcrOptions, OcrPipeline},
    utils::error::OcrError,
    Config, Result,
};
use axum::{
    extract::{Multipart, State},
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::mpsc;

/// JSON请求体（base64模式）
#[derive(Debug, Deserialize)]
pub struct OcrJsonRequest {
    /// Base64编码的图像数据
    pub image: String,
    
    /// 是否强制OCR（忽略缓存）
    #[serde(default)]
    pub force_ocr: bool,
    
    /// 输出格式
    #[serde(default)]
    pub output_format: Option<String>,
    
    /// 是否启用角度分类
    #[serde(default)]
    pub use_angle_cls: Option<bool>,
    
    /// 最小置信度阈值
    #[serde(default)]
    pub min_confidence: Option<f32>,
    
    /// 是否分页输出
    #[serde(default)]
    pub paginate_output: bool,
}

/// JSON响应格式
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ApiError>,
    pub timestamp: String,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
            request_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    pub fn error(code: String, message: String) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            error: Some(ApiError { code, message }),
            timestamp: chrono::Utc::now().to_rfc3339(),
            request_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

/// JSON base64上传处理器
pub async fn ocr_json_handler(
    State(_config): State<Config>,
    Json(request): Json<OcrJsonRequest>,
) -> Result<Json<ApiResponse<crate::ocr::OcrResult>>> {
    let start_time = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    
    tracing::info!(
        "Processing JSON OCR request: request_id={}, force_ocr={}, output_format={:?}",
        request_id, request.force_ocr, request.output_format
    );

    // 验证请求参数
    if request.image.is_empty() {
        return Err(OcrError::InvalidInput("Empty image data".to_string()));
    }

    // 创建OCR选项
    let options = OcrOptions {
        force_ocr: request.force_ocr,
        output_format: request.output_format,
        use_angle_cls: request.use_angle_cls,
        min_confidence: request.min_confidence,
        paginate_output: request.paginate_output,
    };

    // 创建状态通道（可选的进度监控）
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<OcrStatus>();
    
    // 启动后台任务监控进度（开发模式）
    if config.dev_mode {
        tokio::spawn(async move {
            while let Some(status) = status_rx.recv().await {
                tracing::debug!(
                    "OCR Progress [{}]: {:?} - {:.1}% - {}",
                    request_id,
                    status.stage,
                    status.progress * 100.0,
                    status.message
                );
            }
        });
    }

    // 执行OCR处理
    let result = OcrPipeline::process_base64(
        &request.image,
        options,
        if config.dev_mode { Some(status_tx) } else { None },
    ).await?;

    let processing_time = start_time.elapsed();
    
    tracing::info!(
        "JSON OCR completed: request_id={}, texts={}, time={:.3}s",
        request_id,
        result.results.len(),
        processing_time.as_secs_f32()
    );

    Ok(Json(ApiResponse::success(result)))
}

/// Multipart文件上传处理器
pub async fn ocr_upload_handler(
    State(_config): State<Config>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse<crate::ocr::OcrResult>>> {
    let start_time = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    
    tracing::info!("Processing multipart OCR request: request_id={}", request_id);

    let mut image_data: Option<axum::body::Bytes> = None;
    let options = OcrOptions::default();

    // 解析multipart数据
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        OcrError::InvalidInput(format!("Failed to read multipart field: {}", e))
    })? {
        let field_name = field.name().unwrap_or("unknown").to_string();
        
        match field_name.as_str() {
            "file" => {
                // 验证内容类型
                if let Some(content_type) = field.content_type() {
                    if !content_type.starts_with("image/") {
                        return Err(OcrError::UnsupportedFormat(content_type.to_string()));
                    }
                }

                // 读取文件数据（流式处理）
                let data = field.bytes().await.map_err(|e| {
                    OcrError::InvalidInput(format!("Failed to read file data: {}", e))
                })?;

                if data.is_empty() {
                    return Err(OcrError::InvalidInput("Empty file".to_string()));
                }

                tracing::debug!("Received file: {} bytes", data.len());
                image_data = Some(data);
            }
            "force_ocr" => {
                let value = field.text().await.unwrap_or_default();
                options.force_ocr = value.parse().unwrap_or(false);
            }
            "output_format" => {
                let value = field.text().await.unwrap_or_default();
                if !value.is_empty() {
                    options.output_format = Some(value);
                }
            }
            "use_angle_cls" => {
                let value = field.text().await.unwrap_or_default();
                options.use_angle_cls = Some(value.parse().unwrap_or(true));
            }
            "min_confidence" => {
                let value = field.text().await.unwrap_or_default();
                if let Ok(confidence) = value.parse::<f32>() {
                    options.min_confidence = Some(confidence.clamp(0.0, 1.0));
                }
            }
            "paginate_output" => {
                let value = field.text().await.unwrap_or_default();
                options.paginate_output = value.parse().unwrap_or(false);
            }
            _ => {
                tracing::debug!("Ignoring unknown field: {}", field_name);
            }
        }
    }

    // 验证必需的图像数据
    let image_data = image_data.ok_or_else(|| {
        OcrError::InvalidInput("No image file provided".to_string())
    })?;

    // 创建状态通道（可选的进度监控）
    let (status_tx, mut status_rx) = mpsc::unbounded_channel::<OcrStatus>();
    
    // 启动后台任务监控进度（开发模式）
    if config.dev_mode {
        tokio::spawn(async move {
            while let Some(status) = status_rx.recv().await {
                tracing::debug!(
                    "OCR Progress [{}]: {:?} - {:.1}% - {}",
                    request_id,
                    status.stage,
                    status.progress * 100.0,
                    status.message
                );
            }
        });
    }

    // 执行OCR处理
    let result = OcrPipeline::process_bytes(
        image_data,
        options,
        if config.dev_mode { Some(status_tx) } else { None },
    ).await?;

    let processing_time = start_time.elapsed();
    
    tracing::info!(
        "Upload OCR completed: request_id={}, texts={}, time={:.3}s",
        request_id,
        result.results.len(),
        processing_time.as_secs_f32()
    );

    Ok(Json(ApiResponse::success(result)))
}

/// 批处理上传处理器（支持多个文件）
pub async fn ocr_batch_handler(
    State(_config): State<Config>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse<Vec<crate::ocr::OcrResult>>>> {
    let start_time = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    
    tracing::info!("Processing batch OCR request: request_id={}", request_id);

    let mut files = Vec::new();
    let options = OcrOptions::default();

    // 解析multipart数据
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        OcrError::InvalidInput(format!("Failed to read multipart field: {}", e))
    })? {
        let field_name = field.name().unwrap_or("unknown").to_string();
        
        match field_name.as_str() {
            "files" => {
                // 验证内容类型
                if let Some(content_type) = field.content_type() {
                    if !content_type.starts_with("image/") {
                        continue; // 跳过非图像文件
                    }
                }

                // 读取文件数据
                let data = field.bytes().await.map_err(|e| {
                    OcrError::InvalidInput(format!("Failed to read file data: {}", e))
                })?;

                if !data.is_empty() {
                    files.push(data);
                }
            }
            // 其他选项字段的处理与单文件上传相同...
            _ => {}
        }
    }

    if files.is_empty() {
        return Err(OcrError::InvalidInput("No valid image files provided".to_string()));
    }

    // 并行处理多个文件
    let mut results = Vec::with_capacity(files.len());
    
    for (i, file_data) in files.into_iter().enumerate() {
        tracing::debug!("Processing file {} of batch {}", i + 1, request_id);
        
        let result = OcrPipeline::process_bytes(
            file_data,
            options.clone(),
            None, // 批处理模式不提供进度监控
        ).await?;
        
        results.push(result);
    }

    let processing_time = start_time.elapsed();
    let total_texts: usize = results.iter().map(|r| r.results.len()).sum();
    
    tracing::info!(
        "Batch OCR completed: request_id={}, files={}, total_texts={}, time={:.3}s",
        request_id,
        results.len(),
        total_texts,
        processing_time.as_secs_f32()
    );

    Ok(Json(ApiResponse::success(results)))
}