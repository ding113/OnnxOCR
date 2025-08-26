use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),

    #[error("Image processing failed: {0}")]
    ImageProcessing(String),

    #[error("OCR inference failed: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("File too large: {0} bytes, max allowed: {1} bytes")]
    FileTooLarge(usize, usize),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("Image decode error: {0}")]
    ImageDecode(#[from] image::ImageError),

    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Internal server error: {0}")]
    Internal(String),
}

impl OcrError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            OcrError::InvalidInput(_) => StatusCode::BAD_REQUEST,
            OcrError::FileTooLarge(_, _) => StatusCode::PAYLOAD_TOO_LARGE,
            OcrError::UnsupportedFormat(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            OcrError::Base64(_) => StatusCode::BAD_REQUEST,
            OcrError::Json(_) => StatusCode::BAD_REQUEST,
            OcrError::ModelLoad(_) => StatusCode::SERVICE_UNAVAILABLE,
            OcrError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            OcrError::ModelLoad(_) => "MODEL_LOAD_ERROR",
            OcrError::ImageProcessing(_) => "IMAGE_PROCESSING_ERROR", 
            OcrError::Inference(_) => "INFERENCE_ERROR",
            OcrError::InvalidInput(_) => "INVALID_INPUT",
            OcrError::FileTooLarge(_, _) => "FILE_TOO_LARGE",
            OcrError::UnsupportedFormat(_) => "UNSUPPORTED_FORMAT",
            OcrError::Config(_) => "CONFIG_ERROR",
            OcrError::Io(_) => "IO_ERROR",
            OcrError::Json(_) => "JSON_ERROR",
            OcrError::Base64(_) => "BASE64_DECODE_ERROR",
            OcrError::ImageDecode(_) => "IMAGE_DECODE_ERROR",
            OcrError::Ort(_) => "ORT_ERROR",
            OcrError::Internal(_) => "INTERNAL_ERROR",
        }
    }
}

impl IntoResponse for OcrError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_response = serde_json::json!({
            "error": {
                "code": self.error_code(),
                "message": self.to_string(),
            }
        });

        tracing::error!("Request failed: {} ({})", self, status);

        (status, axum::Json(error_response)).into_response()
    }
}