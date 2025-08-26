use axum::{
    async_trait,
    extract::{FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

/// 验证的JSON提取器
pub struct ValidatedJson<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for ValidatedJson<T>
where
    T: for<'de> Deserialize<'de> + Validate,
    S: Send + Sync,
{
    type Rejection = ValidationError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state)
            .await
            .map_err(|err| ValidationError::JsonParse(err.to_string()))?;

        value.validate()
            .map_err(ValidationError::Validation)?;

        Ok(ValidatedJson(value))
    }
}

/// 验证trait
pub trait Validate {
    type Error: std::fmt::Display;
    
    fn validate(&self) -> Result<(), Self::Error>;
}

/// 验证错误类型
#[derive(Debug)]
pub enum ValidationError {
    JsonParse(String),
    Validation(String),
}

impl IntoResponse for ValidationError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ValidationError::JsonParse(msg) => {
                (StatusCode::BAD_REQUEST, format!("JSON parse error: {}", msg))
            }
            ValidationError::Validation(msg) => {
                (StatusCode::BAD_REQUEST, format!("Validation error: {}", msg))
            }
        };

        let body = serde_json::json!({
            "error": {
                "code": "VALIDATION_ERROR",
                "message": error_message
            }
        });

        (status, Json(body)).into_response()
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::JsonParse(msg) => write!(f, "JSON parse error: {}", msg),
            ValidationError::Validation(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for ValidationError {}

/// 为OCR请求实现验证
impl Validate for crate::web::handlers::OcrJsonRequest {
    type Error = String;
    
    fn validate(&self) -> Result<(), Self::Error> {
        // 验证image字段
        if self.image.trim().is_empty() {
            return Err("Image data cannot be empty".to_string());
        }

        // 验证置信度范围
        if let Some(confidence) = self.min_confidence {
            if !(0.0..=1.0).contains(&confidence) {
                return Err("Confidence must be between 0.0 and 1.0".to_string());
            }
        }

        // 验证输出格式
        if let Some(ref format) = self.output_format {
            let valid_formats = ["json", "text", "csv"];
            if !valid_formats.contains(&format.as_str()) {
                return Err(format!(
                    "Invalid output format '{}'. Supported formats: {}",
                    format,
                    valid_formats.join(", ")
                ));
            }
        }

        Ok(())
    }
}

/// 请求ID提取器
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

#[async_trait]
impl<S> FromRequest<S> for RequestId
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request(req: Request, _state: &S) -> Result<Self, Self::Rejection> {
        let request_id = req
            .headers()
            .get("X-Request-ID")
            .and_then(|value| value.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        Ok(RequestId(request_id))
    }
}