pub mod handlers;
pub mod middleware;
pub mod extractors;
pub mod ui;

use crate::{models::ModelManager, Config, Result};
use axum::{
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::json;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::TcpListener;
use tower_http::{
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer,
};

pub async fn serve(config: Config) -> Result<()> {
    // 初始化模型管理器
    ModelManager::init(config.clone())?;

    // 构建应用路由
    let app = create_app(config.clone()).await?;

    // 解析绑定地址
    let addr: SocketAddr = config.bind_addr
        .parse()
        .map_err(|e| crate::utils::error::OcrError::Config(
            format!("Invalid bind address {}: {}", config.bind_addr, e)
        ))?;

    tracing::info!("Server starting on http://{}", addr);
    tracing::info!("API endpoints:");
    tracing::info!("  POST /ocr       - JSON base64 upload");
    tracing::info!("  POST /ocr/upload - Multipart file upload");
    tracing::info!("  GET  /          - Web UI");
    tracing::info!("  GET  /health    - Health check");
    tracing::info!("  GET  /api/info  - Service information");

    // 启动服务器
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| crate::utils::error::OcrError::Internal(
            format!("Failed to bind to address {}: {}", addr, e)
        ))?;
        
    axum::serve(listener, app)
        .await
        .map_err(|e| crate::utils::error::OcrError::Internal(
            format!("Server failed to start: {}", e)
        ))?;

    Ok(())
}

async fn create_app(config: Config) -> Result<Router> {
    let app = Router::new()
        // OCR API路由
        .route("/ocr", post(handlers::ocr_json_handler))
        .route("/ocr/upload", post(handlers::ocr_upload_handler))
        
        // Web UI路由
        .route("/", get(ui::index_handler))
        
        // 系统路由
        .route("/health", get(health_handler))
        .route("/api/info", get(info_handler))
        
        // 添加中间件 - 使用分层模式避免复杂类型嵌套
        .layer(RequestBodyLimitLayer::new(config.server_config.max_request_size))
        .layer(TimeoutLayer::new(Duration::from_secs(config.server_config.request_timeout)))
        .layer(CorsLayer::permissive()) // 开发环境使用宽松CORS
        // 传递配置到处理器
        .with_state(config);

    Ok(app)
}

/// 健康检查端点
async fn health_handler() -> Result<Json<serde_json::Value>> {
    match crate::models::health_check() {
        Ok(_) => Ok(Json(json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "version": env!("CARGO_PKG_VERSION")
        }))),
        Err(e) => Err(e)
    }
}

/// 服务信息端点
async fn info_handler() -> Result<Json<serde_json::Value>> {
    match crate::models::get_model_stats() {
        Ok(stats) => Ok(Json(json!({
            "service": "ONNX OCR Service",
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
            "models": stats,
            "features": {
                "dual_upload_modes": true,
                "streaming_upload": true,
                "angle_classification": stats.has_classifier,
                "batch_processing": true
            }
        }))),
        Err(e) => Err(e)
    }
}