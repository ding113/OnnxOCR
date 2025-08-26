use axum::{
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::time::Instant;

/// 请求日志中间件
pub async fn request_logging<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let user_agent = req
        .headers()
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");
    
    let start_time = Instant::now();
    
    tracing::info!(
        "Request started: {} {} - User-Agent: {}",
        method,
        uri,
        user_agent
    );

    // 执行请求
    let response = next.run(req).await;
    
    let duration = start_time.elapsed();
    let status = response.status();
    
    tracing::info!(
        "Request completed: {} {} - {} - {:.3}ms",
        method,
        uri,
        status,
        duration.as_millis()
    );

    Ok(response)
}

/// 速率限制中间件（简化版）
pub struct RateLimiter {
    // 在实际应用中，这里应该使用更复杂的限流算法
    // 比如令牌桶或滑动窗口
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn check<B>(
        &self,
        req: Request<B>,
        next: Next<B>,
    ) -> Result<Response, StatusCode> {
        // TODO: 实现真正的速率限制逻辑
        // 这里暂时简单地通过所有请求
        Ok(next.run(req).await)
    }
}

/// 安全头中间件
pub async fn security_headers<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    let mut response = next.run(req).await;
    
    // 添加安全相关的HTTP头
    let headers = response.headers_mut();
    
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    
    Ok(response)
}

/// 错误处理中间件
pub async fn error_handler<B>(
    req: Request<B>,
    next: Next<B>,
) -> Result<Response, StatusCode> {
    match next.run(req).await {
        Ok(response) => Ok(response),
        Err(err) => {
            tracing::error!("Request failed with error: {:?}", err);
            Err(err)
        }
    }
}