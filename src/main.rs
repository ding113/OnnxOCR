use anyhow::Result;
use clap::Parser;
use onnx_ocr::{config::Config, web::serve};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "onnx-ocr")]
#[command(about = "High-performance ONNX-powered OCR service")]
struct Args {
    /// Server bind address
    #[arg(long, default_value = "0.0.0.0:5005")]
    bind: String,

    /// Number of worker threads
    #[arg(long)]
    workers: Option<usize>,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Model directory path
    #[arg(long, default_value = "models")]
    models_dir: String,

    /// Enable development mode
    #[arg(long)]
    dev: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化日志系统
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&args.log_level))
        )
        .with_target(false)
        .init();

    tracing::info!("Starting ONNX OCR service...");
    tracing::info!("Bind address: {}", args.bind);
    tracing::info!("Models directory: {}", args.models_dir);

    // 创建配置
    let config = Config::new(args.bind, args.models_dir, args.workers, args.dev)?;

    // 启动服务器
    serve(config).await?;

    Ok(())
}