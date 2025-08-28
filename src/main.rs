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

    // 设置 panic hook
    std::panic::set_hook(Box::new(|panic_info| {
        let location = panic_info.location().unwrap();
        let msg = match panic_info.payload().downcast_ref::<&'static str>() {
            Some(s) => *s,
            None => match panic_info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<dyn Any>",
            }
        };
        
        tracing::error!(
            "Panic occurred: {} at {}:{}",
            msg,
            location.file(),
            location.line()
        );
    }));

    // 初始化日志系统
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&args.log_level))
        )
        .with_target(false)
        .init();

    // 减少 ONNX Runtime 日志噪音
    std::env::set_var("ORT_LOG_LEVEL", "3"); // 只显示 Error 级别日志

    tracing::info!("Starting ONNX OCR service...");
    tracing::info!("Bind address: {}", args.bind);
    tracing::info!("Models directory: {}", args.models_dir);

    // 创建配置
    let config = Config::new(args.bind, args.models_dir, args.workers, args.dev)?;

    // 启动服务器
    serve(config).await?;

    Ok(())
}