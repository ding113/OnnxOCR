pub mod config;
pub mod models;
pub mod image;
pub mod ocr;
pub mod web;
pub mod utils;

// 重新导出主要类型
pub use config::Config;
pub use ocr::OcrResult;
pub use utils::error::OcrError;

pub type Result<T> = std::result::Result<T, OcrError>;