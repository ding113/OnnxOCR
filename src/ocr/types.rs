use serde::{Deserialize, Serialize};

/// OCR处理选项
#[derive(Debug, Clone, Deserialize)]
pub struct OcrOptions {
    /// 是否强制OCR（忽略缓存）
    #[serde(default)]
    pub force_ocr: bool,
    
    /// 输出格式 ("json", "text", "csv")
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

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            force_ocr: false,
            output_format: Some("json".to_string()),
            use_angle_cls: Some(true),
            min_confidence: Some(0.5),
            paginate_output: false,
        }
    }
}

/// OCR处理统计信息
#[derive(Debug, Clone, Serialize)]
pub struct OcrStats {
    /// 总耗时（毫秒）
    pub total_time_ms: u64,
    /// 检测耗时（毫秒）
    pub detection_time_ms: u64,
    /// 分类耗时（毫秒）
    pub classification_time_ms: u64,
    /// 识别耗时（毫秒）
    pub recognition_time_ms: u64,
    /// 后处理耗时（毫秒）
    pub postprocess_time_ms: u64,
    /// 检测到的文本框数量
    pub detected_boxes: usize,
    /// 最终识别的文本数量
    pub recognized_texts: usize,
    /// 平均置信度
    pub average_confidence: f32,
}

/// 分段处理结果（用于大图片分块处理）
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// 分块在原图中的位置偏移
    pub offset_x: usize,
    pub offset_y: usize,
    /// 分块的文本框结果
    pub text_boxes: Vec<Vec<[f32; 2]>>,
    /// 识别结果
    pub recognition_results: Vec<(String, f32)>,
}

/// OCR处理阶段
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OcrStage {
    Preprocessing,
    Detection,
    Classification,
    Recognition,
    Postprocessing,
    Completed,
    Error,
}

/// OCR处理状态
#[derive(Debug, Clone)]
pub struct OcrStatus {
    /// 当前处理阶段
    pub stage: OcrStage,
    /// 进度百分比 (0.0 - 1.0)
    pub progress: f32,
    /// 状态消息
    pub message: String,
    /// 已处理的文本框数量
    pub processed_boxes: usize,
    /// 总文本框数量
    pub total_boxes: usize,
}

impl OcrStatus {
    pub fn new(stage: OcrStage, progress: f32, message: &str) -> Self {
        Self {
            stage,
            progress,
            message: message.to_string(),
            processed_boxes: 0,
            total_boxes: 0,
        }
    }

    pub fn with_boxes(mut self, processed: usize, total: usize) -> Self {
        self.processed_boxes = processed;
        self.total_boxes = total;
        self
    }
}

// 重新导出主要类型
pub use crate::image::postprocessing::{OcrResult, OcrTextResult, ModelInfo};