use crate::models::{Classifier, Detector, Recognizer};
use crate::utils::error::OcrError;
use crate::{Config, Result};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::sync::Arc;

/// 全局模型管理器单例
pub struct ModelManager {
    detector: Arc<Detector>,
    recognizer: Arc<Recognizer>,
    classifier: Option<Arc<Classifier>>,
    config: Config,
}

static MODEL_MANAGER: OnceCell<Arc<Mutex<ModelManager>>> = OnceCell::new();

impl ModelManager {
    /// 初始化全局模型管理器
    pub fn init(config: Config) -> Result<()> {
        tracing::info!("Initializing model manager...");

        let detector = Arc::new(Detector::new(&config)?);
        let recognizer = Arc::new(Recognizer::new(&config)?);
        
        // 分类器是可选的
        let classifier = if config.cls_model_path().exists() {
            match Classifier::new(&config) {
                Ok(cls) => {
                    tracing::info!("Classification model loaded successfully");
                    Some(Arc::new(cls))
                }
                Err(e) => {
                    tracing::warn!("Failed to load classification model: {}", e);
                    None
                }
            }
        } else {
            tracing::info!("Classification model not found, skipping angle classification");
            None
        };

        let manager = ModelManager {
            detector,
            recognizer,
            classifier,
            config,
        };

        MODEL_MANAGER.set(Arc::new(Mutex::new(manager)))
            .map_err(|_| OcrError::Internal("Failed to initialize model manager".to_string()))?;

        tracing::info!("Model manager initialized successfully");
        Ok(())
    }

    /// 获取全局模型管理器实例
    pub fn instance() -> Result<Arc<Mutex<ModelManager>>> {
        MODEL_MANAGER.get()
            .cloned()
            .ok_or_else(|| OcrError::Internal("Model manager not initialized".to_string()))
    }

    /// 获取检测器引用
    pub fn detector(&self) -> Arc<Detector> {
        Arc::clone(&self.detector)
    }

    /// 获取识别器引用
    pub fn recognizer(&self) -> Arc<Recognizer> {
        Arc::clone(&self.recognizer)
    }

    /// 获取分类器引用（如果可用）
    pub fn classifier(&self) -> Option<Arc<Classifier>> {
        self.classifier.as_ref().map(Arc::clone)
    }

    /// 检查是否启用角度分类
    pub fn has_classifier(&self) -> bool {
        self.classifier.is_some()
    }

    /// 获取配置引用
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// 模型健康检查
    pub fn health_check(&self) -> Result<()> {
        tracing::debug!("Performing model health check...");
        
        // TODO: 实现基础的模型健康检查
        // 可以包括：
        // 1. 模型文件是否存在
        // 2. 模型是否可以正常推理（使用测试数据）
        // 3. 内存使用情况检查
        
        tracing::debug!("Model health check passed");
        Ok(())
    }

    /// 获取模型统计信息
    pub fn get_stats(&self) -> ModelStats {
        ModelStats {
            has_detector: true,
            has_recognizer: true,
            has_classifier: self.has_classifier(),
            intra_threads: self.config.onnx_config.intra_threads,
            optimization_level: self.config.onnx_config.optimization_level,
        }
    }
}

/// 模型统计信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelStats {
    pub has_detector: bool,
    pub has_recognizer: bool,
    pub has_classifier: bool,
    pub intra_threads: usize,
    pub optimization_level: i32,
}

/// 便捷函数：获取检测器
pub fn get_detector() -> Result<Arc<Detector>> {
    let manager = ModelManager::instance()?;
    let guard = manager.lock();
    Ok(guard.detector())
}

/// 便捷函数：获取识别器
pub fn get_recognizer() -> Result<Arc<Recognizer>> {
    let manager = ModelManager::instance()?;
    let guard = manager.lock();
    Ok(guard.recognizer())
}

/// 便捷函数：获取分类器（如果可用）
pub fn get_classifier() -> Result<Option<Arc<Classifier>>> {
    let manager = ModelManager::instance()?;
    let guard = manager.lock();
    Ok(guard.classifier())
}

/// 便捷函数：检查模型健康状态
pub fn health_check() -> Result<()> {
    let manager = ModelManager::instance()?;
    let guard = manager.lock();
    guard.health_check()
}

/// 便捷函数：获取模型统计信息
pub fn get_model_stats() -> Result<ModelStats> {
    let manager = ModelManager::instance()?;
    let guard = manager.lock();
    Ok(guard.get_stats())
}