use crate::utils::error::OcrError;
use crate::{Config, Result};
use ndarray::Array3;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct Classifier {
    #[allow(dead_code)]
    session: Arc<Mutex<Session>>,
    #[allow(dead_code)]
    input_size: (usize, usize, usize), // (C, H, W)
    #[allow(dead_code)]
    thresh: f32,
}

impl Classifier {
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.cls_model_path();
        
        if !model_path.exists() {
            return Err(OcrError::ModelLoad(
                format!("Classification model not found: {}", model_path.display())
            ));
        }

        tracing::info!("Loading classification model from: {}", model_path.display());
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.onnx_config.intra_threads)?
            .commit_from_file(&model_path)?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            input_size: (3, 48, 192), // PPOCRv4 分类器默认输入尺寸
            thresh: 0.9,
        })
    }

    /// 文字方向分类
    pub fn classify(&self, images: Vec<Array3<f32>>) -> Result<Vec<(Array3<f32>, bool)>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(images.len());
        
        for image in images {
            let (processed_image, is_rotated) = self.classify_single(&image)?;
            results.push((processed_image, is_rotated));
        }

        Ok(results)
    }

    fn classify_single(&self, image: &Array3<f32>) -> Result<(Array3<f32>, bool)> {
        // TODO: 实现方向分类逻辑
        // 这里先返回原图像，实际实现需要：
        // 1. 预处理图像到标准尺寸
        // 2. 推理得到角度分类结果
        // 3. 根据分类结果旋转图像
        
        Ok((image.clone(), false))
    }
}