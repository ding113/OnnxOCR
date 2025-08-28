use crate::utils::error::OcrError;
use crate::{Config, Result};
use ndarray::{Array3, Axis};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
    inputs,
};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct Classifier {
    session: Arc<Mutex<Session>>,
    output_name: String, // 动态发现的输出名称
    input_size: (usize, usize, usize), // (C, H, W)
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

        // 动态发现输出名称
        let output_name = if session.outputs.is_empty() {
            return Err(OcrError::ModelLoad(
                "Classification model has no outputs".to_string()
            ));
        } else {
            let output_name = session.outputs[0].name.clone();
            tracing::info!("Classification model output: '{}'", output_name);
            
            // 记录所有可用输出用于调试
            for (i, output) in session.outputs.iter().enumerate() {
                tracing::debug!("Classification output[{}]: '{}'", i, output.name);
            }
            
            output_name
        };

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            output_name,
            input_size: (3, 48, 192), // PPOCRv4 分类器默认输入尺寸
            thresh: 0.9,
        })
    }

    /// 文字方向分类
    pub fn classify(&self, images: Vec<Array3<f32>>) -> Result<Vec<(Array3<f32>, f32)>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(images.len());
        
        for image in images {
            let (processed_image, angle) = self.classify_single(&image)?;
            results.push((processed_image, angle));
        }

        Ok(results)
    }

    fn classify_single(&self, image: &Array3<f32>) -> Result<(Array3<f32>, f32)> {
        // 预处理图像
        let processed_input = self.preprocess(image)?;
        
        // 添加batch维度
        let input_tensor = processed_input.insert_axis(Axis(0));
        
        // 推理
        let input_tensor = Tensor::from_array(input_tensor)?;
        let predictions = {
            let mut session = self.session.lock();
            let outputs = session.run(inputs!["x" => input_tensor])?;
            
            // 使用动态发现的输出名称
            match outputs.get(&self.output_name) {
                Some(output) => output.try_extract_array::<f32>()?.into_owned(),
                None => {
                    // 提供详细的错误诊断信息
                    let available_outputs: Vec<String> = outputs.keys().map(|s| s.to_string()).collect();
                    return Err(OcrError::Inference(format!(
                        "Classification output '{}' not found. Available outputs: {:?}",
                        self.output_name, available_outputs
                    )));
                }
            }
        };

        // 解析分类结果
        let angle = self.parse_classification(&predictions.view())?;
        
        // 根据角度旋转图像
        let rotated_image = if angle.abs() > self.thresh {
            self.rotate_image(image, angle)?
        } else {
            image.clone()
        };
        
        Ok((rotated_image, angle))
    }

    /// 图像预处理
    fn preprocess(&self, image: &Array3<f32>) -> Result<Array3<f32>> {
        let (orig_h, orig_w) = (image.shape()[0], image.shape()[1]);
        let (target_c, target_h, target_w) = self.input_size;

        // 调整图像大小，保持宽高比
        let scale = (target_h as f32 / orig_h as f32).min(target_w as f32 / orig_w as f32);
        let new_h = (orig_h as f32 * scale) as usize;
        let new_w = (orig_w as f32 * scale) as usize;

        // 创建目标尺寸的图像，用灰色填充
        let mut processed = Array3::<f32>::from_elem((target_c, target_h, target_w), 0.5);
        
        // 中心对齐
        let start_h = (target_h - new_h) / 2;
        let start_w = (target_w - new_w) / 2;
        
        // 简化的resize和padding逻辑
        for c in 0..target_c {
            for h in 0..new_h {
                for w in 0..new_w {
                    let src_h = (h as f32 / scale) as usize;
                    let src_w = (w as f32 / scale) as usize;
                    if src_h < orig_h && src_w < orig_w && c < image.shape()[2] {
                        processed[[c, start_h + h, start_w + w]] = image[[src_h, src_w, c]] / 255.0;
                    }
                }
            }
        }

        Ok(processed)
    }

    /// 解析分类结果
    fn parse_classification(&self, predictions: &ndarray::ArrayViewD<f32>) -> Result<f32> {
        let pred_shape = predictions.shape();
        if pred_shape.len() != 2 {
            return Err(OcrError::Inference(
                "Expected 2D classification tensor".to_string()
            ));
        }

        let (batch_size, num_classes) = (pred_shape[0], pred_shape[1]);
        
        if batch_size != 1 {
            return Err(OcrError::Inference(
                "Expected batch size 1 for classification".to_string()
            ));
        }

        // 找到最大概率的类别
        let mut max_prob = 0.0;
        let mut max_idx = 0;
        
        for i in 0..num_classes {
            let prob = predictions[[0, i]];
            if prob > max_prob {
                max_prob = prob;
                max_idx = i;
            }
        }
        
        // 根据类别索引返回角度
        // 通常: 0=0度, 1=180度
        let angle = match max_idx {
            0 => 0.0,    // 正向
            1 => 180.0,  // 180度旋转
            _ => 0.0,
        };
        
        Ok(angle)
    }

    /// 旋转图像
    fn rotate_image(&self, image: &Array3<f32>, angle: f32) -> Result<Array3<f32>> {
        // 简化实现：只处理180度旋转
        if (angle - 180.0).abs() < 1.0 {
            // 180度旋转 = 上下翻转 + 左右翻转
            let (h, w, c) = image.dim();
            let mut rotated = Array3::<f32>::zeros((h, w, c));
            
            for y in 0..h {
                for x in 0..w {
                    for ch in 0..c {
                        rotated[[h - 1 - y, w - 1 - x, ch]] = image[[y, x, ch]];
                    }
                }
            }
            
            Ok(rotated)
        } else {
            // 其他角度暂不处理，返回原图
            Ok(image.clone())
        }
    }
}