use crate::utils::error::OcrError;
use crate::{Config, Result};
use ndarray::Array3;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
    inputs,
};
use std::fs;
use std::sync::Arc;

pub struct Recognizer {
    session: Arc<Session>,
    input_size: (usize, usize, usize), // (C, H, W)
    dict: Vec<String>,
}

impl Recognizer {
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.rec_model_path();
        let dict_path = config.dict_path();
        
        if !model_path.exists() {
            return Err(OcrError::ModelLoad(
                format!("Recognition model not found: {}", model_path.display())
            ));
        }

        if !dict_path.exists() {
            return Err(OcrError::ModelLoad(
                format!("Dictionary file not found: {}", dict_path.display())
            ));
        }

        tracing::info!("Loading recognition model from: {}", model_path.display());
        tracing::info!("Loading dictionary from: {}", dict_path.display());
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.onnx_config.intra_threads)?
            .commit_from_file(&model_path)?;

        // 加载字典
        let dict_content = fs::read_to_string(&dict_path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to read dictionary: {}", e)))?;
        
        let mut dict = vec!["blank".to_string()]; // CTC blank token
        for line in dict_content.lines() {
            let char = line.trim();
            if !char.is_empty() {
                dict.push(char.to_string());
            }
        }
        dict.push(" ".to_string()); // 空格符

        tracing::info!("Loaded dictionary with {} characters", dict.len());

        Ok(Self {
            session: Arc::new(session),
            input_size: (3, 48, 320), // PPOCRv5 识别器默认输入尺寸
            dict,
        })
    }

    /// 文字识别
    pub fn recognize(&self, images: Vec<Array3<f32>>) -> Result<Vec<(String, f32)>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(images.len());
        
        // 批处理识别（简化版本，实际应支持动态批处理）
        for image in images {
            let (text, confidence) = self.recognize_single(&image)?;
            results.push((text, confidence));
        }

        Ok(results)
    }

    fn recognize_single(&self, image: &Array3<f32>) -> Result<(String, f32)> {
        // 预处理图像
        let processed_image = self.preprocess(image)?;
        
        // 添加batch维度
        let input_tensor = processed_image.insert_axis(ndarray::Axis(0));
        
        // 推理
        let input_tensor = Tensor::from_array(input_tensor)?;
        let outputs = self.session.as_ref().run(inputs!["x" => input_tensor])?;
        let predictions = outputs["softmax_2.tmp_0"]
            .try_extract_array::<f32>()?;

        // CTC解码
        let (text, confidence) = self.ctc_decode(&predictions)?;
        
        Ok((text, confidence))
    }

    /// 图像预处理
    fn preprocess(&self, image: &Array3<f32>) -> Result<Array3<f32>> {
        let (orig_h, orig_w) = (image.shape()[0], image.shape()[1]);
        let (target_c, target_h, target_w) = self.input_size;

        // 调整图像大小，保持宽高比
        let scale = (target_h as f32 / orig_h as f32).min(target_w as f32 / orig_w as f32);
        let new_h = (orig_h as f32 * scale) as usize;
        let new_w = (orig_w as f32 * scale) as usize;

        // 创建目标尺寸的图像，用白色填充
        let mut processed = Array3::<f32>::ones((target_c, target_h, target_w));
        
        // 简化的resize和padding逻辑
        for c in 0..target_c {
            for h in 0..new_h.min(target_h) {
                for w in 0..new_w.min(target_w) {
                    let src_h = (h as f32 / scale) as usize;
                    let src_w = (w as f32 / scale) as usize;
                    if src_h < orig_h && src_w < orig_w && c < image.shape()[2] {
                        processed[[c, h, w]] = image[[src_h, src_w, c]] / 255.0;
                    }
                }
            }
        }

        Ok(processed)
    }

    /// CTC解码器
    fn ctc_decode(&self, predictions: &ndarray::ArrayViewD<f32>) -> Result<(String, f32)> {
        let pred_shape = predictions.shape();
        if pred_shape.len() != 3 {
            return Err(OcrError::Inference(
                "Expected 3D prediction tensor".to_string()
            ));
        }

        let (batch_size, seq_len, vocab_size) = (pred_shape[0], pred_shape[1], pred_shape[2]);
        
        if batch_size != 1 {
            return Err(OcrError::Inference(
                "Expected batch size 1 for recognition".to_string()
            ));
        }

        let mut decoded_text = String::new();
        let mut confidence_sum = 0.0;
        let mut valid_chars = 0;
        let mut last_char_idx = None;

        // 简化的贪心CTC解码
        for t in 0..seq_len {
            let mut max_prob = 0.0;
            let mut max_idx = 0;
            
            // 找到最大概率的字符
            for c in 0..vocab_size.min(self.dict.len()) {
                let prob = predictions[[0, t, c]];
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = c;
                }
            }
            
            // CTC规则：忽略重复字符和blank token
            if max_idx != 0 && Some(max_idx) != last_char_idx { // 0是blank token
                if let Some(char) = self.dict.get(max_idx) {
                    decoded_text.push_str(char);
                    confidence_sum += max_prob;
                    valid_chars += 1;
                }
            }
            
            last_char_idx = Some(max_idx);
        }

        let avg_confidence = if valid_chars > 0 {
            confidence_sum / valid_chars as f32
        } else {
            0.0
        };

        Ok((decoded_text, avg_confidence))
    }
}