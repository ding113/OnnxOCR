use crate::utils::error::OcrError;
use crate::{Config, Result};
use ndarray::Array3;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
    inputs,
};
use parking_lot::Mutex;
use std::fs;
use std::sync::Arc;

pub struct Recognizer {
    session: Arc<Mutex<Session>>,
    output_name: String, // 动态发现的输出名称
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

        // 动态发现输出名称
        let output_name = if session.outputs.is_empty() {
            return Err(OcrError::ModelLoad(
                "Recognition model has no outputs".to_string()
            ));
        } else {
            let output_name = session.outputs[0].name.clone();
            tracing::info!("Recognition model output: '{}'", output_name);
            
            // 记录所有可用输出用于调试
            for (i, output) in session.outputs.iter().enumerate() {
                tracing::debug!("Recognition output[{}]: '{}'", i, output.name);
            }
            
            output_name
        };

        // 加载字典 - 基于Python的BaseRecLabelDecode实现
        let dict_content = fs::read_to_string(&dict_path)
            .map_err(|e| OcrError::ModelLoad(format!("Failed to read dictionary: {}", e)))?;
        
        // 构建字符列表（不包含特殊字符）
        let mut character_list = Vec::new();
        for line in dict_content.lines() {
            let char = line.trim();
            if !char.is_empty() {
                character_list.push(char.to_string());
            }
        }
        
        // 添加空格字符（如果使用空格）
        character_list.push(" ".to_string());
        
        // CTC需要在开头添加blank token - 对应Python的add_special_char
        let mut dict = vec!["blank".to_string()];
        dict.extend(character_list);

        tracing::info!("Loaded dictionary with {} characters (including blank)", dict.len());
        tracing::debug!("First 10 chars: {:?}", dict.iter().take(10).collect::<Vec<_>>());

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            output_name,
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
        
        // 批处理识别（TODO: 应支持动态批处理）
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
        
        // 推理 - 立即提取数据避免生命周期冲突
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
                        "Output '{}' not found. Available outputs: {:?}",
                        self.output_name, available_outputs
                    )));
                }
            }
        };

        // CTC解码
        let (text, confidence) = self.ctc_decode(&predictions.view())?;
        
        Ok((text, confidence))
    }

    /// 图像预处理 - 基于Python的resize_norm_img实现
    fn preprocess(&self, image: &Array3<f32>) -> Result<Array3<f32>> {
        let (orig_h, orig_w) = (image.shape()[0], image.shape()[1]);
        let (target_c, target_h, target_w) = self.input_size;
        
        // 确保输入图像是HWC格式的
        if image.shape()[2] != target_c {
            return Err(OcrError::Inference(format!(
                "Expected {} channels, got {}", target_c, image.shape()[2]
            )));
        }

        // 计算宽高比和调整后的宽度（PPOCRv5算法）
        let ratio = orig_w as f32 / orig_h as f32;
        let resized_w = if (target_h as f32 * ratio).ceil() as usize > target_w {
            target_w
        } else {
            (target_h as f32 * ratio).ceil() as usize
        };
        
        // Step 1: 使用双线性插值调整图像大小到 (resized_w, target_h)
        let resized_image = self.resize_image(image, resized_w, target_h)?;
        
        // Step 2: 归一化到[-1, 1]范围
        let mut normalized_image = resized_image / 255.0; // 先归一化到[0, 1]
        normalized_image -= 0.5; // 减去0.5，变为[-0.5, 0.5]
        normalized_image /= 0.5;  // 除以0.5，变为[-1, 1]
        
        // Step 3: 转换为CHW格式并填充到目标尺寸
        let mut padding_image = Array3::<f32>::zeros((target_c, target_h, target_w));
        
        // 复制调整后的图像到左侧，右侧保持为0（黑色填充）
        for c in 0..target_c {
            for h in 0..target_h {
                for w in 0..resized_w.min(target_w) {
                    padding_image[[c, h, w]] = normalized_image[[h, w, c]];
                }
            }
        }

        Ok(padding_image)
    }
    
    /// 简化的双线性插值图像缩放
    fn resize_image(&self, image: &Array3<f32>, new_w: usize, new_h: usize) -> Result<Array3<f32>> {
        let (orig_h, orig_w, channels) = (image.shape()[0], image.shape()[1], image.shape()[2]);
        let mut resized = Array3::<f32>::zeros((new_h, new_w, channels));
        
        let scale_x = orig_w as f32 / new_w as f32;
        let scale_y = orig_h as f32 / new_h as f32;
        
        for c in 0..channels {
            for y in 0..new_h {
                for x in 0..new_w {
                    // 简化的最近邻插值
                    let orig_x = (x as f32 * scale_x) as usize;
                    let orig_y = (y as f32 * scale_y) as usize;
                    
                    if orig_y < orig_h && orig_x < orig_w {
                        resized[[y, x, c]] = image[[orig_y, orig_x, c]];
                    }
                }
            }
        }
        
        Ok(resized)
    }

    /// CTC解码器 - 基于Python的CTCLabelDecode实现
    fn ctc_decode(&self, predictions: &ndarray::ArrayViewD<f32>) -> Result<(String, f32)> {
        let pred_shape = predictions.shape();
        if pred_shape.len() != 3 {
            return Err(OcrError::Inference(
                format!("Expected 3D prediction tensor, got {}D", pred_shape.len())
            ));
        }

        let (batch_size, seq_len, vocab_size) = (pred_shape[0], pred_shape[1], pred_shape[2]);
        
        if batch_size != 1 {
            return Err(OcrError::Inference(
                format!("Expected batch size 1, got {}", batch_size)
            ));
        }
        
        // 验证vocab_size与字典大小匹配
        if vocab_size != self.dict.len() {
            tracing::warn!("Model vocab size ({}) != dict size ({})", vocab_size, self.dict.len());
        }

        // Step 1: 计算argmax和max - 对应Python的preds.argmax(axis=2)和preds.max(axis=2)
        let mut preds_idx = Vec::with_capacity(seq_len);
        let mut preds_prob = Vec::with_capacity(seq_len);
        
        for t in 0..seq_len {
            let mut max_prob = f32::NEG_INFINITY;
            let mut max_idx = 0;
            
            for c in 0..vocab_size {
                let prob = predictions[[0, t, c]];
                if prob > max_prob {
                    max_prob = prob;
                    max_idx = c;
                }
            }
            
            preds_idx.push(max_idx);
            preds_prob.push(max_prob);
        }
        
        // Step 2: CTC解码 - 基于Python的decode方法实现
        let mut char_list = Vec::new();
        let mut conf_list = Vec::new();
        
        // 去重复处理：selection[1:] = text_index[1:] != text_index[:-1]
        let mut selection = vec![true; seq_len];
        for i in 1..seq_len {
            selection[i] = preds_idx[i] != preds_idx[i - 1];
        }
        
        // 忽略blank token (idx=0) 和应用selection
        for (i, &pred_idx) in preds_idx.iter().enumerate() {
            // 忽略blank token并应用去重复选择
            if pred_idx != 0 && selection[i] {
                if let Some(character) = self.dict.get(pred_idx) {
                    if !character.is_empty() && character != "blank" {
                        char_list.push(character.clone());
                        conf_list.push(preds_prob[i]);
                    }
                } else {
                    tracing::warn!("Character index {} out of dictionary bounds ({})", pred_idx, self.dict.len());
                }
            }
        }
        
        // 计算平均置信度
        let avg_confidence = if conf_list.is_empty() {
            0.0
        } else {
            conf_list.iter().sum::<f32>() / conf_list.len() as f32
        };
        
        let decoded_text = char_list.join("");
        
        tracing::debug!(
            "CTC decode: seq_len={}, chars={}, text='{}', confidence={:.3}",
            seq_len, char_list.len(), decoded_text, avg_confidence
        );
        
        Ok((decoded_text, avg_confidence))
    }
}