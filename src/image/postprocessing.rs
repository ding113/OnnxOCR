use crate::Result;
use serde::{Deserialize, Serialize};

/// OCR识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrTextResult {
    /// 识别的文本内容
    pub text: String,
    /// 置信度分数 (0.0 - 1.0)
    pub confidence: f32,
    /// 文本框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pub bounding_box: Vec<[f32; 2]>,
}

/// 完整的OCR处理结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    /// 处理耗时（秒）
    pub processing_time: f32,
    /// 识别结果列表
    pub results: Vec<OcrTextResult>,
    /// 输出格式
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// 模型信息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<ModelInfo>,
}

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 检测模型版本
    pub detector_version: String,
    /// 识别模型版本
    pub recognizer_version: String,
    /// 是否启用分类器
    pub classifier_enabled: bool,
}

/// 结果格式化器
pub struct ResultFormatter;

impl ResultFormatter {
    /// 格式化OCR结果
    pub fn format_result(
        text_boxes: Vec<Vec<[f32; 2]>>,
        recognition_results: Vec<(String, f32)>,
        processing_time: f32,
        output_format: Option<String>,
        min_confidence: f32,
    ) -> Result<OcrResult> {
        let mut ocr_results = Vec::new();

        // 确保文本框和识别结果数量匹配
        let min_len = text_boxes.len().min(recognition_results.len());
        
        for i in 0..min_len {
            let (text, confidence) = &recognition_results[i];
            
            // 过滤低置信度结果
            if *confidence >= min_confidence && !text.trim().is_empty() {
                let result = OcrTextResult {
                    text: text.clone(),
                    confidence: *confidence,
                    bounding_box: text_boxes[i].clone(),
                };
                ocr_results.push(result);
            }
        }

        // 按坐标排序（从上到下，从左到右）
        ocr_results.sort_by(|a, b| {
            let a_center_y = (a.bounding_box[0][1] + a.bounding_box[2][1]) / 2.0;
            let b_center_y = (b.bounding_box[0][1] + b.bounding_box[2][1]) / 2.0;
            
            let y_diff = (a_center_y - b_center_y).abs();
            
            if y_diff < 10.0 { // 同一行的文本
                let a_center_x = (a.bounding_box[0][0] + a.bounding_box[2][0]) / 2.0;
                let b_center_x = (b.bounding_box[0][0] + b.bounding_box[2][0]) / 2.0;
                a_center_x.partial_cmp(&b_center_x).unwrap()
            } else {
                a_center_y.partial_cmp(&b_center_y).unwrap()
            }
        });

        let model_info = Some(ModelInfo {
            detector_version: "PPOCRv5".to_string(),
            recognizer_version: "PPOCRv5-SVTR_LCNet".to_string(),
            classifier_enabled: true, // 从模型管理器获取实际状态
        });

        Ok(OcrResult {
            processing_time,
            results: ocr_results,
            output_format,
            model_info,
        })
    }

    /// 格式化为简化版本（仅包含文本和置信度）
    pub fn format_simple(
        recognition_results: Vec<(String, f32)>,
        processing_time: f32,
        min_confidence: f32,
    ) -> Result<SimpleOcrResult> {
        let filtered_results: Vec<_> = recognition_results
            .into_iter()
            .filter(|(text, confidence)| *confidence >= min_confidence && !text.trim().is_empty())
            .collect();

        Ok(SimpleOcrResult {
            processing_time,
            text_count: filtered_results.len(),
            texts: filtered_results.into_iter().map(|(text, _)| text).collect(),
        })
    }

    /// 格式化为纯文本输出
    pub fn format_plain_text(
        recognition_results: Vec<(String, f32)>,
        min_confidence: f32,
    ) -> String {
        recognition_results
            .into_iter()
            .filter(|(text, confidence)| *confidence >= min_confidence && !text.trim().is_empty())
            .map(|(text, _)| text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 格式化为CSV格式
    pub fn format_csv(
        text_boxes: Vec<Vec<[f32; 2]>>,
        recognition_results: Vec<(String, f32)>,
        min_confidence: f32,
    ) -> String {
        let mut csv = String::from("text,confidence,x1,y1,x2,y2,x3,y3,x4,y4\n");
        
        let min_len = text_boxes.len().min(recognition_results.len());
        
        for i in 0..min_len {
            let (text, confidence) = &recognition_results[i];
            
            if *confidence >= min_confidence && !text.trim().is_empty() {
                let bbox = &text_boxes[i];
                let escaped_text = text.replace("\"", "\"\""); // CSV转义
                
                csv.push_str(&format!(
                    "\"{}\",{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                    escaped_text, confidence,
                    bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],
                    bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]
                ));
            }
        }
        
        csv
    }

    /// 计算平均置信度
    pub fn calculate_average_confidence(results: &[OcrTextResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = results.iter().map(|r| r.confidence).sum();
        sum / results.len() as f32
    }

    /// 统计识别字符数
    pub fn count_characters(results: &[OcrTextResult]) -> usize {
        results.iter().map(|r| r.text.chars().count()).sum()
    }

    /// 过滤重叠的文本框
    pub fn filter_overlapping_boxes(mut results: Vec<OcrTextResult>) -> Vec<OcrTextResult> {
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut filtered: Vec<OcrTextResult> = Vec::new();
        
        for result in results {
            let mut is_overlapping = false;
            
            for existing in &filtered {
                if Self::calculate_iou(&result.bounding_box, &existing.bounding_box) > 0.5 {
                    is_overlapping = true;
                    break;
                }
            }
            
            if !is_overlapping {
                filtered.push(result);
            }
        }
        
        filtered
    }

    /// 从OCR结果中提取纯文本
    pub fn extract_text_only(result: &OcrResult) -> String {
        result.results
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 从OCR结果转换为CSV格式
    pub fn result_to_csv(result: &OcrResult) -> String {
        let mut csv = String::from("text,confidence,x1,y1,x2,y2,x3,y3,x4,y4\n");
        
        for item in &result.results {
            let escaped_text = item.text.replace("\"", "\"\"");
            let bbox = &item.bounding_box;
            
            csv.push_str(&format!(
                "\"{}\",{:.4},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                escaped_text, item.confidence,
                bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],
                bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]
            ));
        }
        
        csv
    }

    /// 计算两个边界框的IoU（交并比）
    fn calculate_iou(box1: &[[f32; 2]], box2: &[[f32; 2]]) -> f32 {
        if box1.len() != 4 || box2.len() != 4 {
            return 0.0;
        }

        // 简化的矩形IoU计算（假设边界框是轴对齐的）
        let box1_min_x = box1.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let box1_max_x = box1.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let box1_min_y = box1.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let box1_max_y = box1.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);

        let box2_min_x = box2.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let box2_max_x = box2.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let box2_min_y = box2.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let box2_max_y = box2.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);

        // 计算交集
        let intersection_min_x = box1_min_x.max(box2_min_x);
        let intersection_max_x = box1_max_x.min(box2_max_x);
        let intersection_min_y = box1_min_y.max(box2_min_y);
        let intersection_max_y = box1_max_y.min(box2_max_y);

        if intersection_min_x >= intersection_max_x || intersection_min_y >= intersection_max_y {
            return 0.0;
        }

        let intersection_area = (intersection_max_x - intersection_min_x) * (intersection_max_y - intersection_min_y);
        
        // 计算并集
        let box1_area = (box1_max_x - box1_min_x) * (box1_max_y - box1_min_y);
        let box2_area = (box2_max_x - box2_min_x) * (box2_max_y - box2_min_y);
        let union_area = box1_area + box2_area - intersection_area;

        if union_area <= 0.0 {
            return 0.0;
        }

        intersection_area / union_area
    }
}

/// 简化版OCR结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleOcrResult {
    pub processing_time: f32,
    pub text_count: usize,
    pub texts: Vec<String>,
}