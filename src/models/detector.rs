use crate::utils::error::OcrError;
use crate::{Config, Result};
use ndarray::{Array3, Array2, Axis, s};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
    inputs,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::VecDeque;

pub struct Detector {
    session: Arc<Mutex<Session>>,
    output_name: String, // 动态发现的输出名称
    input_size: (usize, usize), // (height, width)
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
    max_candidates: usize,
    min_size: f32,
}

impl Detector {
    pub fn new(config: &Config) -> Result<Self> {
        let model_path = config.det_model_path();
        
        if !model_path.exists() {
            return Err(OcrError::ModelLoad(
                format!("Detection model not found: {}", model_path.display())
            ));
        }

        tracing::info!("Loading detection model from: {}", model_path.display());
        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.onnx_config.intra_threads)?
            .commit_from_file(&model_path)?;

        // 动态发现输出名称
        let output_name = if session.outputs.is_empty() {
            return Err(OcrError::ModelLoad(
                "Detection model has no outputs".to_string()
            ));
        } else {
            let output_name = session.outputs[0].name.clone();
            tracing::info!("Detection model output: '{}'", output_name);
            
            // 记录所有可用输出用于调试
            for (i, output) in session.outputs.iter().enumerate() {
                tracing::debug!("Detection output[{}]: '{}'", i, output.name);
            }
            
            output_name
        };

        let session = Arc::new(Mutex::new(session));
        
        Ok(Self {
            session,
            output_name,
            input_size: (960, 960), // PPOCRv5 默认输入尺寸
            thresh: 0.3,
            box_thresh: 0.7,
            unclip_ratio: 2.0,
            max_candidates: 1000,
            min_size: 3.0,
        })
    }

    /// 文字检测推理
    pub fn detect(&self, image: &Array3<f32>) -> Result<Vec<Vec<[f32; 2]>>> {
        // 预处理：调整图像大小并归一化
        let (resized_img, scale_x, scale_y) = self.preprocess(image)?;
        
        // 创建输入tensor
        let input_tensor = resized_img.insert_axis(Axis(0)); // 添加batch维度
        
        // 推理 - 立即提取数据避免生命周期冲突
        let input_tensor = Tensor::from_array(input_tensor)?;
        let prediction = {
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

        // 后处理：从概率图中提取文字框
        let boxes = self.postprocess(&prediction.view(), scale_x, scale_y)?;
        
        Ok(boxes)
    }

    /// 图像预处理
    fn preprocess(&self, image: &Array3<f32>) -> Result<(Array3<f32>, f32, f32)> {
        let (orig_h, orig_w) = (image.shape()[0], image.shape()[1]);
        let (target_h, target_w) = self.input_size;

        // 计算缩放比例，保持宽高比
        let scale = (target_h as f32 / orig_h as f32)
            .min(target_w as f32 / orig_w as f32);

        let new_h = (orig_h as f32 * scale) as usize;
        let new_w = (orig_w as f32 * scale) as usize;

        // 这里需要实现图像resize逻辑
        // 为了简化，我们暂时用简单的双线性插值
        let mut resized = Array3::<f32>::zeros((target_h, target_w, 3));
        
        // 简化的resize实现（生产环境应使用更高质量的插值）
        for y in 0..new_h {
            for x in 0..new_w {
                let src_y = (y as f32 / scale) as usize;
                let src_x = (x as f32 / scale) as usize;
                if src_y < orig_h && src_x < orig_w {
                    resized[[y, x, 0]] = image[[src_y, src_x, 0]];
                    resized[[y, x, 1]] = image[[src_y, src_x, 1]];  
                    resized[[y, x, 2]] = image[[src_y, src_x, 2]];
                }
            }
        }

        // 归一化：从[0,255] -> [0,1]并调整通道顺序为CHW
        let mut normalized = Array3::<f32>::zeros((3, target_h, target_w));
        for c in 0..3 {
            for h in 0..target_h {
                for w in 0..target_w {
                    normalized[[c, h, w]] = resized[[h, w, c]] / 255.0;
                }
            }
        }

        let scale_x = orig_w as f32 / new_w as f32;
        let scale_y = orig_h as f32 / new_h as f32;

        Ok((normalized, scale_x, scale_y))
    }

    /// 后处理：从概率图提取文字框 - 基于DBNet算法
    fn postprocess(
        &self, 
        prediction: &ndarray::ArrayViewD<f32>,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<Vec<Vec<[f32; 2]>>> {
        let pred_shape = prediction.shape();
        
        // 支持 3D 和 4D 张量
        let (batch_size, height, width, pred_map) = if pred_shape.len() == 3 {
            // 3D 张量: (batch, height, width)
            let (batch_size, height, width) = (pred_shape[0], pred_shape[1], pred_shape[2]);
            let pred_map = prediction.slice(s![0, .., ..]);
            (batch_size, height, width, pred_map)
        } else if pred_shape.len() == 4 {
            // 4D 张量: (batch, channels, height, width) - 取第一个通道
            let (batch_size, _channels, height, width) = (pred_shape[0], pred_shape[1], pred_shape[2], pred_shape[3]);
            let pred_map = prediction.slice(s![0, 0, .., ..]);
            (batch_size, height, width, pred_map)
        } else {
            return Err(OcrError::ModelCompatibility(format!(
                "Unsupported detection output shape: {:?}. Expected 3D (batch,height,width) or 4D (batch,channels,height,width)", 
                pred_shape
            )));
        };
        
        if batch_size != 1 {
            return Err(OcrError::Inference(
                "Expected batch size 1 for detection".to_string()
            ));
        }

        tracing::debug!("Detection prediction shape: {:?}, thresh: {}", pred_shape, self.thresh);

        // Step 1: 二值化 - 创建掩码
        let segmentation = self.create_segmentation(&pred_map);
        tracing::debug!("Created segmentation mask: {}x{}", height, width);

        // Step 2: 查找轮廓和文本框
        let boxes = self.boxes_from_bitmap(&pred_map, &segmentation, scale_x, scale_y)?;
        
        tracing::info!("Detected {} text boxes", boxes.len());
        Ok(boxes)
    }

    /// 创建二值化掩码
    fn create_segmentation(&self, pred_map: &ndarray::ArrayView2<f32>) -> Array2<bool> {
        let (height, width) = pred_map.dim();
        let mut segmentation = Array2::<bool>::default((height, width));
        
        for y in 0..height {
            for x in 0..width {
                segmentation[[y, x]] = pred_map[[y, x]] > self.thresh;
            }
        }
        
        segmentation
    }

    /// 从二值化图像中提取文本框 - 基于Python的boxes_from_bitmap
    fn boxes_from_bitmap(
        &self,
        pred_map: &ndarray::ArrayView2<f32>,
        segmentation: &Array2<bool>,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<Vec<Vec<[f32; 2]>>> {
        let (height, width) = segmentation.dim();
        
        // 找到所有连通组件
        let contours = self.find_contours(segmentation);
        tracing::debug!("Found {} contours", contours.len());
        
        let mut boxes = Vec::new();
        let mut scores = Vec::new();
        
        let num_contours = contours.len().min(self.max_candidates);
        
        for i in 0..num_contours {
            let contour = &contours[i];
            
            // 获取最小包围矩形
            if let Some((box_points, sside)) = self.get_mini_boxes(contour) {
                if sside < self.min_size {
                    continue;
                }
                
                // 计算框的评分
                let score = self.box_score_fast(pred_map, &box_points);
                if score < self.box_thresh {
                    continue;
                }
                
                // unclip操作 - 扩展文本框
                if let Some(unclipped_box) = self.unclip(&box_points, self.unclip_ratio) {
                    if let Some((final_box, final_sside)) = self.get_mini_boxes(&unclipped_box) {
                        if final_sside < self.min_size + 2.0 {
                            continue;
                        }
                        
                        // 缩放到原图坐标
                        let scaled_box = self.scale_box(&final_box, width, height, scale_x, scale_y);
                        boxes.push(scaled_box);
                        scores.push(score);
                    }
                }
            }
        }
        
        tracing::debug!("After filtering: {} boxes", boxes.len());
        Ok(boxes)
    }

    /// 简单的连通组件查找 - 类似cv2.findContours的功能
    fn find_contours(&self, segmentation: &Array2<bool>) -> Vec<Vec<[f32; 2]>> {
        let (height, width) = segmentation.dim();
        let mut visited = Array2::<bool>::default((height, width));
        let mut contours = Vec::new();
        
        for y in 0..height {
            for x in 0..width {
                if segmentation[[y, x]] && !visited[[y, x]] {
                    let contour = self.trace_contour(segmentation, &mut visited, x, y);
                    if contour.len() >= 4 {
                        contours.push(contour);
                    }
                }
            }
        }
        
        contours
    }

    /// 跟踪轮廓边界 - 8连通
    fn trace_contour(
        &self,
        segmentation: &Array2<bool>,
        visited: &mut Array2<bool>,
        start_x: usize,
        start_y: usize,
    ) -> Vec<[f32; 2]> {
        let (height, width) = segmentation.dim();
        let mut contour = Vec::new();
        let mut queue = VecDeque::new();
        
        queue.push_back((start_x, start_y));
        visited[[start_y, start_x]] = true;
        
        // 8方向偏移
        let directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ];
        
        while let Some((x, y)) = queue.pop_front() {
            contour.push([x as f32, y as f32]);
            
            for (dx, dy) in directions {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                
                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    
                    if segmentation[[ny, nx]] && !visited[[ny, nx]] {
                        visited[[ny, nx]] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
        
        contour
    }

    /// 获取最小外接矩形 - 类似cv2.minAreaRect + cv2.boxPoints
    fn get_mini_boxes(&self, contour: &[[f32; 2]]) -> Option<(Vec<[f32; 2]>, f32)> {
        if contour.len() < 4 {
            return None;
        }
        
        // 计算边界框
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        
        for point in contour {
            min_x = min_x.min(point[0]);
            max_x = max_x.max(point[0]);
            min_y = min_y.min(point[1]);
            max_y = max_y.max(point[1]);
        }
        
        let width = max_x - min_x;
        let height = max_y - min_y;
        let min_side = width.min(height);
        
        let box_points = vec![
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ];
        
        Some((box_points, min_side))
    }

    /// 快速框评分 - 类似Python的box_score_fast
    fn box_score_fast(&self, pred_map: &ndarray::ArrayView2<f32>, box_points: &[[f32; 2]]) -> f32 {
        let (height, width) = pred_map.dim();
        
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        
        for point in box_points {
            min_x = min_x.min(point[0]);
            max_x = max_x.max(point[0]);
            min_y = min_y.min(point[1]);
            max_y = max_y.max(point[1]);
        }
        
        let xmin = (min_x.floor() as usize).min(width - 1);
        let xmax = (max_x.ceil() as usize).min(width - 1);
        let ymin = (min_y.floor() as usize).min(height - 1);
        let ymax = (max_y.ceil() as usize).min(height - 1);
        
        if xmin >= xmax || ymin >= ymax {
            return 0.0;
        }
        
        // 计算区域内的平均分数
        let mut sum = 0.0;
        let mut count = 0;
        
        for y in ymin..=ymax {
            for x in xmin..=xmax {
                sum += pred_map[[y, x]];
                count += 1;
            }
        }
        
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// 简化的unclip操作 - 扩展多边形
    fn unclip(&self, box_points: &[[f32; 2]], unclip_ratio: f32) -> Option<Vec<[f32; 2]>> {
        if box_points.len() != 4 {
            return None;
        }
        
        // 计算多边形面积和周长
        let area = self.polygon_area(box_points);
        let perimeter = self.polygon_perimeter(box_points);
        
        if perimeter <= 0.0 {
            return None;
        }
        
        // 计算扩展距离
        let distance = area * unclip_ratio / perimeter;
        
        // 简化的扩展：向外扩展每个点
        let center_x = box_points.iter().map(|p| p[0]).sum::<f32>() / 4.0;
        let center_y = box_points.iter().map(|p| p[1]).sum::<f32>() / 4.0;
        
        let expanded_points: Vec<[f32; 2]> = box_points.iter().map(|point| {
            let dx = point[0] - center_x;
            let dy = point[1] - center_y;
            let len = (dx * dx + dy * dy).sqrt();
            
            if len > 0.0 {
                let scale = (len + distance) / len;
                [
                    center_x + dx * scale,
                    center_y + dy * scale,
                ]
            } else {
                *point
            }
        }).collect();
        
        Some(expanded_points)
    }

    /// 计算多边形面积
    fn polygon_area(&self, points: &[[f32; 2]]) -> f32 {
        if points.len() < 3 {
            return 0.0;
        }
        
        let mut area = 0.0;
        let n = points.len();
        
        for i in 0..n {
            let j = (i + 1) % n;
            area += points[i][0] * points[j][1];
            area -= points[j][0] * points[i][1];
        }
        
        area.abs() / 2.0
    }

    /// 计算多边形周长
    fn polygon_perimeter(&self, points: &[[f32; 2]]) -> f32 {
        if points.len() < 2 {
            return 0.0;
        }
        
        let mut perimeter = 0.0;
        let n = points.len();
        
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = points[j][0] - points[i][0];
            let dy = points[j][1] - points[i][1];
            perimeter += (dx * dx + dy * dy).sqrt();
        }
        
        perimeter
    }

    /// 将框坐标缩放到原图尺寸
    fn scale_box(
        &self,
        box_points: &[[f32; 2]],
        model_width: usize,
        model_height: usize,
        scale_x: f32,
        scale_y: f32,
    ) -> Vec<[f32; 2]> {
        box_points.iter().map(|point| {
            [
                (point[0] * scale_x).clamp(0.0, model_width as f32 * scale_x),
                (point[1] * scale_y).clamp(0.0, model_height as f32 * scale_y),
            ]
        }).collect()
    }
}