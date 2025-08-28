use crate::utils::error::OcrError;
use crate::{Config, Result};
use ndarray::{Array3, Axis, s};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
    inputs,
};
use parking_lot::Mutex;
use std::sync::Arc;

pub struct Detector {
    session: Arc<Mutex<Session>>,
    input_size: (usize, usize), // (height, width)
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f32,
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

        let session = Arc::new(Mutex::new(session));
        
        Ok(Self {
            session,
            input_size: (960, 960), // PPOCRv5 默认输入尺寸
            thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.5,
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
            // 立即提取数据，释放对session的借用
            outputs["sigmoid_0.tmp_0"].try_extract_array::<f32>()?.into_owned()
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

    /// 后处理：从概率图提取文字框
    fn postprocess(
        &self, 
        prediction: &ndarray::ArrayViewD<f32>,
        scale_x: f32,
        scale_y: f32,
    ) -> Result<Vec<Vec<[f32; 2]>>> {
        let pred_shape = prediction.shape();
        let (batch_size, height, width) = (pred_shape[0], pred_shape[1], pred_shape[2]);
        
        if batch_size != 1 {
            return Err(OcrError::Inference(
                "Expected batch size 1 for detection".to_string()
            ));
        }

        let pred_map = prediction.slice(s![0, .., ..]);
        let mut boxes = Vec::new();

        // 简化的文字框提取逻辑
        // 在实际实现中应该使用更复杂的轮廓检测和多边形近似
        for y in 1..height-1 {
            for x in 1..width-1 {
                let score = pred_map[[y, x]];
                if score > self.thresh {
                    // 寻找连通区域并拟合四边形
                    // 这里使用简化版本
                    let box_coords = vec![
                        [x as f32 * scale_x, y as f32 * scale_y],
                        [(x + 10) as f32 * scale_x, y as f32 * scale_y],
                        [(x + 10) as f32 * scale_x, (y + 10) as f32 * scale_y],
                        [x as f32 * scale_x, (y + 10) as f32 * scale_y],
                    ];
                    boxes.push(box_coords);
                }
            }
        }

        // 过滤重叠的框（简化版NMS）
        self.filter_boxes(boxes)
    }

    /// 简单的框过滤（生产环境需要更复杂的NMS算法）
    fn filter_boxes(&self, mut boxes: Vec<Vec<[f32; 2]>>) -> Result<Vec<Vec<[f32; 2]>>> {
        // 按面积排序，保留较大的框
        boxes.sort_by(|a, b| {
            let area_a = self.box_area(a);
            let area_b = self.box_area(b);
            area_b.partial_cmp(&area_a).unwrap()
        });

        // 限制数量以避免过多框
        boxes.truncate(100);
        Ok(boxes)
    }

    fn box_area(&self, box_coords: &[[f32; 2]]) -> f32 {
        if box_coords.len() != 4 {
            return 0.0;
        }
        
        let width = (box_coords[1][0] - box_coords[0][0]).abs();
        let height = (box_coords[2][1] - box_coords[1][1]).abs();
        width * height
    }
}