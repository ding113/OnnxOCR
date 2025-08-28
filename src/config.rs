use anyhow::Result;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    /// 服务器绑定地址
    pub bind_addr: String,
    
    /// 模型文件目录
    pub models_dir: PathBuf,
    
    /// 工作线程数量
    pub workers: usize,
    
    /// 开发模式
    pub dev_mode: bool,
    
    /// ONNX Runtime配置
    pub onnx_config: OnnxConfig,
    
    /// 服务器配置
    pub server_config: ServerConfig,
}

#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// CPU线程数
    pub intra_threads: usize,
    
    /// 优化级别
    pub optimization_level: i32,
    
    /// 启用图优化
    pub enable_optimization: bool,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// 请求超时时间（秒）
    pub request_timeout: u64,
    
    /// 最大请求体大小（字节）
    pub max_request_size: usize,
    
    /// 最大并发连接数
    pub max_connections: usize,
}

impl Config {
    pub fn new(
        bind_addr: String, 
        models_dir: String, 
        workers: Option<usize>,
        dev_mode: bool,
    ) -> Result<Self> {
        let cpu_cores = num_cpus::get();
        let workers = workers.unwrap_or(cpu_cores);
        
        let onnx_config = OnnxConfig {
            intra_threads: (cpu_cores * 3 / 4).max(1), // 使用75%的CPU核心
            optimization_level: 3, // 最高优化级别
            enable_optimization: true,
        };
        
        let server_config = ServerConfig {
            request_timeout: if dev_mode { 300 } else { 60 }, // 开发模式更长超时
            max_request_size: 50 * 1024 * 1024, // 50MB
            max_connections: if dev_mode { 10 } else { 1000 },
        };

        Ok(Self {
            bind_addr,
            models_dir: PathBuf::from(models_dir),
            workers,
            dev_mode,
            onnx_config,
            server_config,
        })
    }
    
    /// 获取检测模型路径
    pub fn det_model_path(&self) -> PathBuf {
        self.models_dir.join("ppocrv5/det/det.onnx")
    }
    
    /// 获取识别模型路径
    pub fn rec_model_path(&self) -> PathBuf {
        self.models_dir.join("ppocrv5/rec/rec.onnx")
    }
    
    /// 获取分类模型路径
    pub fn cls_model_path(&self) -> PathBuf {
        self.models_dir.join("ppocrv5/cls/cls.onnx")
    }
    
    /// 获取字典文件路径
    pub fn dict_path(&self) -> PathBuf {
        self.models_dir.join("ppocrv5/ppocrv5_dict.txt")
    }
}