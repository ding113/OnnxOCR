# ONNX OCR Rust服务

高性能的Rust OCR服务，支持双模式上传（JSON Base64 + 文件上传），专为解决NGINX缓冲区问题设计。

## ✨ 主要特性

- 🚀 **高性能**: Rust异步架构，CPU推理最大化利用
- 🔄 **双上传模式**: JSON Base64 + Multipart文件上传
- 🌊 **流式处理**: 避免大文件NGINX缓冲区问题  
- 🎯 **智能OCR**: PPOCRv5检测 + SVTR识别 + 角度分类
- 🐳 **容器化**: Docker一键部署，适配国内网络环境
- 🌐 **Web UI**: 现代化响应式界面

## 🚀 快速开始

### 前置要求

- Docker & Docker Compose
- 模型文件位于 `models/` 目录



### 手动部署

```bash
# 创建数据目录
mkdir -p data/{logs,results,uploads}

# 构建并启动
docker-compose build
docker-compose up -d

# 检查状态
curl http://localhost:5005/health
```

## 📡 API接口

### 1. JSON Base64上传 (向后兼容)

```bash
curl -X POST http://localhost:5005/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "use_angle_cls": true,
    "min_confidence": 0.5,
    "output_format": "json"
  }'
```

### 2. 文件上传 (推荐，支持大文件)

```bash
curl -X POST http://localhost:5005/ocr/upload \
  -F "file=@image.jpg" \
  -F "use_angle_cls=true" \
  -F "min_confidence=0.5" \
  -F "output_format=json"
```

### 3. 系统接口

- `GET /` - Web UI界面
- `GET /health` - 健康检查  
- `GET /api/info` - 服务信息

## 📊 响应格式

```json
{
  "success": true,
  "data": {
    "processing_time": 0.456,
    "results": [
      {
        "text": "识别的文本",
        "confidence": 0.9999,
        "bounding_box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
      }
    ]
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid"
}
```

## 🔧 配置选项

### 环境变量

```bash
RUST_LOG=info          # 日志级别
RUST_BACKTRACE=1       # 错误堆栈
```

### 启动参数

```bash
onnx-ocr \
  --bind 0.0.0.0:5005 \
  --models-dir /path/to/models \
  --workers 4 \
  --log-level info
```

## 📁 项目结构

```
├── src/
│   ├── models/         # ONNX模型管理
│   ├── image/         # 图像处理
│   ├── ocr/           # OCR流水线
│   ├── web/           # Web服务
│   └── utils/         # 工具函数
├── templates/         # Web UI模板
├── Dockerfile         # Docker构建文件
├── docker-compose.yml # Docker编排
├── deploy.bat         # Windows部署脚本
└── deploy.sh          # Linux部署脚本
```

## 🤝 开发

### 本地开发

```bash
# 安装Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 构建项目
cargo build --release

# 运行服务
cargo run -- --bind 127.0.0.1:5005
```

### 代码检查

```bash
cargo check          # 检查编译错误
cargo clippy         # 代码规范检查
cargo test           # 运行测试
```

## 📄 许可证

MIT License