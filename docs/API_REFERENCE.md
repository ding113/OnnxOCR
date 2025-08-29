# OnnxOCR API 参考文档

## 概览

OnnxOCR 提供基于FastAPI的高性能OCR服务，支持两套API接口：
- **v1接口**：与原Flask服务100%兼容的base64接口
- **v2接口**：增强功能接口，支持multipart上传、多文件处理、多种输出格式

## 快速开始

### 自动生成文档
FastAPI提供交互式API文档：
- **Swagger UI**: http://localhost:5005/docs
- **ReDoc**: http://localhost:5005/redoc

### 基础信息
- **服务地址**: http://localhost:5005
- **内容类型**: v1使用`application/json`，v2使用`multipart/form-data`
- **支持文件**: 图片格式(JPG/PNG/BMP)，PDF(单页)
- **最大上传**: 50MB（可通过环境变量调整）

---

## v1兼容接口

### POST /ocr
与原Flask服务完全兼容的OCR接口，使用base64编码图像输入。

**请求格式**：
```http
POST /ocr
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

**请求参数**：
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| image | string | 是 | base64编码的图像数据 |

**成功响应**：
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "processing_time": 0.456,
  "results": [
    {
      "text": "识别的文本",
      "confidence": 0.9999,
      "bounding_box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    }
  ]
}
```

**错误响应**：
```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": "错误描述"
}
```

**使用示例**：
```python
import requests
import base64

# 读取图片并编码
with open('test.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

# 发送请求
response = requests.post('http://localhost:5005/ocr', 
                        json={'image': img_data})
result = response.json()

print(f"识别耗时: {result['processing_time']}秒")
for item in result['results']:
    print(f"文本: {item['text']}, 置信度: {item['confidence']}")
```

---

## v2增强接口

### POST /api/v2/ocr
功能增强的OCR接口，支持multipart文件上传和多种配置选项。

**请求格式**：
```http
POST /api/v2/ocr
Content-Type: multipart/form-data

# 表单字段：
- file: 单个文件 (与files互斥)
- files: 多个文件 (与file互斥)  
- model_name: 模型选择
- conf_threshold: 置信度阈值
- output_format: 输出格式
- bbox: 是否返回边界框
- return_image: 是否返回处理后图像
```

**请求参数**：
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| file | File | 否* | - | 单个文件上传 |
| files | List[File] | 否* | - | 多个文件上传 |
| model_name | string | 否 | PP-OCRv5 | 模型选择：PP-OCRv5/PP-OCRv4/ch_ppocr_server_v2.0 |
| conf_threshold | float | 否 | 0.5 | 置信度阈值 (0.0-1.0) |
| output_format | string | 否 | json | 输出格式：json/text/tsv/hocr |
| bbox | boolean | 否 | true | 是否返回文本边界框坐标 |
| return_image | boolean | 否 | false | 是否返回带标注的处理图像 |

*注：file和files必须提供其中一个

**单文件响应** (output_format=json):
```json
{
  "processing_time": 0.456,
  "results": [
    {
      "text": "识别的文本",
      "confidence": 0.9999,
      "bounding_box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    }
  ],
  "preview_image": null
}
```

**多文件响应** (output_format=text):
```json
{
  "processing_time": 1.234,
  "items": [
    {
      "filename": "image1.jpg",
      "text": "识别的文本内容"
    },
    {
      "filename": "image2.jpg", 
      "text": "识别的文本内容"
    }
  ],
  "zip_url": "/download/20241229_120000"
}
```

**其他输出格式响应**:
```json
// output_format=text
{"text": "识别的文本内容", "processing_time": 0.456}

// output_format=tsv  
{"tsv": "text\tconfidence\tbbox\n识别文本\t0.99\t[[...]]", "processing_time": 0.456}

// output_format=hocr
{"hocr": "<?xml version=\"1.0\"?>...", "processing_time": 0.456}
```

**使用示例**：
```python
import requests

# 单文件上传
with open('test.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'model_name': 'PP-OCRv5',
        'conf_threshold': 0.6,
        'output_format': 'json',
        'bbox': 'true'
    }
    response = requests.post('http://localhost:5005/api/v2/ocr',
                           files=files, data=data)
    result = response.json()

# 多文件上传
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb'))
]
data = {'output_format': 'text'}
response = requests.post('http://localhost:5005/api/v2/ocr',
                       files=files, data=data)
```

```bash
# curl示例
curl -X POST http://localhost:5005/api/v2/ocr \
  -F "file=@test.jpg" \
  -F "model_name=PP-OCRv5" \
  -F "conf_threshold=0.6" \
  -F "output_format=json"
```

### GET /api/v2/tasks/{task_id}
查询异步任务状态（当前版本为同步处理，预留接口）。

**响应格式**：
```json
{
  "status": "done",
  "progress": 100.0,
  "result": {...},
  "error": null
}
```

### GET /api/v2/healthz
基本健康检查，验证服务是否运行。

**响应格式**：
```json
{
  "status": "ok",
  "timestamp": 1703836800.0
}
```

### GET /api/v2/readyz
就绪检查，验证模型是否加载完成。

**成功响应**：
```json
{
  "status": "ready",
  "timestamp": 1703836800.0
}
```

**未就绪响应**：
```http
HTTP/1.1 503 Service Unavailable

{
  "status": "not ready",
  "message": "Models not loaded"
}
```

---

## Web UI

### GET /
访问Web界面，提供图形化文件上传和OCR处理功能。

### GET /download/{timestamp}
下载OCR结果的ZIP压缩包。

**参数**：
- `timestamp`: 处理会话的时间戳标识

**响应**：返回ZIP文件下载。

---

## 错误处理

### HTTP状态码
| 状态码 | 描述 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 413 | 文件过大 |
| 415 | 不支持的文件类型 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用（模型未加载） |

### 错误码参考
| 错误码 | HTTP状态 | 描述 |
|--------|----------|------|
| VALIDATION_ERROR | 400 | 请求参数验证失败 |
| UNSUPPORTED_MEDIA_TYPE | 415 | 不支持的文件类型 |
| FILE_TOO_LARGE | 413 | 文件大小超出限制 |
| NOT_FOUND | 404 | 资源未找到 |
| INFERENCE_ERROR | 500 | 模型推理失败 |
| INTERNAL_ERROR | 500 | 内部服务器错误 |

### 错误响应格式
```json
{
  "error": "Human readable error message",
  "code": "ERROR_CODE"
}
```

---

## 环境配置

### 环境变量
| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 5005 | 服务监听端口 |
| WORKERS | auto | Gunicorn工作进程数 |
| THREADS | 2 | 每个进程的线程数 |
| DEFAULT_MODEL | PP-OCRv5 | 默认使用的OCR模型 |
| MODEL_POOL_SIZE | 1 | 模型实例池大小 |
| MODEL_CONCURRENCY | 1 | 最大并发推理数 |
| MAX_UPLOAD_MB | 50 | 最大上传文件大小(MB) |
| LOG_LEVEL | INFO | 日志等级：DEBUG/INFO/WARNING/ERROR |
| USE_GPU | false | 是否使用GPU加速 |
| WARMUP | true | 是否启动时预热模型 |

### Docker部署
```bash
# 构建镜像
docker build -t onnxocr .

# 运行容器
docker run -p 5005:5005 \
  -e MAX_UPLOAD_MB=100 \
  -e LOG_LEVEL=INFO \
  onnxocr
```

---

## 性能优化

### 推荐配置
- **4C8G服务器**: `WORKERS=4, THREADS=2, MODEL_POOL_SIZE=2`
- **启用GPU**: `USE_GPU=true` (需要GPU支持)
- **高并发**: 适当增加`MODEL_CONCURRENCY`值

### 性能指标
- **单图处理**: ~200-500ms (CPU) / ~100-200ms (GPU)
- **并发处理**: 支持多进程+多线程混合模式
- **内存占用**: 每个模型实例约500MB-1GB

---

## 版本历史

### v2.0.0 (当前版本)
- 重构为FastAPI架构
- 支持v1/v2双接口
- 新增多文件批量处理
- 支持多种输出格式
- 改进错误处理和日志

### v1.x (Flask版本)
- 基础OCR功能
- 单文件处理
- base64输入输出

---

## 技术支持

- **GitHub**: [项目地址]
- **文档更新**: 基于实际API实现 v2.0.0
- **最后更新**: 2025年1月