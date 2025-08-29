# OnnxOCR API 文档

## 概览

OnnxOCR 提供两套API接口：
- **v1接口**：保持与原Flask服务100%兼容
- **v2接口**：新增功能，支持多文件、多格式输出

## v1 兼容接口

### POST /ocr
与原版Flask服务完全兼容的OCR接口。

**请求格式**：
```http
POST /ocr
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

**响应格式**：
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

## v2 新接口

### POST /api/v2/ocr
功能增强的OCR接口，支持多文件上传和多种输出格式。

**请求格式**：
```http
POST /api/v2/ocr
Content-Type: multipart/form-data

# 表单字段：
- file: 单个文件 (可选)
- files: 多个文件 (可选)
- model_name: PP-OCRv5|PP-OCRv4|ch_ppocr_server_v2.0 (默认: PP-OCRv5)
- conf_threshold: 置信度阈值 0.0-1.0 (默认: 0.5)  
- output_format: json|text|tsv|hocr (默认: json)
- bbox: true|false (默认: true)
- return_image: true|false (默认: false)
```

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
基本健康检查。

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

## Web UI

### GET /
访问Web界面，提供文件上传和批量处理功能。

### GET /download/{timestamp}
下载处理结果的ZIP压缩包。

**响应**：返回ZIP文件下载。

## 错误码

| 错误码 | HTTP状态码 | 描述 |
|--------|------------|------|
| VALIDATION_ERROR | 400 | 请求参数验证失败 |
| UNSUPPORTED_MEDIA_TYPE | 415 | 不支持的文件类型 |
| FILE_TOO_LARGE | 413 | 文件大小超出限制 |
| NOT_FOUND | 404 | 资源未找到 |
| INFERENCE_ERROR | 500 | 模型推理失败 |
| INTERNAL_ERROR | 500 | 内部服务器错误 |

## 环境变量配置

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| HOST | 0.0.0.0 | 监听地址 |
| PORT | 5005 | 监听端口 |
| WORKERS | auto | Gunicorn工作进程数 |
| THREADS | 2 | 每个进程的线程数 |
| DEFAULT_MODEL | PP-OCRv5 | 默认使用的模型 |
| MODEL_POOL_SIZE | 1 | 模型实例池大小 |
| MODEL_CONCURRENCY | 1 | 最大并发推理数 |
| MAX_UPLOAD_MB | 50 | 最大上传文件大小(MB) |
| LOG_LEVEL | INFO | 日志等级 |
| USE_GPU | false | 是否使用GPU |
| WARMUP | true | 是否启动时预热模型 |

## 使用示例

### Python客户端示例
```python
import requests
import base64

# v1接口示例
with open('test.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5005/ocr', 
                        json={'image': img_data})
result = response.json()
print(f"识别耗时: {result['processing_time']}秒")
for item in result['results']:
    print(f"文本: {item['text']}, 置信度: {item['confidence']}")

# v2接口示例
with open('test.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'model_name': 'PP-OCRv5',
        'conf_threshold': 0.6,
        'output_format': 'json'
    }
    response = requests.post('http://localhost:5005/api/v2/ocr',
                           files=files, data=data)
    result = response.json()
```

### curl示例
```bash
# v1接口
curl -X POST http://localhost:5005/ocr \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w 0 test.jpg)'"}'

# v2接口
curl -X POST http://localhost:5005/api/v2/ocr \
  -F "file=@test.jpg" \
  -F "model_name=PP-OCRv5" \
  -F "conf_threshold=0.6"
```