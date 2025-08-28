# Modern ONNX OCR API 文档

## 概述

Modern ONNX OCR Service 是一个基于 FastAPI 和 ONNX Runtime 的现代化 OCR 服务，支持中英文文字识别。

### 特性
- 🚀 **异步高性能处理** - 基于 FastAPI 异步框架
- 🧠 **智能模型切换** - 支持 PP-OCR v4、v5-Mobile、v5-Server
- 💾 **智能模型管理** - LRU 缓存，自动下载，内存优化
- 📊 **实时监控** - Prometheus 指标，结构化日志
- 🔒 **数据验证** - Pydantic 自动验证
- 🌐 **现代化界面** - 直观的 Web UI
- 📤 **现代化文件上传** - 支持 multipart/form-data，避免 NGINX 缓冲区溢出

### 版本信息
- **Service Version**: 2.0.0
- **Python**: 3.13
- **Framework**: FastAPI
- **Inference Engine**: ONNX Runtime 1.22.1

## ⚡ 推荐 API (V2) - 现代化接口

**强烈推荐使用 V2 API！** V2 API 使用现代的 `multipart/form-data` 文件上传方式，具有以下优势：
- ✅ **避免 NGINX 缓冲区溢出** - 不会因大图片导致代理服务器问题
- ✅ **内存效率更高** - 流式处理，不需要将整个图片载入内存
- ✅ **更好的兼容性** - 标准 HTTP 文件上传，所有客户端都支持
- ✅ **更快的传输** - 没有 Base64 编码开销，传输效率提升约 33%
- ✅ **更好的错误处理** - 内置文件格式验证和安全检查

### V2.1 单图 OCR 识别 (推荐)

**POST** `/v2/ocr`

使用现代 multipart/form-data 格式进行单张图片的 OCR 文字识别。

#### 请求参数 (Form Data)
- `file` **(必需)**: 图片文件，支持 JPG、PNG、BMP、WEBP 格式
- `model_version` **(可选)**: 模型版本，默认 `v5-server`
- `det` **(可选)**: 是否启用文本检测，默认 `true`
- `rec` **(可选)**: 是否启用文本识别，默认 `true`
- `cls` **(可选)**: 是否启用角度分类，默认 `true`
- `drop_score` **(可选)**: 置信度阈值，默认 `0.5`

#### 响应示例
```json
{
  "success": true,
  "results": [
    {
      "text": "示例文本",
      "confidence": 0.9856,
      "bbox": [100, 50, 200, 80],
      "angle": 0
    }
  ],
  "model_version": "v5-server",
  "image_info": {
    "width": 800,
    "height": 600,
    "channels": 3,
    "format": "JPEG",
    "file_size": 245760
  },
  "num_detected": 1,
  "process_time": 156.78,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### V2.2 批量 OCR 识别 (推荐)

**POST** `/v2/batch`

使用现代文件上传格式进行批量图片的 OCR 识别。

#### 请求参数 (Form Data)
- `files` **(必需)**: 多个图片文件
- `model_version` **(可选)**: 模型版本，默认 `v5-server`
- `det` **(可选)**: 是否启用文本检测，默认 `true`
- `rec` **(可选)**: 是否启用文本识别，默认 `true`
- `cls` **(可选)**: 是否启用角度分类，默认 `true`
- `drop_score` **(可选)**: 置信度阈值，默认 `0.5`

#### 响应示例
```json
{
  "success": true,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "results": [...],
      "process_time": 156.78
    },
    {
      "filename": "image2.png", 
      "success": true,
      "results": [...],
      "process_time": 142.33
    }
  ],
  "batch_summary": {
    "total_files": 2,
    "successful": 2,
    "failed": 0,
    "total_time": 299.11,
    "average_time": 149.56
  },
  "model_version": "v5-server",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### V2 API 使用示例

#### Python 客户端示例 (推荐)

```python
import requests
from pathlib import Path

def ocr_v2_recognize(image_path: str, model_version: str = "v5-server"):
    """使用 V2 API 进行 OCR 识别"""
    url = "http://localhost:5005/v2/ocr"
    
    # 准备文件和参数
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'model_version': model_version,
            'det': True,
            'rec': True,
            'cls': True,
            'drop_score': 0.5
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print(f"使用模型: {result['model_version']}")
            print(f"图片信息: {result['image_info']['width']}x{result['image_info']['height']}")
            print(f"处理时间: {result['process_time']:.2f}ms")
            print("-" * 50)
            
            for item in result["results"]:
                print(f"文本: {item['text']}")
                print(f"置信度: {item['confidence']:.4f}")
                print(f"位置: {item['bbox']}")
                print("-" * 40)
        else:
            print(f"处理失败: {result.get('error')}")
    else:
        print(f"请求失败: {response.status_code}")

def ocr_v2_batch(image_paths: list, model_version: str = "v5-server"):
    """使用 V2 API 进行批量 OCR 识别"""
    url = "http://localhost:5005/v2/batch"
    
    # 准备多个文件
    files = []
    for path in image_paths:
        files.append(('files', open(path, 'rb')))
    
    data = {
        'model_version': model_version,
        'det': True,
        'rec': True,
        'cls': True
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print(f"批量处理完成: {result['batch_summary']}")
                for item in result["results"]:
                    if item["success"]:
                        print(f"✅ {item['filename']}: {len(item['results'])} 个文本")
                    else:
                        print(f"❌ {item['filename']}: 处理失败")
        else:
            print(f"批量请求失败: {response.status_code}")
    finally:
        # 关闭所有文件
        for _, f in files:
            f.close()

# 使用示例
if __name__ == "__main__":
    # 单图识别
    ocr_v2_recognize("test_image.jpg")
    
    # 批量识别
    ocr_v2_batch(["image1.jpg", "image2.png", "image3.bmp"])
```

#### JavaScript 客户端示例 (推荐)

```javascript
class OCRV2Client {
    constructor(baseURL = 'http://localhost:5005') {
        this.baseURL = baseURL;
    }
    
    async recognizeSingle(imageFile, options = {}) {
        const {
            modelVersion = 'v5-server',
            det = true,
            rec = true,
            cls = true,
            dropScore = 0.5
        } = options;
        
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('model_version', modelVersion);
        formData.append('det', det);
        formData.append('rec', rec);
        formData.append('cls', cls);
        formData.append('drop_score', dropScore);
        
        try {
            const response = await fetch(`${this.baseURL}/v2/ocr`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                return result;
            } else {
                throw new Error(result.error || '处理失败');
            }
        } catch (error) {
            console.error('V2 OCR recognition failed:', error);
            throw error;
        }
    }
    
    async recognizeBatch(imageFiles, options = {}) {
        const {
            modelVersion = 'v5-server',
            det = true,
            rec = true,
            cls = true,
            dropScore = 0.5
        } = options;
        
        const formData = new FormData();
        
        // 添加多个文件
        for (const file of imageFiles) {
            formData.append('files', file);
        }
        
        formData.append('model_version', modelVersion);
        formData.append('det', det);
        formData.append('rec', rec);
        formData.append('cls', cls);
        formData.append('drop_score', dropScore);
        
        try {
            const response = await fetch(`${this.baseURL}/v2/batch`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                return result;
            } else {
                throw new Error(result.error || '批量处理失败');
            }
        } catch (error) {
            console.error('V2 Batch OCR failed:', error);
            throw error;
        }
    }
}

// 使用示例
const ocrClient = new OCRV2Client();

document.getElementById('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        try {
            const result = await ocrClient.recognizeSingle(file, {
                modelVersion: 'v5-server'
            });
            
            console.log(`处理时间: ${result.process_time}ms`);
            console.log(`检测到 ${result.num_detected} 个文本区域`);
            
            result.results.forEach(item => {
                console.log(`文本: ${item.text}`);
                console.log(`置信度: ${item.confidence.toFixed(4)}`);
                console.log(`位置: [${item.bbox.join(', ')}]`);
            });
        } catch (error) {
            console.error('V2 OCR 处理失败:', error);
        }
    }
});
```

#### cURL 示例 (推荐)

```bash
# 单图 OCR 识别
curl -X POST "http://localhost:5005/v2/ocr" \
  -F "file=@test_image.jpg" \
  -F "model_version=v5-server" \
  -F "det=true" \
  -F "rec=true" \
  -F "cls=true" \
  -F "drop_score=0.5"

# 批量 OCR 识别
curl -X POST "http://localhost:5005/v2/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.png" \
  -F "files=@image3.bmp" \
  -F "model_version=v5-server" \
  -F "det=true" \
  -F "rec=true" \
  -F "cls=true"
```

---

## 🏛️ 传统 API (V1) - 兼容性接口

> ⚠️ **注意**: 以下为传统 Base64 API，仅为向后兼容而保留。强烈建议新项目使用上述 V2 API。
> 
> **传统 API 的局限性**:
> - ❌ **NGINX 缓冲区风险** - 大图片可能导致代理服务器问题
> - ❌ **内存效率低** - Base64 编码增加约 33% 的数据量
> - ❌ **传输效率低** - 编码/解码增加 CPU 开销
> - ❌ **兼容性问题** - 某些代理服务器可能有大 payload 限制

## 服务配置

### 基础信息
- **Base URL**: `http://localhost:5005`
- **Content-Type**: `application/json`
- **Timeout**: 30秒（建议）

### 支持的模型版本
| 模型版本 | 描述 | 适用场景 |
|---------|------|----------|
| `v4` | PP-OCR v4 模型 | 通用场景，兼容性强 |
| `v5` | PP-OCR v5 Mobile | 移动端优化，速度快 |
| `v5-server` | PP-OCR v5 Server | 服务器优化，精度高 |

## API 端点

### 1. 健康检查

**GET** `/health`

检查服务健康状态和模型加载状态。

#### 响应示例
```json
{
  "status": "healthy",
  "model_loaded": true,
  "service": "Modern ONNX OCR Service",
  "version": "2.0.0",
  "python_version": "3.13",
  "timestamp": 1703686800.123
}
```

#### 响应字段
- `status`: 服务状态 (`healthy` | `unhealthy`)
- `model_loaded`: 模型是否已加载
- `service`: 服务名称
- `version`: 服务版本
- `python_version`: Python 版本
- `timestamp`: 检查时间戳

---

### 2. 服务信息

**GET** `/info`

获取详细的服务信息和功能特性。

#### 响应示例
```json
{
  "service": "Modern ONNX OCR Service",
  "version": "2.0.0",
  "python_version": "3.13",
  "framework": "FastAPI",
  "inference_engine": "ONNX Runtime 1.22.1",
  "model_type": "PP-OCR v5",
  "model_ready": true,
  "features": [
    "异步高性能处理",
    "CPU多核优化",
    "实时性能监控",
    "自动API文档",
    "数据验证",
    "结构化日志"
  ],
  "endpoints": {
    "v2_ocr": "/v2/ocr - 单图OCR识别 (推荐现代API)",
    "v2_batch": "/v2/batch - 批量文件处理 (推荐现代API)",
    "ocr": "/ocr - 单图OCR识别 (传统Base64 API)",
    "batch": "/ocr/batch - 批量文件处理 (传统Base64 API)",
    "models_available": "/models/available - 获取可用模型版本",
    "models_info": "/models/info - 获取模型详细信息",
    "models_switch": "/models/switch - 切换默认模型版本",
    "webui": "/webui - 现代Web界面",
    "docs": "/docs - 交互式API文档",
    "metrics": "/metrics - Prometheus指标",
    "health": "/health - 健康检查"
  }
}
```

---

### 3. OCR 识别

**POST** `/ocr`

对单张图片进行 OCR 文字识别。

#### 请求参数
```json
{
  "image": "base64_encoded_image_data",
  "model_version": "v5-server",
  "det": true,
  "rec": true,
  "cls": true
}
```

#### 请求字段说明
- `image` **(必需)**: Base64 编码的图片数据（不包含数据URI前缀）
- `model_version` **(可选)**: 模型版本，默认 `v5-server`
- `det` **(可选)**: 是否启用文本检测，默认 `true`
- `rec` **(可选)**: 是否启用文本识别，默认 `true`
- `cls` **(可选)**: 是否启用角度分类，默认 `true`

#### 响应示例
```json
{
  "success": true,
  "results": [
    {
      "text": "示例文本",
      "confidence": 0.9856,
      "bbox": [100, 50, 200, 80],
      "angle": 0
    },
    {
      "text": "另一行文本",
      "confidence": 0.9234,
      "bbox": [100, 90, 250, 120],
      "angle": 0
    }
  ],
  "metadata": {
    "model_version": "v5-server",
    "process_time": 156.78,
    "num_detected": 2,
    "image_shape": [400, 600, 3],
    "performance": {
      "detection_time": 45.2,
      "recognition_time": 89.1,
      "classification_time": 12.3,
      "total_time": 156.78
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 响应字段说明
- `success`: 处理是否成功
- `results`: 识别结果数组
  - `text`: 识别的文本内容
  - `confidence`: 置信度 (0-1)
  - `bbox`: 边界框坐标 [x1, y1, x2, y2]
  - `angle`: 文本角度
- `metadata`: 处理元数据
  - `model_version`: 使用的模型版本
  - `process_time`: 总处理时间(ms)
  - `num_detected`: 检测到的文本区域数量
  - `image_shape`: 图片尺寸
  - `performance`: 各阶段耗时详情

#### 错误响应
```json
{
  "success": false,
  "error": "Invalid image format",
  "code": "INVALID_IMAGE",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

### 4. 批量 OCR 识别

**POST** `/ocr/batch`

批量处理多张图片的 OCR 识别。

#### 请求参数
```json
{
  "images": [
    "base64_encoded_image_1",
    "base64_encoded_image_2"
  ],
  "model_version": "v5-server",
  "det": true,
  "rec": true,
  "cls": true
}
```

#### 请求字段说明
- `images` **(必需)**: Base64 编码的图片数据数组
- 其他参数同单张图片识别

#### 响应示例
```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "results": [...],
      "metadata": {...}
    },
    {
      "success": false,
      "error": "Processing failed",
      "metadata": {...}
    }
  ],
  "batch_metadata": {
    "total_images": 2,
    "successful": 1,
    "failed": 1,
    "total_time": 234.56,
    "avg_time_per_image": 117.28
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

### 5. 获取可用模型

**GET** `/models/available`

获取系统中可用的模型版本列表。

#### 响应示例
```json
{
  "success": true,
  "available_models": ["v4", "v5", "v5-server"],
  "default_model": "v5-server"
}
```

---

### 6. 获取模型信息

**GET** `/models/info`

获取当前加载模型的详细信息。

#### 响应示例
```json
{
  "success": true,
  "current_model": "v5-server",
  "model_info": {
    "v4": {
      "name": "PP-OCR v4",
      "det_model": "/path/to/ppocrv4/det/det.onnx",
      "rec_model": "/path/to/ppocrv4/rec/rec.onnx",
      "cls_model": "/path/to/ppocrv4/cls/cls.onnx",
      "loaded": true,
      "load_time": "2024-01-01T12:00:00Z"
    },
    "v5": {
      "name": "PP-OCR v5 Mobile",
      "det_model": "/path/to/ppocrv5/det/det.onnx",
      "rec_model": "/path/to/ppocrv5/rec/rec.onnx",
      "cls_model": "/path/to/ppocrv5/cls/cls.onnx",
      "loaded": false,
      "load_time": null
    },
    "v5-server": {
      "name": "PP-OCR v5 Server",
      "det_model": "/path/to/ppocrv5-server/det.onnx",
      "rec_model": "/path/to/ppocrv5-server/rec.onnx",
      "cls_model": "/path/to/ppocrv5/cls/cls.onnx",
      "loaded": true,
      "load_time": "2024-01-01T12:00:00Z"
    }
  },
  "cache_info": {
    "cache_size": 2,
    "max_cache_size": 3,
    "cache_hit_rate": 0.85
  }
}
```

---

### 7. 切换模型版本

**POST** `/models/switch`

切换默认使用的模型版本。

#### 请求参数
```json
{
  "model_version": "v5-server"
}
```

#### 响应示例
```json
{
  "success": true,
  "message": "Model switched successfully",
  "previous_model": "v5",
  "current_model": "v5-server",
  "switch_time": 1.234
}
```

---

### 8. Prometheus 指标

**GET** `/metrics`

获取 Prometheus 格式的性能指标。

#### 响应示例
```
# HELP ocr_requests_total Total OCR requests
# TYPE ocr_requests_total counter
ocr_requests_total{method="POST",endpoint="/ocr",status="200",model_version="v5-server"} 150
ocr_requests_total{method="POST",endpoint="/ocr",status="400",model_version="v5-server"} 5

# HELP ocr_request_duration_seconds OCR request duration
# TYPE ocr_request_duration_seconds histogram
ocr_request_duration_seconds_bucket{method="POST",endpoint="/ocr",model_version="v5-server",le="0.1"} 10
ocr_request_duration_seconds_bucket{method="POST",endpoint="/ocr",model_version="v5-server",le="0.5"} 100
ocr_request_duration_seconds_bucket{method="POST",endpoint="/ocr",model_version="v5-server",le="1.0"} 140
ocr_request_duration_seconds_bucket{method="POST",endpoint="/ocr",model_version="v5-server",le="+Inf"} 155

# HELP ocr_model_load_time_seconds Time to load models
# TYPE ocr_model_load_time_seconds histogram
ocr_model_load_time_seconds_bucket{model_version="v5-server",le="1.0"} 5
ocr_model_load_time_seconds_bucket{model_version="v5-server",le="2.0"} 8
ocr_model_load_time_seconds_bucket{model_version="v5-server",le="+Inf"} 10
```

---

### 9. Web 界面

**GET** `/webui`

访问现代化的 Web 用户界面。

提供完整的图形化界面，支持：
- 图片拖拽上传
- 实时参数调整
- 结果可视化展示
- 性能监控面板
- 结果导出功能

---

## 错误代码

| 错误代码 | HTTP状态 | 描述 |
|---------|---------|------|
| `INVALID_IMAGE` | 400 | 无效的图片格式 |
| `IMAGE_TOO_LARGE` | 400 | 图片文件过大 |
| `MISSING_PARAMETERS` | 400 | 缺少必需参数 |
| `MODEL_NOT_FOUND` | 404 | 指定的模型版本不存在 |
| `MODEL_LOAD_ERROR` | 500 | 模型加载失败 |
| `PROCESSING_ERROR` | 500 | 图片处理失败 |
| `INTERNAL_ERROR` | 500 | 内部服务器错误 |

## 使用示例

### 🚀 推荐使用 V2 API

强烈建议所有新项目使用上述的 V2 API 示例。V2 API 提供了更好的性能、安全性和兼容性。

### 🏛️ 传统 API 示例（仅供参考）

> ⚠️ **重要提醒**: 以下示例使用传统的 Base64 编码方式，存在 NGINX 缓冲区溢出风险和性能问题。新项目请使用上述 V2 API。

#### Python 客户端示例 (传统方式)

```python
import requests
import base64
from pathlib import Path

# ⚠️ 传统方式 - 不推荐用于新项目
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def ocr_recognize_legacy(image_path, model_version="v5-server"):
    """传统 Base64 方式 - 仅为兼容性保留"""
    url = "http://localhost:5005/ocr"
    
    payload = {
        "image": image_to_base64(image_path),
        "model_version": model_version,
        "det": True,
        "rec": True,
        "cls": True
    }
    
    response = requests.post(url, json=payload, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            for item in result["results"]:
                print(f"文本: {item['text']}")
                print(f"置信度: {item['confidence']:.4f}")
                print(f"位置: {item['bbox']}")
                print("-" * 40)
        else:
            print(f"处理失败: {result.get('error')}")
    else:
        print(f"请求失败: {response.status_code}")

# ⚠️ 建议替换为上述 V2 API 示例
if __name__ == "__main__":
    print("⚠️ 此示例使用传统API，建议使用 V2 API")
    ocr_recognize_legacy("test_image.jpg")
```

#### JavaScript 客户端示例 (传统方式)

```javascript
// ⚠️ 传统方式 - 不推荐用于新项目
class OCRLegacyClient {
    constructor(baseURL = 'http://localhost:5005') {
        this.baseURL = baseURL;
    }
    
    async fileToBase64(file) {
        console.warn('⚠️ 使用传统Base64方式，建议升级到V2 API');
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }
    
    async recognize(imageFile, options = {}) {
        const {
            modelVersion = 'v5-server',
            det = true,
            rec = true,
            cls = true
        } = options;
        
        try {
            const imageBase64 = await this.fileToBase64(imageFile);
            
            const response = await fetch(`${this.baseURL}/ocr`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageBase64,
                    model_version: modelVersion,
                    det,
                    rec,
                    cls
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                return result.results;
            } else {
                throw new Error(result.error || '处理失败');
            }
        } catch (error) {
            console.error('Legacy OCR recognition failed:', error);
            throw error;
        }
    }
}

// ⚠️ 建议使用上述 OCRV2Client 替代此传统客户端
console.warn('⚠️ 检测到传统API使用，建议升级到V2 API以获得更好的性能和安全性');
```

#### cURL 示例 (传统方式)

```bash
# ⚠️ 传统方式 - 存在NGINX缓冲区风险
echo "⚠️ 传统Base64方式可能导致NGINX缓冲区溢出，建议使用V2 API"

# 健康检查
curl -X GET "http://localhost:5005/health"

# 获取可用模型
curl -X GET "http://localhost:5005/models/available"

# 传统OCR识别（需要先将图片转换为base64）
base64_image=$(base64 -w 0 test_image.jpg)
curl -X POST "http://localhost:5005/ocr" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"$base64_image\",
    \"model_version\": \"v5-server\",
    \"det\": true,
    \"rec\": true,
    \"cls\": true
  }"

# 推荐使用V2 API替代:
# curl -X POST "http://localhost:5005/v2/ocr" -F "file=@test_image.jpg" -F "model_version=v5-server"
```

## 性能基准

### 测试环境
- **CPU**: Intel Core i7-12700K (16核)
- **内存**: 32GB DDR4
- **Python**: 3.13
- **ONNX Runtime**: 1.22.1

### API 性能对比

#### V2 API (multipart/form-data) - 推荐

| 模型版本 | 平均响应时间 | 吞吐量 | 内存占用 | 传输效率 |
|---------|-------------|-------|---------|----------|
| v4 | 165ms | 50 req/s | 1.1GB | +33% |
| v5 | 135ms | 60 req/s | 1.0GB | +33% |
| v5-server | 105ms | 70 req/s | 1.4GB | +33% |

#### V1 API (Base64) - 传统方式

| 模型版本 | 平均响应时间 | 吞吐量 | 内存占用 | NGINX风险 |
|---------|-------------|-------|---------|----------|
| v4 | 180ms | 45 req/s | 1.2GB | 高 |
| v5 | 150ms | 55 req/s | 1.1GB | 高 |
| v5-server | 120ms | 65 req/s | 1.5GB | 高 |

### 性能优势分析

**V2 API 优势**:
- ✅ **传输效率提升 33%** - 无Base64编码开销
- ✅ **内存使用优化** - 流式处理，减少内存峰值
- ✅ **更快响应时间** - 平均减少 15ms 处理时间
- ✅ **更高并发** - 支持更高的请求吞吐量
- ✅ **零NGINX风险** - 避免代理服务器缓冲区问题

**传统API局限**:
- ❌ **编码开销** - Base64增加33%数据量和CPU消耗
- ❌ **内存消耗** - 需要将完整图片加载到内存
- ❌ **代理风险** - 大payload可能导致NGINX等代理服务器问题
- ❌ **传输延迟** - 编码/解码增加处理时间

### 优化建议

1. **API选择**
   - 🚀 **新项目**: 强制使用 V2 API
   - 🔄 **现有项目**: 尽快迁移到 V2 API
   - 🏛️ **传统项目**: 仅在必要时保留 V1 API

2. **模型选择**
   - 高精度场景：使用 `v5-server`
   - 平衡场景：使用 `v5`  
   - 兼容场景：使用 `v4`

3. **性能优化**
   - 启用模型缓存（默认启用）
   - 合理设置 worker 数量
   - 使用 V2 批量处理减少网络开销
   - 配置适当的客户端超时时间

4. **资源配置**
   - 建议最少 4GB 内存
   - CPU 核心数影响并发性能
   - SSD 存储提升模型加载速度
   - 使用 V2 API 可减少约 20% 内存占用

## 部署指南

### Docker 部署

```bash
# 构建镜像
docker build -t onnx-ocr .

# 运行容器
docker run -d \
  --name onnx-ocr \
  -p 5005:5005 \
  -e WORKERS=8 \
  -e LOG_LEVEL=info \
  onnx-ocr

# 使用 docker-compose
docker-compose up -d
```

### 生产环境配置

```bash
# 环境变量配置
export HOST=0.0.0.0
export PORT=5005
export WORKERS=auto
export LOG_LEVEL=info

# 启动生产服务器
python start_production.py
```

## 监控与日志

### 结构化日志
系统使用结构化日志记录关键事件：

```json
{
  "event": "OCR处理完成",
  "logger": "onnxocr.api",
  "level": "info", 
  "timestamp": "2024-01-01T12:00:00Z",
  "model_version": "v5-server",
  "process_time": 156.78,
  "num_detected": 2,
  "image_shape": [400, 600, 3]
}
```

### Prometheus 监控
集成 Prometheus 指标监控：
- 请求计数和响应时间
- 模型加载时间和缓存命中率
- 错误率和成功率统计
- 系统资源使用情况

## 支持与联系

- **项目地址**: [GitHub Repository]
- **问题反馈**: [Issues]  
- **文档更新**: 2024-01-01
- **API 版本**: v2.0.0