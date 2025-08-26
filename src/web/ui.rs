use axum::{
    response::{Html, IntoResponse},
    http::{StatusCode, HeaderMap, HeaderValue},
};

/// 首页处理器
pub async fn index_handler() -> impl IntoResponse {
    let html = include_str!("../../templates/index.html");
    Html(html)
}

/// 生成首页HTML内容
fn generate_index_html() -> String {
    r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX OCR Service</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #333;
        }

        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 90%;
            text-align: center;
        }

        h1 {
            color: #5a67d8;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }

        .upload-area {
            border: 2px dashed #cbd5e0;
            border-radius: 15px;
            padding: 40px 20px;
            margin: 30px 0;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8fafc;
        }

        .upload-area:hover, .upload-area.drag-over {
            border-color: #5a67d8;
            background: #edf2f7;
            transform: translateY(-2px);
        }

        .upload-text {
            font-size: 1.2em;
            color: #4a5568;
            margin-bottom: 10px;
        }

        .upload-hint {
            color: #718096;
            font-size: 0.9em;
        }

        #fileInput {
            display: none;
        }

        .options {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
            gap: 15px;
        }

        .option {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #f1f5f9;
            padding: 10px 15px;
            border-radius: 10px;
            font-size: 0.9em;
        }

        .option input {
            margin: 0;
        }

        .btn {
            background: linear-gradient(135deg, #5a67d8, #667eea);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
            min-width: 150px;
        }

        .btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(90, 103, 216, 0.3);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .loading {
            display: none;
            color: #5a67d8;
            margin: 20px 0;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #5a67d8;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results {
            margin-top: 30px;
            text-align: left;
            display: none;
        }

        .result-item {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }

        .result-text {
            font-size: 1.1em;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .result-confidence {
            color: #666;
            font-size: 0.9em;
        }

        .api-info {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid #e2e8f0;
            text-align: left;
        }

        .api-endpoint {
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', monospace;
            margin: 10px 0;
            overflow-x: auto;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin: 10px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            .options {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 ONNX OCR Service</h1>
        <p class="subtitle">高性能文字识别服务 - 支持双模式上传</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-text">📁 点击或拖拽图片到此处</div>
            <div class="upload-hint">支持 PNG, JPEG, BMP, TIFF 格式，最大 50MB</div>
        </div>
        
        <input type="file" id="fileInput" accept="image/*">
        
        <div class="options">
            <label class="option">
                <input type="checkbox" id="useAngleCls" checked>
                <span>启用角度分类</span>
            </label>
            <label class="option">
                <input type="number" id="minConfidence" value="0.5" min="0" max="1" step="0.1">
                <span>最小置信度</span>
            </label>
            <label class="option">
                <select id="outputFormat">
                    <option value="json">JSON格式</option>
                    <option value="text">纯文本</option>
                    <option value="csv">CSV格式</option>
                </select>
            </label>
        </div>
        
        <button class="btn" id="processBtn" onclick="processImage()">🚀 开始识别</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <span>正在处理图像，请稍候...</span>
        </div>
        
        <div class="results" id="results">
            <h3>识别结果</h3>
            <div id="resultContent"></div>
        </div>
        
        <div class="api-info">
            <h3>API接口文档</h3>
            <h4>1. JSON Base64上传</h4>
            <div class="api-endpoint">
POST /ocr
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "force_ocr": false,
  "use_angle_cls": true,
  "min_confidence": 0.5,
  "output_format": "json"
}</div>
            
            <h4>2. 文件上传</h4>
            <div class="api-endpoint">
POST /ocr/upload
Content-Type: multipart/form-data

Form fields:
- file: 图像文件
- use_angle_cls: true/false
- min_confidence: 0.0-1.0
- output_format: json/text/csv</div>
            
            <h4>3. 其他接口</h4>
            <div class="api-endpoint">
GET /health      - 健康检查
GET /api/info    - 服务信息</div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const resultContent = document.getElementById('resultContent');
        
        let selectedFile = null;

        // 点击上传区域
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        // 文件选择
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });

        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            if (!file) return;
            
            if (!file.type.startsWith('image/')) {
                alert('请选择图像文件！');
                return;
            }
            
            if (file.size > 50 * 1024 * 1024) {
                alert('文件大小不能超过50MB！');
                return;
            }
            
            selectedFile = file;
            uploadArea.innerHTML = `
                <div class="upload-text">✅ 已选择: ${file.name}</div>
                <div class="upload-hint">${(file.size / 1024 / 1024).toFixed(2)} MB</div>
            `;
            processBtn.disabled = false;
        }

        async function processImage() {
            if (!selectedFile) {
                alert('请先选择图像文件！');
                return;
            }

            processBtn.disabled = true;
            loading.style.display = 'block';
            results.style.display = 'none';

            try {
                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('use_angle_cls', document.getElementById('useAngleCls').checked);
                formData.append('min_confidence', document.getElementById('minConfidence').value);
                formData.append('output_format', document.getElementById('outputFormat').value);

                const response = await fetch('/ocr/upload', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    displayResults(result.data);
                } else {
                    throw new Error(result.error?.message || '处理失败');
                }
            } catch (error) {
                alert('处理失败: ' + error.message);
                console.error('Error:', error);
            } finally {
                processBtn.disabled = false;
                loading.style.display = 'none';
            }
        }

        function displayResults(data) {
            if (!data.results || data.results.length === 0) {
                resultContent.innerHTML = '<p style="color: #666;">未识别到文字内容</p>';
            } else {
                const html = data.results.map(item => `
                    <div class="result-item">
                        <div class="result-text">${item.text}</div>
                        <div class="result-confidence">置信度: ${(item.confidence * 100).toFixed(1)}%</div>
                    </div>
                `).join('');
                
                resultContent.innerHTML = `
                    <p><strong>处理时间:</strong> ${data.processing_time.toFixed(3)}秒</p>
                    <p><strong>识别结果:</strong> ${data.results.length}个文本</p>
                    <div style="margin-top: 15px;">${html}</div>
                `;
            }
            
            results.style.display = 'block';
        }
    </script>
</body>
</html>"#.to_string()
}

/// 样式文件处理器
pub async fn style_handler() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/css"));
    
    let css = r#"
    /* 简化的CSS样式 */
    body { 
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 20px;
        background-color: #f5f5f5;
    }
    .container {
        max-width: 800px;
        margin: 0 auto;
        background: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    "#;
    
    (headers, css)
}

/// 在实际部署中，这个函数会返回嵌入的HTML内容
pub fn get_embedded_html() -> &'static str {
    // 在编译时嵌入HTML文件，如果文件不存在则返回生成的HTML
    match std::include_str!("../../templates/index.html") {
        html if html.len() > 100 => html, // 简单检查文件是否有效
        _ => {
            // 如果文件不存在或无效，使用内联HTML
            &generate_index_html()
        }
    }
}