/**
 * Modern ONNX OCR Web UI JavaScript
 * Handles image upload, API calls, and result visualization
 */

class OCRWebUI {
    constructor() {
        this.currentImage = null;
        this.currentImageData = null;
        this.apiBase = '';
        this.requestCount = 0;
        this.totalTime = 0;
        
        this.initializeElements();
        this.attachEventListeners();
        this.updatePerformanceInfo();
        this.checkSystemStatus();
    }

    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        
        // Control elements
        this.modelSelect = document.getElementById('modelSelect');
        this.detCheck = document.getElementById('detCheck');
        this.recCheck = document.getElementById('recCheck');
        this.clsCheck = document.getElementById('clsCheck');
        this.processBtn = document.getElementById('processBtn');
        
        // Result elements
        this.resultsSection = document.getElementById('resultsSection');
        this.originalImage = document.getElementById('originalImage');
        this.resultCanvas = document.getElementById('resultCanvas');
        this.textList = document.getElementById('textList');
        
        // Info elements
        this.currentModelSpan = document.getElementById('currentModel');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.imageInfo = document.getElementById('imageInfo');
        this.imageResolution = document.getElementById('imageResolution');
        this.imageSize = document.getElementById('imageSize');
        this.processStats = document.getElementById('processStats');
        this.processTime = document.getElementById('processTime');
        this.boxCount = document.getElementById('boxCount');
        this.textCount = document.getElementById('textCount');
        
        // Progress elements
        this.progressOverlay = document.getElementById('progressOverlay');
        this.progressText = document.getElementById('progressText');
        this.progressFill = document.getElementById('progressFill');
        
        // Modal elements
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.modalClose = document.getElementById('modalClose');
        
        // Export elements
        this.exportJsonBtn = document.getElementById('exportJsonBtn');
        this.exportTxtBtn = document.getElementById('exportTxtBtn');
        
        // Performance elements
        this.totalRequests = document.getElementById('totalRequests');
        this.avgTime = document.getElementById('avgTime');
    }

    attachEventListeners() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.fileInput.click();
        });
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Control events
        this.modelSelect.addEventListener('change', () => this.handleModelChange());
        this.processBtn.addEventListener('click', () => this.processImage());
        
        // Export events
        this.exportJsonBtn.addEventListener('click', () => this.exportResults('json'));
        this.exportTxtBtn.addEventListener('click', () => this.exportResults('txt'));
        
        // Modal events
        this.modalClose.addEventListener('click', () => this.hideError());
        this.errorModal.addEventListener('click', (e) => {
            if (e.target === this.errorModal) this.hideError();
        });
        
        // Keyboard events
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideError();
            }
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateStatus('ready', '就绪');
                this.currentModelSpan.textContent = data.default_model || 'v5-server';
            } else {
                this.updateStatus('error', '系统异常');
            }
        } catch (error) {
            this.updateStatus('error', '连接失败');
            console.error('System status check failed:', error);
        }
    }

    updateStatus(type, text) {
        this.statusIndicator.className = `status-indicator ${type}`;
        this.statusIndicator.querySelector('span').textContent = text;
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (!this.uploadArea.contains(e.relatedTarget)) {
            this.uploadArea.classList.remove('dragover');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    async handleFile(file) {
        // Validate file
        if (!file.type.startsWith('image/')) {
            this.showError('请选择有效的图片文件');
            return;
        }

        if (file.size > 10 * 1024 * 1024) { // 10MB
            this.showError('图片大小不能超过 10MB');
            return;
        }

        try {
            // Read file as base64
            const base64 = await this.fileToBase64(file);
            this.currentImageData = base64.split(',')[1]; // Remove data URL prefix
            
            // Load image for display and info
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.displayImage(img);
                this.updateImageInfo(img, file);
                this.processBtn.disabled = false;
            };
            img.src = base64;
            
        } catch (error) {
            this.showError('图片加载失败: ' + error.message);
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    displayImage(img) {
        this.originalImage.src = img.src;
        this.originalImage.style.display = 'block';
        
        // Update upload area
        this.uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="fas fa-check-circle upload-icon" style="color: var(--success-color);"></i>
                <h3>图片已加载</h3>
                <p>点击下方"开始识别"按钮进行处理</p>
            </div>
        `;
    }

    updateImageInfo(img, file) {
        this.imageResolution.textContent = `${img.width} × ${img.height}`;
        this.imageSize.textContent = this.formatFileSize(file.size);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async handleModelChange() {
        const newModel = this.modelSelect.value;
        
        try {
            this.updateStatus('processing', '切换模型中...');
            
            const response = await fetch('/models/switch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_version: newModel
                })
            });

            const result = await response.json();
            
            if (response.ok && result.success) {
                this.currentModelSpan.textContent = newModel;
                this.updateStatus('ready', '就绪');
                this.showSuccess(`已切换到 ${newModel} 模型`);
            } else {
                throw new Error(result.error || '模型切换失败');
            }
        } catch (error) {
            this.showError('模型切换失败: ' + error.message);
            this.updateStatus('error', '模型切换失败');
        }
    }

    async processImage() {
        if (!this.currentImageData) {
            this.showError('请先上传图片');
            return;
        }

        const startTime = Date.now();
        
        try {
            this.showProgress('正在处理图片...', 0);
            this.updateStatus('processing', '处理中...');
            
            // Simulate progress
            this.simulateProgress();
            
            const requestData = {
                image: this.currentImageData,
                model_version: this.modelSelect.value,
                det: this.detCheck.checked,
                rec: this.recCheck.checked,
                cls: this.clsCheck.checked
            };

            const response = await fetch('/ocr', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || '处理失败');
            }

            const processTime = Date.now() - startTime;
            this.requestCount++;
            this.totalTime += processTime;
            
            this.hideProgress();
            this.displayResults(result, processTime);
            this.updateStatus('ready', '就绪');
            this.updatePerformanceInfo();
            
        } catch (error) {
            this.hideProgress();
            this.showError('图片处理失败: ' + error.message);
            this.updateStatus('error', '处理失败');
        }
    }

    showProgress(text, progress = 0) {
        this.progressText.textContent = text;
        this.progressFill.style.width = progress + '%';
        this.progressOverlay.style.display = 'flex';
    }

    hideProgress() {
        this.progressOverlay.style.display = 'none';
    }

    simulateProgress() {
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress >= 90) {
                clearInterval(interval);
                progress = 95;
            }
            this.progressFill.style.width = progress + '%';
        }, 200);
        
        // Clear interval after 10 seconds as fallback
        setTimeout(() => clearInterval(interval), 10000);
    }

    displayResults(result, processTime) {
        // Update stats
        this.processTime.textContent = processTime;
        this.boxCount.textContent = result.num_detected || 0;
        this.textCount.textContent = result.results ? result.results.length : 0;
        
        // Draw result on canvas
        this.drawResultCanvas(result.results || []);
        
        // Display text results
        this.displayTextResults(result.results || []);
        
        // Show results section
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Store results for export
        this.lastResults = result;
    }

    drawResultCanvas(results) {
        const canvas = this.resultCanvas;
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match image
        const img = this.currentImage;
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw image
        ctx.drawImage(img, 0, 0);
        
        // Draw bounding boxes and text
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(37, 99, 235, 0.1)';
        ctx.font = '16px -apple-system, sans-serif';
        
        results.forEach((item, index) => {
            if (item.bbox && item.bbox.length >= 4) {
                const [x1, y1, x2, y2] = item.bbox;
                
                // Draw bounding box
                ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Draw index number
                ctx.fillStyle = '#2563eb';
                ctx.fillText((index + 1).toString(), x1 + 2, y1 - 5);
                ctx.fillStyle = 'rgba(37, 99, 235, 0.1)';
            }
        });
    }

    displayTextResults(results) {
        this.textList.innerHTML = '';
        
        if (results.length === 0) {
            this.textList.innerHTML = '<div class="text-item"><div class="text-content"><div class="text-value">未检测到文本</div></div></div>';
            return;
        }

        results.forEach((item, index) => {
            const confidence = item.confidence || 0;
            const confidenceLevel = confidence > 0.8 ? 'high' : confidence > 0.5 ? 'medium' : 'low';
            
            const textItem = document.createElement('div');
            textItem.className = `text-item ${confidenceLevel}-confidence`;
            textItem.innerHTML = `
                <div class="text-content">
                    <div class="text-value">${this.escapeHtml(item.text || '')}</div>
                    <div class="text-confidence">位置: ${item.bbox ? item.bbox.map(v => Math.round(v)).join(', ') : '未知'}</div>
                </div>
                <div class="confidence-badge ${confidenceLevel}">
                    ${Math.round(confidence * 100)}%
                </div>
            `;
            this.textList.appendChild(textItem);
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    exportResults(format) {
        if (!this.lastResults) {
            this.showError('没有可导出的结果');
            return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        let content, filename, mimeType;

        if (format === 'json') {
            content = JSON.stringify(this.lastResults, null, 2);
            filename = `ocr_results_${timestamp}.json`;
            mimeType = 'application/json';
        } else if (format === 'txt') {
            content = this.lastResults.results
                .map(item => item.text || '')
                .filter(text => text.trim())
                .join('\n');
            filename = `ocr_results_${timestamp}.txt`;
            mimeType = 'text/plain';
        }

        this.downloadFile(content, filename, mimeType);
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    updatePerformanceInfo() {
        this.totalRequests.textContent = this.requestCount;
        this.avgTime.textContent = this.requestCount > 0 
            ? Math.round(this.totalTime / this.requestCount) + 'ms' 
            : '-';
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'flex';
    }

    showSuccess(message) {
        // Create temporary success notification
        const notification = document.createElement('div');
        notification.className = 'success-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success-color);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-lg);
            z-index: 3000;
            animation: slideInDown 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideInUp 0.3s ease reverse';
                setTimeout(() => document.body.removeChild(notification), 300);
            }
        }, 3000);
    }

    hideError() {
        this.errorModal.style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new OCRWebUI();
});

// Add some utility CSS for success notifications
const style = document.createElement('style');
style.textContent = `
    .success-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--success-color);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-lg);
        z-index: 3000;
        font-weight: 500;
        animation: slideInDown 0.3s ease;
    }
`;
document.head.appendChild(style);