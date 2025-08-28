# [BUILD] 第一阶段: 构建环境 (Builder Stage)
FROM python:latest AS builder

# [MIRROR] 更换为清华大学镜像源 (避免新旧源同时使用)
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装uv包管理器
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /usr/local/bin/uv

# 设置工作目录
WORKDIR /app

# 复制项目配置文件
COPY pyproject.toml ./
COPY uv.lock* ./

# 创建虚拟环境并安装依赖
ENV UV_SYSTEM_PYTHON=1 \
    UV_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/
RUN uv sync --frozen --no-dev

# ============================================================================
# [RUN] 第二阶段: 运行时环境 (Runtime Stage)  
FROM python:latest AS runtime

# [META] 元数据标签
LABEL maintainer="ONNX OCR Team <team@onnxocr.com>"
LABEL version="2.0.0"
LABEL description="Modern high-performance ONNX OCR service"
LABEL python.version="3.13"
LABEL framework="FastAPI"

# [SECURITY] 创建非特权用户
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# [MIRROR] 更换为清华大学镜像源 (避免新旧源同时使用)
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources

# [DEPS] 安装运行时系统依赖 (OpenCV + ONNX Runtime所需)
RUN apt-get update && apt-get install -y \
    # OpenCV依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # 网络和监控工具
    curl \
    # 图像处理增强
    libglib2.0-0 \
    libgtk-3-0 \
    # 清理缓存
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# [DIR] 设置工作目录和权限
WORKDIR /app
RUN chown -R appuser:appuser /app

# [PYTHON] 复制Python环境
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# [FOLDER] 创建必要目录
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R appuser:appuser /app

# [CODE] 复制应用代码
COPY --chown=appuser:appuser . .

# [ENV] 设置环境变量
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    # ONNX Runtime优化
    ORT_DISABLE_ALL_OPTIMIZATIONS=0 \
    ORT_ENABLE_CPU_FP16_OPS=1 \
    # OpenMP优化 (自动检测CPU核心数)
    OMP_NUM_THREADS=0 \
    MKL_NUM_THREADS=0 \
    OPENBLAS_NUM_THREADS=0 \
    # 服务配置
    HOST=0.0.0.0 \
    PORT=5005 \
    LOG_LEVEL=info \
    WORKERS=auto

# [USER] 切换到非特权用户
USER appuser

# [VERIFY] 验证安装
RUN python -c "import onnxruntime, cv2, fastapi; print('[OK] 核心依赖验证成功')" && \
    python -c "import numpy, structlog; print('[OK] 辅助依赖验证成功')"

# [PORT] 暴露端口
EXPOSE 5005

# [HEALTH] 健康检查 (每30秒检查一次，启动等待60秒)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

# [LABELS] 添加启动信息标签
LABEL startup.info="Service will be available at http://localhost:5005"
LABEL api.docs="http://localhost:5005/docs"
LABEL health.check="http://localhost:5005/health"
LABEL metrics="http://localhost:5005/metrics"

# [START] 启动命令 (生产模式)
CMD ["python", "start_production.py"]