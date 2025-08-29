# 使用 Python 3.7 作为基础镜像
FROM python:3.7-slim

# 设置工作目录
WORKDIR /app

# 复制新的FastAPI依赖文件
COPY requirements-fastapi.txt .

# 更换为清华大学镜像源 (避免新旧源同时使用)
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements-fastapi.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 复制项目目录中的所有文件到镜像中
COPY . .

# 设置环境变量（如果需要）
ENV PYTHONUNBUFFERED=1

# 暴露服务端口（假设你的 Flask 服务运行在 5005 端口）
EXPOSE 5005

# 启动 FastAPI 服务
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5005", "--workers", "4", "--threads", "2", "--preload"]