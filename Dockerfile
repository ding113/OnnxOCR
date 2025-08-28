# 使用国内镜像源的Rust基础镜像
FROM rust:slim as builder

# 设置工作目录
WORKDIR /app

# 配置国内Cargo镜像源
RUN mkdir -p ~/.cargo && echo '[source.crates-io] \n\
replace-with = "tuna" \n\
[source.tuna] \n\
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"' > ~/.cargo/config

# 更换为清华大学镜像源
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY templates/ ./templates/

# 构建项目（release模式）
RUN cargo build --release

# 运行时镜像 - 使用国内镜像
FROM debian:bookworm-slim

# 更换为清华大学镜像源
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 创建数据目录
RUN mkdir -p /app/models /app/logs /app/results /app/uploads

# 复制编译好的二进制文件
COPY --from=builder /app/target/release/onnx-ocr /usr/local/bin/onnx-ocr
COPY templates/ ./templates/
COPY models/ ./models/

# 设置权限
RUN chmod +x /usr/local/bin/onnx-ocr

# 设置环境变量
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# 暴露端口
EXPOSE 5005

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

# 启动命令
CMD ["onnx-ocr", "--bind", "0.0.0.0:5005", "--models-dir", "/app/models"]