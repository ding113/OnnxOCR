@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo 🦀 Rust OCR 本地开发环境配置脚本
echo ===================================

REM 检查Rust是否已安装
rustc --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Rust 已安装
    rustc --version
    cargo --version
    goto :build_project
)

echo ❌ Rust 未安装，开始安装...
echo.
echo 请选择安装方式：
echo 1. 自动下载安装 (推荐)
echo 2. 手动安装说明
choice /c 12 /m "请输入选择"

if errorlevel 2 goto :manual_install
if errorlevel 1 goto :auto_install

:auto_install
echo 📥 正在下载 Rust 安装程序...
powershell -Command "& {Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile 'rustup-init.exe'}"

if not exist rustup-init.exe (
    echo ❌ 下载失败，请检查网络连接或手动安装
    goto :manual_install
)

echo 🔧 启动 Rust 安装程序...
echo 安装提示中请选择默认选项（按回车键）
rustup-init.exe
del rustup-init.exe

echo 重新加载环境变量...
call refreshenv

goto :verify_rust

:manual_install
echo 📖 手动安装说明：
echo 1. 访问 https://rustup.rs/
echo 2. 下载 rustup-init.exe
echo 3. 运行安装程序，选择默认选项
echo 4. 重启命令提示符
echo 5. 重新运行此脚本
echo.
pause
exit /b 1

:verify_rust
echo 🔍 验证 Rust 安装...
rustc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Rust 安装失败或环境变量未生效
    echo 请重启命令提示符后重新运行此脚本
    pause
    exit /b 1
)

echo ✅ Rust 安装成功！
rustc --version
cargo --version

:build_project
echo.
echo 🔨 开始编译项目...
echo =====================

REM 检查项目文件
if not exist Cargo.toml (
    echo ❌ 当前目录下未找到 Cargo.toml 文件
    echo 请确保在项目根目录执行此脚本
    pause
    exit /b 1
)

REM 配置国内镜像源
if not exist "%USERPROFILE%\.cargo" mkdir "%USERPROFILE%\.cargo"
echo [source.crates-io] > "%USERPROFILE%\.cargo\config"
echo replace-with = "tuna" >> "%USERPROFILE%\.cargo\config"
echo [source.tuna] >> "%USERPROFILE%\.cargo\config"
echo registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git" >> "%USERPROFILE%\.cargo\config"
echo ✅ 已配置国内 Cargo 镜像源

REM 检查模型文件
if not exist "resources\onnxocr\models" (
    echo ❌ 模型文件目录不存在: resources\onnxocr\models
    echo 请从原Python项目复制模型文件到此目录
    pause
    exit /b 1
)

REM 编译项目
echo 📦 正在下载依赖和编译...
cargo check
if %errorlevel% neq 0 (
    echo ❌ 编译检查失败
    pause
    exit /b 1
)

echo ✅ 编译检查成功！

echo.
echo 🚀 编译 Release 版本...
cargo build --release
if %errorlevel% neq 0 (
    echo ❌ Release 编译失败
    pause
    exit /b 1
)

echo ✅ Release 编译成功！

REM 创建本地数据目录
if not exist "data" mkdir data
if not exist "data\logs" mkdir data\logs
if not exist "data\results" mkdir data\results
if not exist "data\uploads" mkdir data\uploads

echo.
echo 🎉 编译完成！可执行文件位置：
echo target\release\onnx-ocr.exe
echo.
echo 📋 启动服务：
echo target\release\onnx-ocr.exe --bind 127.0.0.1:5005 --models-dir resources\onnxocr\models
echo.
echo 🌐 访问地址：
echo http://127.0.0.1:5005
echo.
echo 是否现在启动服务？(Y/N)
choice /c YN /m "请选择"

if errorlevel 2 goto :end
if errorlevel 1 goto :start_service

:start_service
echo 🚀 启动 OCR 服务...
target\release\onnx-ocr.exe --bind 127.0.0.1:5005 --models-dir resources\onnxocr\models

:end
echo.
echo 📚 常用命令：
echo   编译检查: cargo check
echo   运行服务: cargo run -- --bind 127.0.0.1:5005 --models-dir resources\onnxocr\models
echo   编译发布: cargo build --release
echo.
pause