"""
模型下载管理器
自动下载PPOCRv5 Server版本模型文件，支持断点续传、重试机制和原子性操作
"""

import asyncio
import os
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import aiohttp
from aiohttp import ClientSession, ClientTimeout

from .logging import get_logger

logger = get_logger("app.model_downloader")


class ModelDownloadManager:
    """模型下载管理器"""
    
    def __init__(self):
        self.download_urls = {
            "det.onnx": "https://box.ygxz.in/f/d/MgCk/det.onnx",
            "rec.onnx": "https://box.ygxz.in/f/d/Z9cl/rec.onnx"
        }
        
        # 预期文件大小（字节），用于完整性检查
        self.expected_sizes = {
            "det.onnx": None,  # 暂时不设置，下载后记录实际大小
            "rec.onnx": None   # 暂时不设置，下载后记录实际大小
        }
        
        # 下载配置
        self.timeout = ClientTimeout(total=300, connect=30)  # 5分钟超时
        self.max_retries = 3
        self.chunk_size = 8192  # 8KB chunks
        
        # 模型目录路径
        self.base_dir = Path(__file__).parent.parent / "onnxocr" / "models"
        self.server_model_dir = self.base_dir / "ppocrv5-server"
        self.mobile_model_dir = self.base_dir / "ppocrv5"
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """
        确保模型文件可用，自动下载缺失文件
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否可用
        """
        if model_name == "PP-OCRv5-Server":
            return await self._prepare_server_model()
        return True  # 其他模型默认可用
    
    async def _prepare_server_model(self) -> bool:
        """
        准备PPOCRv5 Server版本模型文件
        
        Returns:
            bool: 准备是否成功
        """
        try:
            logger.info("Preparing PPOCRv5-Server model files")
            
            # 1. 创建目录结构
            await self._create_directory_structure()
            
            # 2. 检查并下载det.onnx和rec.onnx
            download_success = await self._download_missing_files()
            
            if not download_success:
                logger.warning("Failed to download server model files")
                return False
            
            # 3. 创建软链接到cls.onnx和字典文件
            await self._create_symlinks()
            
            logger.info("PPOCRv5-Server model prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare PPOCRv5-Server model: {e}")
            return False
    
    async def _create_directory_structure(self):
        """创建目录结构"""
        directories = [
            self.server_model_dir / "det",
            self.server_model_dir / "rec",
            self.server_model_dir / "cls"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    async def _download_missing_files(self) -> bool:
        """
        下载缺失的模型文件
        
        Returns:
            bool: 所有文件下载是否成功
        """
        download_tasks = []
        
        for filename, url in self.download_urls.items():
            if filename == "det.onnx":
                target_path = self.server_model_dir / "det" / filename
            elif filename == "rec.onnx":
                target_path = self.server_model_dir / "rec" / filename
            else:
                continue
                
            if not target_path.exists():
                logger.info(f"Need to download: {filename}")
                download_tasks.append(
                    self._download_file_with_retry(url, target_path, filename)
                )
            else:
                logger.info(f"File already exists: {filename}")
        
        if not download_tasks:
            logger.info("All server model files already exist")
            return True
        
        # 并行下载所有缺失文件
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # 检查下载结果
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Download task {i} failed: {result}")
            elif result:
                success_count += 1
        
        total_tasks = len(download_tasks)
        logger.info(f"Download completed: {success_count}/{total_tasks} files successful")
        
        return success_count == total_tasks
    
    async def _download_file_with_retry(
        self, 
        url: str, 
        target_path: Path, 
        filename: str
    ) -> bool:
        """
        带重试机制的文件下载
        
        Args:
            url: 下载URL
            target_path: 目标文件路径
            filename: 文件名
            
        Returns:
            bool: 下载是否成功
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading {filename} (attempt {attempt + 1}/{self.max_retries})")
                success = await self._download_file(url, target_path)
                if success:
                    logger.info(f"Successfully downloaded {filename}")
                    return True
                    
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed for {filename}: {e}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
        
        logger.error(f"Failed to download {filename} after {self.max_retries} attempts")
        return False
    
    async def _download_file(self, url: str, target_path: Path) -> bool:
        """
        下载单个文件，支持原子性操作
        
        Args:
            url: 下载URL
            target_path: 目标文件路径
            
        Returns:
            bool: 下载是否成功
        """
        # 使用临时文件确保原子性操作
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            async with ClientSession(timeout=self.timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    # 获取文件大小
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    logger.info(f"Downloading {target_path.name}, size: {total_size} bytes")
                    
                    # 分块下载
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 记录进度（每10%记录一次）
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if downloaded % (total_size // 10) < self.chunk_size:
                                    logger.info(f"Download progress: {progress:.1f}%")
                    
                    # 验证文件大小
                    if total_size > 0 and downloaded != total_size:
                        raise ValueError(f"Downloaded size mismatch: {downloaded} != {total_size}")
                    
                    # 原子性移动到目标位置
                    shutil.move(str(temp_path), str(target_path))
                    logger.info(f"Successfully downloaded and moved to {target_path}")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    async def _create_symlinks(self):
        """
        创建软链接复用mobile版本的cls.onnx和字典文件
        """
        # 软链接映射：目标路径 -> 源路径
        symlinks = {
            self.server_model_dir / "cls" / "cls.onnx": self.mobile_model_dir / "cls" / "cls.onnx",
            self.server_model_dir / "ppocrv5_dict.txt": self.mobile_model_dir / "ppocrv5_dict.txt"
        }
        
        for target, source in symlinks.items():
            try:
                # 检查源文件是否存在
                if not source.exists():
                    logger.error(f"Source file does not exist: {source}")
                    continue
                
                # 如果目标已存在且是正确的软链接，跳过
                if target.exists() or target.is_symlink():
                    if target.is_symlink() and target.resolve() == source.resolve():
                        logger.debug(f"Symlink already correct: {target}")
                        continue
                    else:
                        # 删除错误的链接或文件
                        target.unlink()
                
                # 创建软链接
                if os.name == 'nt':  # Windows
                    # Windows使用硬链接或复制文件
                    try:
                        target.hardlink_to(source)
                        logger.info(f"Created hardlink: {target} -> {source}")
                    except OSError:
                        # 如果硬链接失败，复制文件
                        shutil.copy2(source, target)
                        logger.info(f"Copied file: {source} -> {target}")
                else:  # Unix-like systems
                    target.symlink_to(source)
                    logger.info(f"Created symlink: {target} -> {source}")
                    
            except Exception as e:
                logger.error(f"Failed to create symlink {target} -> {source}: {e}")
                # 对于关键文件，如果软链接失败则复制
                try:
                    shutil.copy2(source, target)
                    logger.info(f"Fallback: copied file {source} -> {target}")
                except Exception as copy_error:
                    logger.error(f"Failed to copy file as fallback: {copy_error}")
    
    def is_server_model_complete(self) -> bool:
        """
        检查PPOCRv5-Server模型文件是否完整
        
        Returns:
            bool: 模型是否完整
        """
        required_files = [
            self.server_model_dir / "det" / "det.onnx",
            self.server_model_dir / "rec" / "rec.onnx", 
            self.server_model_dir / "cls" / "cls.onnx",
            self.server_model_dir / "ppocrv5_dict.txt"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.debug(f"Missing model file: {file_path}")
                return False
        
        logger.debug("All PPOCRv5-Server model files are present")
        return True
    
    async def cleanup_incomplete_download(self):
        """清理不完整的下载文件"""
        try:
            if self.server_model_dir.exists():
                temp_files = list(self.server_model_dir.rglob("*.tmp"))
                temp_files.extend(list(self.server_model_dir.rglob("*.part")))
                
                for temp_file in temp_files:
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup incomplete downloads: {e}")


# 全局实例
model_downloader = ModelDownloadManager()


def get_model_downloader() -> ModelDownloadManager:
    """获取模型下载管理器实例"""
    return model_downloader