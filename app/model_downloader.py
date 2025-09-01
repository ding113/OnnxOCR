"""
模型下载管理器
自动下载PPOCRv5 Server版本模型文件，支持断点续传、重试机制和原子性操作
"""

import asyncio
import os
import hashlib
import tempfile
import shutil
import time
import fcntl
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
        
        # 锁文件路径
        self.lock_file_path = self.server_model_dir / ".download.lock"
        self.download_status_file = self.server_model_dir / ".downloading"
    
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
        准备PPOCRv5 Server版本模型文件（带跨进程锁保护）
        
        Returns:
            bool: 准备是否成功
        """
        try:
            logger.info("Preparing PPOCRv5-Server model files")
            
            # 首先快速检查模型是否完整
            if self.is_server_model_complete():
                logger.info("PPOCRv5-Server model files already complete")
                return True
            
            # 获取跨进程锁
            return await self._prepare_server_model_with_lock()
            
        except Exception as e:
            logger.error(f"Failed to prepare PPOCRv5-Server model: {e}")
            return False
    
    async def _prepare_server_model_with_lock(self) -> bool:
        """
        使用文件锁确保只有一个进程下载模型
        
        Returns:
            bool: 准备是否成功
        """
        # 确保锁文件目录存在
        self.server_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试获取锁，最多等待5分钟
        lock_acquired = False
        max_wait_time = 300  # 5分钟
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait_time:
            try:
                # 尝试获取排他锁
                with open(self.lock_file_path, 'w') as lock_file:
                    if os.name != 'nt':  # Unix-like systems
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    lock_acquired = True
                    logger.info(f"Acquired download lock (PID: {os.getpid()})")
                    
                    # 再次检查模型是否已完整（其他进程可能已下载完成）
                    if self.is_server_model_complete():
                        logger.info("Model files already complete, no download needed")
                        return True
                    
                    # 标记正在下载状态
                    await self._mark_download_in_progress()
                    
                    # 执行下载
                    result = await self._perform_model_download()
                    
                    # 清理下载状态
                    await self._clear_download_status()
                    
                    return result
                    
            except (OSError, IOError) as e:
                if os.name == 'nt':
                    # Windows下使用文件存在检查作为锁机制
                    if self.lock_file_path.exists():
                        # 检查是否有其他进程正在下载
                        if await self._wait_for_other_download():
                            return True
                        await asyncio.sleep(2)
                        continue
                    else:
                        # 创建锁文件继续执行
                        lock_acquired = True
                        break
                else:
                    # Unix系统锁被占用，等待
                    logger.info("Download lock held by another process, waiting...")
                    await asyncio.sleep(5)
                    
                    # 检查其他进程是否已完成下载
                    if self.is_server_model_complete():
                        logger.info("Another process completed the download")
                        return True
        
        if not lock_acquired:
            logger.warning("Failed to acquire download lock within timeout, proceeding anyway")
            
        # 如果没有获取到锁，但模型已完整，返回成功
        if self.is_server_model_complete():
            return True
        
        logger.warning("Download lock timeout, attempting download without lock")
        return await self._perform_model_download()
    
    async def _mark_download_in_progress(self):
        """标记下载正在进行"""
        try:
            with open(self.download_status_file, 'w') as f:
                f.write(f"pid={os.getpid()},start={time.time()}")
            logger.debug("Marked download in progress")
        except Exception as e:
            logger.warning(f"Failed to mark download status: {e}")
    
    async def _clear_download_status(self):
        """清理下载状态标记"""
        try:
            if self.download_status_file.exists():
                self.download_status_file.unlink()
            if self.lock_file_path.exists():
                self.lock_file_path.unlink()
            logger.debug("Cleared download status")
        except Exception as e:
            logger.warning(f"Failed to clear download status: {e}")
    
    async def _wait_for_other_download(self) -> bool:
        """
        等待其他进程完成下载
        
        Returns:
            bool: 其他进程是否成功完成下载
        """
        logger.info("Waiting for another process to complete download...")
        max_wait = 300  # 5分钟最大等待
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            await asyncio.sleep(10)  # 每10秒检查一次
            
            # 检查下载状态文件是否还存在
            if not self.download_status_file.exists() and not self.lock_file_path.exists():
                # 检查模型是否已完成
                if self.is_server_model_complete():
                    logger.info("Another process completed download successfully")
                    return True
                else:
                    logger.warning("Another process finished but model incomplete")
                    return False
            
            logger.debug("Still waiting for other process...")
        
        logger.warning("Timeout waiting for other process")
        return False
    
    async def _perform_model_download(self) -> bool:
        """
        执行实际的模型下载
        
        Returns:
            bool: 下载是否成功
        """
        try:
            logger.info("Starting model download process")
            
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
            logger.error(f"Model download failed: {e}")
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