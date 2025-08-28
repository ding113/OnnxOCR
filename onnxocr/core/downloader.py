"""
Model downloader with async support and progress monitoring
"""

import asyncio
import hashlib
import aiofiles
import httpx
import structlog
from pathlib import Path
from typing import Optional, Dict, Callable, Any
from urllib.parse import urlparse

from .exceptions import DownloadError

logger = structlog.get_logger()


class ModelDownloader:
    """
    Async model downloader with modern features:
    - httpx for async HTTP requests
    - aiofiles for async file operations  
    - Progress monitoring and callbacks
    - Resume/retry capability
    - Integrity verification
    """
    
    def __init__(
        self,
        timeout: int = 300,
        retry_attempts: int = 3,
        chunk_size: int = 8192
    ):
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.chunk_size = chunk_size
        
        self.logger = logger.bind(component="ModelDownloader")
        
        # HTTP client configuration
        self.client_config = {
            "timeout": httpx.Timeout(timeout),
            "follow_redirects": True,
            "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10)
        }
    
    async def download_model(
        self,
        url: str,
        target_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verify_integrity: bool = True,
        expected_size: Optional[int] = None
    ) -> bool:
        """
        Download a model file with progress monitoring
        
        Args:
            url: Download URL
            target_path: Local file path to save
            progress_callback: Optional callback for progress updates
            verify_integrity: Whether to verify file integrity
            expected_size: Expected file size for validation
            
        Returns:
            bool: True if download succeeded, False otherwise
        """
        self.logger.info(
            "Starting model download",
            url=url,
            target_path=str(target_path),
            expected_size=expected_size
        )
        
        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists and is complete
        if target_path.exists():
            if expected_size and target_path.stat().st_size == expected_size:
                self.logger.info("Model file already exists and is complete")
                return True
            else:
                self.logger.info("Existing file incomplete or size unknown, re-downloading")
        
        for attempt in range(self.retry_attempts):
            try:
                success = await self._download_with_resume(
                    url=url,
                    target_path=target_path,
                    progress_callback=progress_callback,
                    expected_size=expected_size
                )
                
                if success and verify_integrity:
                    integrity_ok = await self._verify_file_integrity(
                        target_path, 
                        expected_size
                    )
                    if not integrity_ok:
                        self.logger.warning("File integrity check failed, retrying")
                        continue
                
                if success:
                    self.logger.info("Model download completed successfully")
                    return True
                    
            except Exception as e:
                self.logger.warning(
                    "Download attempt failed",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise DownloadError(
                        f"Failed to download model after {self.retry_attempts} attempts: {e}",
                        url=url
                    ) from e
        
        return False
    
    async def _download_with_resume(
        self,
        url: str,
        target_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        expected_size: Optional[int] = None
    ) -> bool:
        """
        Download file with resume capability
        """
        resume_pos = 0
        if target_path.exists():
            resume_pos = target_path.stat().st_size
        
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
            self.logger.info("Resuming download", resume_position=resume_pos)
        
        async with httpx.AsyncClient(**self.client_config) as client:
            try:
                response = await client.get(url, headers=headers)
                
                if response.status_code not in (200, 206):
                    raise DownloadError(
                        f"HTTP error {response.status_code}",
                        url=url,
                        status_code=response.status_code
                    )
                
                # Get content length
                content_length = response.headers.get('content-length')
                if content_length:
                    total_size = int(content_length) + resume_pos
                else:
                    total_size = expected_size or 0
                
                # Open file for writing (append mode if resuming)
                mode = 'ab' if resume_pos > 0 else 'wb'
                downloaded = resume_pos
                
                async with aiofiles.open(target_path, mode) as f:
                    async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
                
                return True
                
            except httpx.HTTPError as e:
                raise DownloadError(f"HTTP error: {e}", url=url) from e
    
    async def _verify_file_integrity(
        self,
        file_path: Path,
        expected_size: Optional[int] = None
    ) -> bool:
        """
        Verify downloaded file integrity
        """
        if not file_path.exists():
            return False
        
        file_size = file_path.stat().st_size
        
        # Check size if expected size is provided
        if expected_size and file_size != expected_size:
            self.logger.warning(
                "File size mismatch",
                expected=expected_size,
                actual=file_size
            )
            return False
        
        # Basic integrity check - ensure file is not empty and can be read
        if file_size == 0:
            return False
        
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                # Read first and last chunk to verify file is readable
                await f.read(min(1024, file_size))
                if file_size > 1024:
                    await f.seek(-1024, 2)  # Seek from end
                    await f.read(1024)
            return True
        except Exception as e:
            self.logger.error("File integrity check failed", error=str(e))
            return False
    
    async def get_remote_file_info(self, url: str) -> Dict[str, Any]:
        """
        Get remote file information without downloading
        """
        async with httpx.AsyncClient(**self.client_config) as client:
            try:
                response = await client.head(url)
                response.raise_for_status()
                
                return {
                    "size": int(response.headers.get('content-length', 0)),
                    "content_type": response.headers.get('content-type'),
                    "last_modified": response.headers.get('last-modified'),
                    "etag": response.headers.get('etag'),
                    "supports_range": "accept-ranges" in response.headers
                }
            except httpx.HTTPError as e:
                raise DownloadError(f"Failed to get file info: {e}", url=url) from e
    
    def create_progress_callback(self, description: str) -> Callable[[int, int], None]:
        """
        Create a progress callback that logs download progress
        """
        last_percent = -1
        
        def progress_callback(downloaded: int, total: int):
            nonlocal last_percent
            if total > 0:
                percent = int(downloaded * 100 / total)
                if percent != last_percent and percent % 10 == 0:  # Log every 10%
                    last_percent = percent
                    self.logger.info(
                        f"Download progress: {description}",
                        downloaded=downloaded,
                        total=total,
                        percent=percent
                    )
        
        return progress_callback