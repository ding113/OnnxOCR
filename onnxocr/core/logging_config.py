"""
📋 Logging Configuration Module

Modern logging setup for ONNX OCR service with structured logging and intelligent level management.

Features:
- Environment-based log level configuration
- Structured JSON logging with contextual information
- Production-ready defaults with debug fallback
- Request correlation ID support
- Performance-optimized configuration
"""

import logging
import os
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any

import structlog


# Request correlation context variable
REQUEST_ID_CONTEXT: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_log_level() -> str:
    """
    Get effective log level from environment with intelligent defaults.
    
    Returns:
        Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Default to INFO for production, DEBUG for development
    default_level = "INFO" if os.getenv("ENVIRONMENT", "development") == "production" else "DEBUG"
    return os.getenv("LOG_LEVEL", default_level).upper()


def add_request_id(logger, method_name, event_dict):
    """
    Add request ID from context to log entries.
    
    Args:
        logger: Logger instance
        method_name: Method name
        event_dict: Event dictionary
        
    Returns:
        Updated event dictionary with request_id
    """
    request_id = REQUEST_ID_CONTEXT.get()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def setup_logging(
    log_level: Optional[str] = None,
    enable_json: bool = True,
    enable_request_id: bool = True
) -> structlog.BoundLogger:
    """
    Configure comprehensive logging system for OCR service.
    
    Args:
        log_level: Override log level (default: from environment)
        enable_json: Enable JSON output format (default: True)
        enable_request_id: Enable request ID tracking (default: True)
        
    Returns:
        Configured structlog logger instance
    """
    # Determine effective log level
    effective_level = log_level or get_log_level()
    
    # Configure Python standard logging
    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format='%(message)s',  # structlog handles detailed formatting
        force=True  # Override any existing configuration
    )
    
    # Ensure root logger is properly configured
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, effective_level, logging.INFO))
    
    # Build processor chain
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add request ID processor if enabled
    if enable_request_id:
        processors.insert(-1, add_request_id)
    
    # Add JSON renderer for production or structured output
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Create and return logger
    logger = structlog.get_logger()
    
    # Log configuration info (only in non-JSON mode for readability)
    if not enable_json:
        print(f"[LOGGING] System configured: level={effective_level}, json={enable_json}, request_id={enable_request_id}")
    
    # Test log levels in DEBUG mode only
    if effective_level == "DEBUG":
        logger.debug("🔍 Debug logging enabled - detailed trace information available")
        logger.info("ℹ️ Info logging active - general operational messages")
        logger.warning("⚠️ Warning logging active - attention-worthy events")
    
    return logger


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID for current context.
    
    Args:
        request_id: Optional custom request ID (generates UUID4 if None)
        
    Returns:
        The request ID that was set
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]  # Short UUID for readability
    
    REQUEST_ID_CONTEXT.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """
    Get current request ID from context.
    
    Returns:
        Current request ID or None if not set
    """
    return REQUEST_ID_CONTEXT.get()


def clear_request_id() -> None:
    """Clear request ID from current context."""
    REQUEST_ID_CONTEXT.set(None)


def create_contextual_logger(
    name: str,
    extra_context: Optional[Dict[str, Any]] = None
) -> structlog.BoundLogger:
    """
    Create a logger with pre-bound context.
    
    Args:
        name: Logger name/component identifier
        extra_context: Additional context to bind to logger
        
    Returns:
        Contextual logger with bound information
    """
    logger = structlog.get_logger(name)
    
    # Bind extra context if provided
    if extra_context:
        logger = logger.bind(**extra_context)
    
    return logger


# Production-ready log level mappings
LOG_LEVEL_MAPPING = {
    "CRITICAL": logging.CRITICAL,  # System failure, immediate attention
    "ERROR": logging.ERROR,        # Error occurred, functionality impacted
    "WARNING": logging.WARNING,    # Warning condition, but service continues
    "INFO": logging.INFO,          # General operational messages
    "DEBUG": logging.DEBUG,        # Detailed diagnostic information
}


def get_performance_logger() -> structlog.BoundLogger:
    """
    Get logger optimized for performance metrics.
    
    Returns:
        Performance-focused logger with component context
    """
    return create_contextual_logger(
        "performance",
        {"component": "metrics", "category": "performance"}
    )


def get_security_logger() -> structlog.BoundLogger:
    """
    Get logger optimized for security events.
    
    Returns:
        Security-focused logger with component context
    """
    return create_contextual_logger(
        "security",
        {"component": "security", "category": "audit"}
    )


def get_api_logger() -> structlog.BoundLogger:
    """
    Get logger optimized for API operations.
    
    Returns:
        API-focused logger with component context
    """
    return create_contextual_logger(
        "api",
        {"component": "api", "category": "request"}
    )


# Convenience function for backwards compatibility
def get_logger(name: str = "onnxocr") -> structlog.BoundLogger:
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Logger name/component
        
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


# Export commonly used items
__all__ = [
    "setup_logging",
    "set_request_id", 
    "get_request_id",
    "clear_request_id",
    "create_contextual_logger",
    "get_logger",
    "get_performance_logger",
    "get_security_logger", 
    "get_api_logger",
    "LOG_LEVEL_MAPPING",
]