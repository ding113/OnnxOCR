"""
统一日志配置
支持等级控制、请求ID追踪、结构化输出
"""
import logging
import logging.config
from .settings import settings


def setup_logging():
    """配置应用日志"""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': settings.LOG_FORMAT,
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'access': {
                'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout',
            },
        },
        'root': {
            'level': settings.LOG_LEVEL,
            'handlers': ['console'],
        },
        'loggers': {
            'uvicorn.access': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn.error': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'fastapi': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'app': {
                'handlers': ['console'],
                'level': settings.LOG_LEVEL,
                'propagate': False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """获取命名logger"""
    return logging.getLogger(name)