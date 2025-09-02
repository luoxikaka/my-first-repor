"""
工具函数模块
提供系统中使用的各种工具函数和辅助方法
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mimetypes

import aiofiles
import numpy as np
from PIL import Image
import magic


# 配置日志
logger = logging.getLogger(__name__)


class FileUtils:
    """文件处理工具类"""
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def get_file_mime_type(file_path: Union[str, Path]) -> str:
        """获取文件的MIME类型"""
        try:
            mime = magic.Magic(mime=True)
            return mime.from_file(str(file_path))
        except:
            # 回退到基于扩展名的检测
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type or "application/octet-stream"
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """获取文件大小（字节）"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """确保目录存在，如果不存在则创建"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """生成安全的文件名"""
        # 移除或替换不安全的字符
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
        # 限制长度
        if len(safe_name) > 255:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:255-len(ext)] + ext
        return safe_name
    
    @staticmethod
    async def save_uploaded_file(file_content: bytes, 
                               filename: str, 
                               upload_dir: Union[str, Path]) -> Path:
        """异步保存上传的文件"""
        upload_path = FileUtils.ensure_directory(upload_dir)
        safe_name = FileUtils.safe_filename(filename)
        
        # 如果文件已存在，添加时间戳
        file_path = upload_path / safe_name
        if file_path.exists():
            name, ext = os.path.splitext(safe_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = f"{name}_{timestamp}{ext}"
            file_path = upload_path / safe_name
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        return file_path
    
    @staticmethod
    def is_image_file(file_path: Union[str, Path]) -> bool:
        """检查是否为图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        return Path(file_path).suffix.lower() in image_extensions
    
    @staticmethod
    def is_audio_file(file_path: Union[str, Path]) -> bool:
        """检查是否为音频文件"""
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    @staticmethod
    def is_video_file(file_path: Union[str, Path]) -> bool:
        """检查是否为视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
        return Path(file_path).suffix.lower() in video_extensions
    
    @staticmethod
    def is_document_file(file_path: Union[str, Path]) -> bool:
        """检查是否为文档文件"""
        doc_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md', '.rtf', '.odt'}
        return Path(file_path).suffix.lower() in doc_extensions


class TextUtils:
    """文本处理工具类"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本，移除多余的空白字符"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空白
        text = text.strip()
        return text
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
        """截断文本到指定长度"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取（实际应用中可以使用更复杂的算法）
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词（简化版）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        # 统计词频
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """计算两个文本的相似度（简化版）"""
        if not text1 or not text2:
            return 0.0
        
        # 转换为小写并分词
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def format_response_text(text: str) -> str:
        """格式化响应文本"""
        # 确保段落之间有适当的间距
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行首尾的空白
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)


class EncodingUtils:
    """编码工具类"""
    
    @staticmethod
    def encode_base64(data: Union[str, bytes]) -> str:
        """Base64编码"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def decode_base64(encoded_data: str) -> bytes:
        """Base64解码"""
        return base64.b64decode(encoded_data)
    
    @staticmethod
    def encode_image_to_base64(image_path: Union[str, Path]) -> str:
        """将图像文件编码为Base64"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return EncodingUtils.encode_base64(image_data)
    
    @staticmethod
    def decode_base64_to_image(base64_data: str, output_path: Union[str, Path]) -> Path:
        """将Base64数据解码为图像文件"""
        image_data = EncodingUtils.decode_base64(base64_data)
        output_path = Path(output_path)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return output_path


class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """验证手机号格式（中国）"""
        pattern = r'^1[3-9]\d{9}$'
        return re.match(pattern, phone) is not None
    
    @staticmethod
    def validate_file_size(file_size: int, max_size: int) -> bool:
        """验证文件大小"""
        return 0 < file_size <= max_size
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
        """验证文件扩展名"""
        ext = Path(filename).suffix.lower()
        return ext in allowed_extensions
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """清理用户输入，防止XSS等攻击"""
        if not text:
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除JavaScript
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        # 移除其他潜在危险字符
        text = re.sub(r'[<>"\']', '', text)
        
        return text.strip()


class TimeUtils:
    """时间工具类"""
    
    @staticmethod
    def get_current_timestamp() -> str:
        """获取当前时间戳（ISO格式）"""
        return datetime.now().isoformat()
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
    
    @staticmethod
    def is_recent(timestamp: datetime, hours: int = 24) -> bool:
        """检查时间戳是否在最近N小时内"""
        now = datetime.now()
        time_diff = now - timestamp
        return time_diff <= timedelta(hours=hours)
    
    @staticmethod
    def get_time_ago(timestamp: datetime) -> str:
        """获取相对时间描述"""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days}天前"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}小时前"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}分钟前"
        else:
            return "刚刚"


class IDUtils:
    """ID生成工具类"""
    
    @staticmethod
    def generate_uuid() -> str:
        """生成UUID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """生成短ID"""
        return str(uuid.uuid4()).replace('-', '')[:length]
    
    @staticmethod
    def generate_session_id() -> str:
        """生成会话ID"""
        timestamp = int(time.time())
        random_part = IDUtils.generate_short_id(8)
        return f"session_{timestamp}_{random_part}"
    
    @staticmethod
    def generate_request_id() -> str:
        """生成请求ID"""
        timestamp = int(time.time() * 1000)  # 毫秒级时间戳
        random_part = IDUtils.generate_short_id(6)
        return f"req_{timestamp}_{random_part}"


class ConfigUtils:
    """配置工具类"""
    
    @staticmethod
    def load_config_from_env(prefix: str = "") -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            # 移除前缀
            config_key = key[len(prefix):] if prefix else key
            config_key = config_key.lower()
            
            # 尝试转换数据类型
            if value.lower() in ('true', 'false'):
                config[config_key] = value.lower() == 'true'
            elif value.isdigit():
                config[config_key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                config[config_key] = float(value)
            else:
                config[config_key] = value
        
        return config
    
    @staticmethod
    def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
        """获取环境变量"""
        value = os.environ.get(key, default)
        
        if required and value is None:
            raise ValueError(f"Required environment variable {key} is not set")
        
        return value


class LogUtils:
    """日志工具类"""
    
    @staticmethod
    def setup_logger(name: str, 
                    level: str = "INFO", 
                    log_file: Optional[str] = None) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if logger.handlers:
            return logger
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_function_call(func):
        """装饰器：记录函数调用"""
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper


class PerformanceUtils:
    """性能工具类"""
    
    @staticmethod
    def measure_time(func):
        """装饰器：测量函数执行时间"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        
        return wrapper
    
    @staticmethod
    async def measure_async_time(func):
        """装饰器：测量异步函数执行时间"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        
        return wrapper
    
    @staticmethod
    def batch_process(items: List[Any], 
                     batch_size: int = 10, 
                     delay: float = 0.1) -> List[List[Any]]:
        """将列表分批处理"""
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
            
            # 添加延迟以避免过载
            if delay > 0 and i + batch_size < len(items):
                time.sleep(delay)
        
        return batches


class SecurityUtils:
    """安全工具类"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密码"""
        import hashlib
        import secrets
        
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return salt + password_hash.hex()
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """验证密码"""
        import hashlib
        
        salt = hashed[:32]
        stored_hash = hashed[32:]
        
        password_hash = hashlib.pbkdf2_hmac('sha256',
                                          password.encode('utf-8'),
                                          salt.encode('utf-8'),
                                          100000)
        
        return password_hash.hex() == stored_hash
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """生成API密钥"""
        import secrets
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
        """掩码敏感数据"""
        if len(data) <= visible_chars * 2:
            return mask_char * len(data)
        # 导出所有工具类
__all__ = [
    'FileUtils',
    'TextUtils', 
    'EncodingUtils',
    'ValidationUtils',
    'TimeUtils',
    'IDUtils',
    'ConfigUtils',
    'LogUtils',
    'PerformanceUtils',
    'SecurityUtils'
]
        start = data[:visible_chars]
        end = data[-visible_chars:]
        middle = mask_char * (len(data) - visible_chars * 2)
        
        return start + middle + end


