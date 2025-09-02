"""
Model Context Protocol (MCP) Server Implementation
处理多模态输入数据，包括文本、图像、音频等，并将其转换为LLM可理解的格式
"""

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import cv2
import librosa
import numpy as np
import speech_recognition as sr
import whisper
from PIL import Image
from pydantic import BaseModel, Field
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """MCP请求模型"""
    request_id: str = Field(..., description="请求唯一标识符")
    content_type: str = Field(..., description="内容类型: text, image, audio, video")
    content: Union[str, bytes] = Field(..., description="内容数据")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据")


class MCPResponse(BaseModel):
    """MCP响应模型"""
    request_id: str = Field(..., description="对应的请求ID")
    processed_content: str = Field(..., description="处理后的文本内容")
    content_type: str = Field(..., description="原始内容类型")
    extracted_features: Optional[Dict[str, Any]] = Field(default={}, description="提取的特征")
    confidence: float = Field(default=1.0, description="处理置信度")
    error: Optional[str] = Field(default=None, description="错误信息")


class MultiModalProcessor:
    """多模态处理器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 初始化图像处理模型
        self._init_image_models()
        
        # 初始化音频处理模型
        self._init_audio_models()
        
        # 语音识别器
        self.speech_recognizer = sr.Recognizer()
        
    def _init_image_models(self):
        """初始化图像处理模型"""
        try:
            # BLIP模型用于图像描述
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            logger.info("BLIP图像描述模型加载成功")
        except Exception as e:
            logger.error(f"图像模型初始化失败: {e}")
            self.blip_processor = None
            self.blip_model = None
    
    def _init_audio_models(self):
        """初始化音频处理模型"""
        try:
            # Whisper模型用于语音识别
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper语音识别模型加载成功")
        except Exception as e:
            logger.error(f"音频模型初始化失败: {e}")
            self.whisper_model = None

    async def process_text(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理文本内容"""
        try:
            # 基本文本处理
            processed_text = content.strip()
            
            # 提取文本特征
            features = {
                "length": len(processed_text),
                "word_count": len(processed_text.split()),
                "language": metadata.get("language", "zh"),
                "encoding": metadata.get("encoding", "utf-8")
            }
            
            return {
                "processed_content": processed_text,
                "extracted_features": features,
                "confidence": 1.0
            }
        except Exception as e:
            logger.error(f"文本处理错误: {e}")
            return {
                "processed_content": content,
                "extracted_features": {},
                "confidence": 0.5,
                "error": str(e)
            }

    async def process_image(self, content: Union[str, bytes], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像内容"""
        try:
            # 解码图像
            if isinstance(content, str):
                # Base64编码的图像
                image_data = base64.b64decode(content)
            else:
                image_data = content
            
            # 加载图像
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # 图像基本信息
            width, height = image.size
            
            # 使用BLIP生成图像描述
            description = ""
            if self.blip_model and self.blip_processor:
                inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
                out = self.blip_model.generate(**inputs, max_length=100)
                description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # 图像特征提取
            features = {
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "format": metadata.get("format", "unknown"),
                "size_bytes": len(image_data),
                "description": description
            }
            
            # 构建处理后的文本内容
            processed_content = f"图像描述: {description}\n"
            processed_content += f"图像尺寸: {width}x{height}\n"
            processed_content += f"图像格式: {metadata.get('format', 'unknown')}"
            
            return {
                "processed_content": processed_content,
                "extracted_features": features,
                "confidence": 0.9 if description else 0.7
            }
            
        except Exception as e:
            logger.error(f"图像处理错误: {e}")
            return {
                "processed_content": "图像处理失败",
                "extracted_features": {},
                "confidence": 0.0,
                "error": str(e)
            }

    async def process_audio(self, content: Union[str, bytes], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频内容"""
        try:
            # 解码音频
            if isinstance(content, str):
                audio_data = base64.b64decode(content)
            else:
                audio_data = content
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # 使用Whisper进行语音识别
                transcription = ""
                if self.whisper_model:
                    result = self.whisper_model.transcribe(temp_path)
                    transcription = result["text"]
                
                # 音频特征提取
                y, sr = librosa.load(temp_path)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # 提取音频特征
                features = {
                    "duration": duration,
                    "sample_rate": sr,
                    "channels": 1,  # librosa默认单声道
                    "transcription": transcription,
                    "format": metadata.get("format", "wav")
                }
                
                # 构建处理后的文本内容
                processed_content = f"音频转录: {transcription}\n"
                processed_content += f"音频时长: {duration:.2f}秒\n"
                processed_content += f"采样率: {sr}Hz"
                
                return {
                    "processed_content": processed_content,
                    "extracted_features": features,
                    "confidence": 0.8 if transcription else 0.5
                }
                
            finally:
                # 清理临时文件
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"音频处理错误: {e}")
            return {
                "processed_content": "音频处理失败",
                "extracted_features": {},
                "confidence": 0.0,
                "error": str(e)
            }

    async def process_video(self, content: Union[str, bytes], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """处理视频内容"""
        try:
            # 解码视频
            if isinstance(content, str):
                video_data = base64.b64decode(content)
            else:
                video_data = content
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name
            
            try:
                # 使用OpenCV读取视频
                cap = cv2.VideoCapture(temp_path)
                
                # 获取视频基本信息
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # 提取关键帧进行图像分析
                key_frames = []
                frame_descriptions = []
                
                # 每秒提取一帧
                interval = max(1, int(fps))
                for i in range(0, frame_count, interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        # 转换为PIL图像
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # 使用BLIP分析帧
                        if self.blip_model and self.blip_processor:
                            inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
                            out = self.blip_model.generate(**inputs, max_length=50)
                            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
                            frame_descriptions.append(f"第{i//interval}秒: {description}")
                
                cap.release()
                
                # 视频特征
                features = {
                    "duration": duration,
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height > 0 else 1,
                    "format": metadata.get("format", "mp4"),
                    "frame_descriptions": frame_descriptions
                }
                
                # 构建处理后的文本内容
                processed_content = f"视频时长: {duration:.2f}秒\n"
                processed_content += f"视频尺寸: {width}x{height}\n"
                processed_content += f"帧率: {fps:.2f}fps\n"
                processed_content += "关键帧描述:\n" + "\n".join(frame_descriptions)
                
                return {
                    "processed_content": processed_content,
                    "extracted_features": features,
                    "confidence": 0.8 if frame_descriptions else 0.6
                }
                
            finally:
                # 清理临时文件
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"视频处理错误: {e}")
            return {
                "processed_content": "视频处理失败",
                "extracted_features": {},
                "confidence": 0.0,
                "error": str(e)
            }


class MCPServer:
    """MCP服务器主类"""
    
    def __init__(self):
        self.processor = MultiModalProcessor()
        logger.info("MCP服务器初始化完成")
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """处理MCP请求"""
        try:
            logger.info(f"处理请求 {request.request_id}, 类型: {request.content_type}")
            
            # 根据内容类型选择处理方法
            if request.content_type == "text":
                result = await self.processor.process_text(request.content, request.metadata)
            elif request.content_type == "image":
                result = await self.processor.process_image(request.content, request.metadata)
            elif request.content_type == "audio":
                result = await self.processor.process_audio(request.content, request.metadata)
            elif request.content_type == "video":
                result = await self.processor.process_video(request.content, request.metadata)
            else:
                raise ValueError(f"不支持的内容类型: {request.content_type}")
            
            # 构建响应
            response = MCPResponse(
                request_id=request.request_id,
                processed_content=result["processed_content"],
                content_type=request.content_type,
                extracted_features=result["extracted_features"],
                confidence=result["confidence"],
                error=result.get("error")
            )
            
            logger.info(f"请求 {request.request_id} 处理完成，置信度: {response.confidence}")
            return response
            
        except Exception as e:
            logger.error(f"处理请求 {request.request_id} 时发生错误: {e}")
            return MCPResponse(
                request_id=request.request_id,
                processed_content="处理失败",
                content_type=request.content_type,
                extracted_features={},
                confidence=0.0,
                error=str(e)
            )

    async def batch_process(self, requests: List[MCPRequest]) -> List[MCPResponse]:
        """批量处理请求"""
        tasks = [self.process_request(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                processed_responses.append(MCPResponse(
                    request_id=requests[i].request_id,
                    processed_content="批量处理失败",
                    content_type=requests[i].content_type,
                    extracted_features={},
                    confidence=0.0,
                    error=str(response)
                ))
            else:
                processed_responses.append(response)
        
        return processed_responses


# 全局MCP服务器实例
mcp_server = MCPServer()


async def process_multimodal_input(
    content: Union[str, bytes],
    content_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> MCPResponse:
    """
    处理多模态输入的便捷函数
    
    Args:
        content: 输入内容
        content_type: 内容类型 (text, image, audio, video)
        metadata: 元数据
        request_id: 请求ID
    
    Returns:
        MCPResponse: 处理结果
    """
    if request_id is None:
        import uuid
        request_id = str(uuid.uuid4())
    
    if metadata is None:
        metadata = {}
    
    request = MCPRequest(
        request_id=request_id,
        content_type=content_type,
        content=content,
        metadata=metadata
    )
    
    return await mcp_server.process_request(request)


if __name__ == "__main__":
    # 测试代码
    async def test_mcp():
        # 测试文本处理
        text_response = await process_multimodal_input(
            content="这是一个销售培训的测试文本",
            content_type="text"
        )
        print("文本处理结果:", text_response.processed_content)
        
        # 测试图像处理（需要实际图像数据）
        # image_response = await process_multimodal_input(
        #     content=image_bytes,
        #     content_type="image",
        #     metadata={"format": "png"}
        # )
        # print("图像处理结果:", image_response.processed_content)
    
    asyncio.run(test_mcp())