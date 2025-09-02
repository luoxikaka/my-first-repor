"""
销售智能问答知识培训智能体 - 主应用程序
提供RESTful API接口，支持多模态输入和智能对话
"""

import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.multimodal_agent import (
    multimodal_agent, 
    chat_with_agent, 
    MultiModalInput, 
    AgentResponse
)
from app.rag_system import rag_system, initialize_rag_system
from app.mcp_server import process_multimodal_input

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="销售智能问答知识培训智能体",
    description="基于LangChain的多模态销售智能问答系统，集成MCP、RAG、Agent、LangGraph等技术",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 静态文件服务
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 知识库上传目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)


# API模型定义
class ChatRequest(BaseModel):
    """聊天请求模型"""
    text: Optional[str] = Field(default=None, description="文本消息")
    image_base64: Optional[str] = Field(default=None, description="Base64编码的图像")
    audio_base64: Optional[str] = Field(default=None, description="Base64编码的音频")
    video_base64: Optional[str] = Field(default=None, description="Base64编码的视频")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    user_profile: Optional[Dict[str, Any]] = Field(default={}, description="用户画像")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[AgentResponse] = Field(default=None, description="响应数据")
    error: Optional[str] = Field(default=None, description="错误信息")


class KnowledgeUploadResponse(BaseModel):
    """知识库上传响应模型"""
    success: bool = Field(..., description="是否成功")
    file_id: Optional[str] = Field(default=None, description="文件ID")
    message: str = Field(..., description="响应消息")


class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str = Field(..., description="系统状态")
    components: Dict[str, str] = Field(..., description="组件状态")
    timestamp: str = Field(..., description="时间戳")


# 全局变量
system_initialized = False


async def initialize_system():
    """初始化系统"""
    global system_initialized
    
    if system_initialized:
        return
    
    try:
        logger.info("开始初始化系统...")
        
        # 初始化RAG系统
        await initialize_rag_system(str(KNOWLEDGE_BASE_DIR))
        
        system_initialized = True
        logger.info("系统初始化完成")
        
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    await initialize_system()


@app.get("/", response_class=JSONResponse)
async def root():
    """根路径"""
    return {
        "message": "销售智能问答知识培训智能体 API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """健康检查"""
    try:
        components = {
            "api": "healthy",
            "rag_system": "healthy" if system_initialized else "initializing",
            "multimodal_agent": "healthy",
            "mcp_server": "healthy"
        }
        
        return SystemStatus(
            status="healthy",
            components=components,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天接口"""
    try:
        if not system_initialized:
            await initialize_system()
        
        # 调用多模态智能代理
        response = await chat_with_agent(
            text=request.text,
            image=request.image_base64,
            audio=request.audio_base64,
            video=request.video_base64,
            session_id=request.session_id,
            user_profile=request.user_profile,
            metadata=request.metadata
        )
        
        return ChatResponse(
            success=True,
            data=response
        )
        
    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        return ChatResponse(
            success=False,
            error=str(e)
        )


@app.post("/upload_knowledge", response_model=KnowledgeUploadResponse)
async def upload_knowledge(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    content_type: str = Form(default="auto"),
    metadata: str = Form(default="{}")
):
    """上传知识库文件"""
    try:
        if not system_initialized:
            await initialize_system()
        
        # 生成文件ID
        file_id = str(uuid.uuid4())
        
        # 保存文件
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 解析元数据
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # 后台任务：添加到知识库
        background_tasks.add_task(
            add_to_knowledge_base,
            str(file_path),
            content_type,
            metadata_dict
        )
        
        return KnowledgeUploadResponse(
            success=True,
            file_id=file_id,
            message=f"文件 {file.filename} 上传成功，正在处理中..."
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return KnowledgeUploadResponse(
            success=False,
            message=f"文件上传失败: {str(e)}"
        )


async def add_to_knowledge_base(file_path: str, content_type: str, metadata: Dict[str, Any]):
    """添加文件到知识库（后台任务）"""
    try:
        await rag_system.knowledge_base.add_document(file_path, content_type)
        logger.info(f"文件已添加到知识库: {file_path}")
    except Exception as e:
        logger.error(f"添加文件到知识库失败: {e}")


@app.post("/upload_multimodal")
async def upload_multimodal_content(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    content_type: str = Form(...),
    source: str = Form(...),
    metadata: str = Form(default="{}")
):
    """上传多模态内容"""
    try:
        if not system_initialized:
            await initialize_system()
        
        # 读取文件内容
        content = await file.read()
        
        # 解析元数据
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # 添加到知识库
        item_id = await rag_system.add_knowledge(
            content=content,
            content_type=content_type,
            source=source,
            metadata=metadata_dict
        )
        
        return {
            "success": True,
            "item_id": item_id,
            "message": "多模态内容上传成功"
        }
        
    except Exception as e:
        logger.error(f"多模态内容上传失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/conversation_history/{session_id}")
async def get_conversation_history(session_id: str):
    """获取对话历史"""
    try:
        history = multimodal_agent.get_conversation_history(session_id)
        return {
            "success": True,
            "session_id": session_id,
            "history": history
        }
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/update_user_profile/{session_id}")
async def update_user_profile(session_id: str, profile_updates: Dict[str, Any]):
    """更新用户画像"""
    try:
        multimodal_agent.update_user_profile(session_id, profile_updates)
        return {
            "success": True,
            "message": "用户画像更新成功"
        }
    except Exception as e:
        logger.error(f"更新用户画像失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/clear_session/{session_id}")
async def clear_session(session_id: str):
    """清除会话"""
    try:
        multimodal_agent.clear_session(session_id)
        return {
            "success": True,
            "message": "会话清除成功"
        }
    except Exception as e:
        logger.error(f"清除会话失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/process_multimodal")
async def process_multimodal_endpoint(
    content_base64: str = Form(...),
    content_type: str = Form(...),
    metadata: str = Form(default="{}")
):
    """处理多模态内容"""
    try:
        # 解码内容
        content = base64.b64decode(content_base64)
        
        # 解析元数据
        try:
            metadata_dict = json.loads(metadata)
        except:
            metadata_dict = {}
        
        # 处理内容
        response = await process_multimodal_input(
            content=content,
            content_type=content_type,
            metadata=metadata_dict
        )
        
        return {
            "success": True,
            "data": {
                "processed_content": response.processed_content,
                "content_type": response.content_type,
                "extracted_features": response.extracted_features,
                "confidence": response.confidence
            }
        }
        
    except Exception as e:
        logger.error(f"多模态内容处理失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/knowledge_stats")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    try:
        # 这里可以添加知识库统计逻辑
        stats = {
            "total_documents": 0,  # 实际实现时从数据库获取
            "total_chunks": 0,
            "content_types": {},
            "last_updated": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"获取知识库统计失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# 错误处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "内部服务器错误",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    # 开发环境运行
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )