"""
多模态智能代理实现
集成文本、图像、音频处理能力，提供统一的智能交互接口
"""

import asyncio
import base64
import io
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field

from .mcp_server import process_multimodal_input, MCPResponse
from .rag_system import rag_system, RAGResponse
from .agent_workflow import run_sales_agent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalInput(BaseModel):
    """多模态输入模型"""
    text: Optional[str] = Field(default=None, description="文本内容")
    image: Optional[str] = Field(default=None, description="Base64编码的图像")
    audio: Optional[str] = Field(default=None, description="Base64编码的音频")
    video: Optional[str] = Field(default=None, description="Base64编码的视频")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class AgentResponse(BaseModel):
    """智能代理响应模型"""
    text_response: str = Field(..., description="文本响应")
    confidence: float = Field(..., description="置信度")
    reasoning: List[str] = Field(..., description="推理过程")
    sources: List[Dict[str, Any]] = Field(default=[], description="信息来源")
    multimodal_analysis: Dict[str, Any] = Field(default={}, description="多模态分析结果")
    suggestions: List[str] = Field(default=[], description="建议")
    session_id: str = Field(..., description="会话ID")
    timestamp: str = Field(..., description="时间戳")


class ConversationContext(BaseModel):
    """对话上下文模型"""
    session_id: str = Field(..., description="会话ID")
    user_profile: Dict[str, Any] = Field(default={}, description="用户画像")
    conversation_history: List[Dict[str, Any]] = Field(default=[], description="对话历史")
    current_topic: Optional[str] = Field(default=None, description="当前话题")
    multimodal_context: Dict[str, Any] = Field(default={}, description="多模态上下文")
    preferences: Dict[str, Any] = Field(default={}, description="用户偏好")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


class MultiModalAgent:
    """多模态智能代理"""
    
    def __init__(self, 
                 model_name: str = "gpt-4-vision-preview",
                 max_history: int = 10):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.1,
            max_tokens=2000
        )
        
        # 对话记忆
        self.memory = ConversationBufferWindowMemory(
            k=max_history,
            return_messages=True
        )
        
        # 会话上下文存储
        self.contexts: Dict[str, ConversationContext] = {}
        
        # 系统提示
        self.system_prompt = """
        你是一个专业的销售培训智能助手，具备以下能力：
        
        1. 多模态理解：能够理解和分析文本、图像、音频、视频内容
        2. 销售知识：掌握丰富的销售技巧、产品知识、客户管理等专业知识
        3. 个性化服务：根据用户的角色、经验水平提供针对性的建议
        4. 实时互动：支持连续对话，记住上下文信息
        
        你的目标是帮助销售人员提升专业技能，解答疑问，提供实用建议。
        
        回答时请：
        - 保持专业和友好的语调
        - 提供具体、可操作的建议
        - 引用相关的知识来源
        - 根据用户的具体情况个性化回答
        - 如果涉及多模态内容，要充分利用分析结果
        """
        
        logger.info("多模态智能代理初始化完成")
    
    def get_or_create_context(self, session_id: str) -> ConversationContext:
        """获取或创建会话上下文"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
        return self.contexts[session_id]
    
    async def process_multimodal_input(self, 
                                     input_data: MultiModalInput,
                                     session_id: str) -> Dict[str, Any]:
        """处理多模态输入"""
        context = self.get_or_create_context(session_id)
        processed_content = {}
        
        try:
            # 处理文本
            if input_data.text:
                processed_content["text"] = {
                    "content": input_data.text,
                    "type": "text",
                    "processed": input_data.text
                }
            
            # 处理图像
            if input_data.image:
                image_response = await process_multimodal_input(
                    content=input_data.image,
                    content_type="image",
                    metadata=input_data.metadata
                )
                processed_content["image"] = {
                    "content": input_data.image,
                    "type": "image",
                    "processed": image_response.processed_content,
                    "features": image_response.extracted_features,
                    "confidence": image_response.confidence
                }
            
            # 处理音频
            if input_data.audio:
                audio_response = await process_multimodal_input(
                    content=input_data.audio,
                    content_type="audio",
                    metadata=input_data.metadata
                )
                processed_content["audio"] = {
                    "content": input_data.audio,
                    "type": "audio",
                    "processed": audio_response.processed_content,
                    "features": audio_response.extracted_features,
                    "confidence": audio_response.confidence
                }
            
            # 处理视频
            if input_data.video:
                video_response = await process_multimodal_input(
                    content=input_data.video,
                    content_type="video",
                    metadata=input_data.metadata
                )
                processed_content["video"] = {
                    "content": input_data.video,
                    "type": "video",
                    "processed": video_response.processed_content,
                    "features": video_response.extracted_features,
                    "confidence": video_response.confidence
                }
            
            # 更新上下文
            context.multimodal_context.update(processed_content)
            context.updated_at = datetime.now().isoformat()
            
            return processed_content
            
        except Exception as e:
            logger.error(f"多模态输入处理失败: {e}")
            return {"error": str(e)}
    
    async def analyze_user_intent(self, 
                                input_data: MultiModalInput,
                                context: ConversationContext) -> Dict[str, Any]:
        """分析用户意图"""
        try:
            # 构建分析提示
            content_summary = []
            
            if input_data.text:
                content_summary.append(f"文本: {input_data.text}")
            
            if input_data.image:
                image_analysis = context.multimodal_context.get("image", {})
                if image_analysis:
                    content_summary.append(f"图像: {image_analysis.get('processed', '图像内容')}")
            
            if input_data.audio:
                audio_analysis = context.multimodal_context.get("audio", {})
                if audio_analysis:
                    content_summary.append(f"音频: {audio_analysis.get('processed', '音频内容')}")
            
            if input_data.video:
                video_analysis = context.multimodal_context.get("video", {})
                if video_analysis:
                    content_summary.append(f"视频: {video_analysis.get('processed', '视频内容')}")
            
            # 历史对话上下文
            recent_history = context.conversation_history[-3:] if context.conversation_history else []
            history_text = "\n".join([
                f"{h['role']}: {h['content'][:100]}..." 
                for h in recent_history
            ])
            
            intent_prompt = f"""
            分析用户的意图和需求：
            
            当前输入内容:
            {chr(10).join(content_summary)}
            
            最近对话历史:
            {history_text}
            
            用户画像: {context.user_profile}
            当前话题: {context.current_topic or '未知'}
            
            请返回JSON格式的意图分析：
            {{
                "primary_intent": "主要意图类别",
                "intent_confidence": 0.0-1.0,
                "specific_needs": ["具体需求1", "具体需求2"],
                "suggested_response_type": "knowledge_query|advice_request|training_content|general_chat",
                "complexity_level": "simple|medium|complex",
                "requires_multimodal_analysis": true/false,
                "topic_category": "销售技巧|产品知识|客户管理|其他",
                "urgency": "low|medium|high"
            }}
            """
            
            response = await self.llm.agenerate([intent_prompt])
            intent_text = response.generations[0][0].text.strip()
            
            try:
                intent_analysis = json.loads(intent_text)
            except:
                # 默认意图分析
                intent_analysis = {
                    "primary_intent": "一般查询",
                    "intent_confidence": 0.5,
                    "specific_needs": ["获取信息"],
                    "suggested_response_type": "knowledge_query",
                    "complexity_level": "medium",
                    "requires_multimodal_analysis": bool(input_data.image or input_data.audio or input_data.video),
                    "topic_category": "其他",
                    "urgency": "medium"
                }
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"意图分析失败: {e}")
            return {
                "primary_intent": "未知",
                "intent_confidence": 0.0,
                "specific_needs": [],
                "suggested_response_type": "general_chat",
                "complexity_level": "simple",
                "requires_multimodal_analysis": False,
                "topic_category": "其他",
                "urgency": "low"
            }
    
    async def generate_personalized_response(self,
                                           input_data: MultiModalInput,
                                           intent_analysis: Dict[str, Any],
                                           context: ConversationContext) -> AgentResponse:
        """生成个性化响应"""
        try:
            # 构建查询文本
            query_parts = []
            
            if input_data.text:
                query_parts.append(input_data.text)
            
            # 添加多模态内容描述
            for modality in ["image", "audio", "video"]:
                if modality in context.multimodal_context:
                    processed = context.multimodal_context[modality].get("processed", "")
                    if processed:
                        query_parts.append(f"[{modality.upper()}内容] {processed}")
            
            combined_query = " ".join(query_parts)
            
            # 准备多模态内容用于工作流
            multimodal_content = None
            if intent_analysis.get("requires_multimodal_analysis", False):
                multimodal_content = {
                    "content": context.multimodal_context,
                    "type": "multimodal",
                    "metadata": input_data.metadata
                }
            
            # 使用智能体工作流生成响应
            workflow_result = await run_sales_agent(
                user_query=combined_query,
                session_id=context.session_id,
                multimodal_content=multimodal_content
            )
            
            # 个性化调整响应
            personalized_response = await self._personalize_response(
                workflow_result,
                intent_analysis,
                context
            )
            
            # 生成建议
            suggestions = await self._generate_suggestions(
                input_data,
                intent_analysis,
                context,
                workflow_result
            )
            
            # 构建最终响应
            response = AgentResponse(
                text_response=personalized_response,
                confidence=workflow_result.get("confidence", 0.0),
                reasoning=workflow_result.get("reasoning_chain", []),
                sources=[
                    {
                        "type": "rag_result",
                        "content": r["answer"][:200] + "...",
                        "confidence": r["confidence"]
                    }
                    for r in workflow_result.get("rag_results", [])
                ],
                multimodal_analysis=context.multimodal_context,
                suggestions=suggestions,
                session_id=context.session_id,
                timestamp=datetime.now().isoformat()
            )
            
            # 更新对话历史
            context.conversation_history.append({
                "role": "user",
                "content": combined_query,
                "timestamp": datetime.now().isoformat(),
                "multimodal": bool(multimodal_content)
            })
            
            context.conversation_history.append({
                "role": "assistant",
                "content": personalized_response,
                "timestamp": datetime.now().isoformat(),
                "confidence": response.confidence
            })
            
            # 更新当前话题
            context.current_topic = intent_analysis.get("topic_category", context.current_topic)
            context.updated_at = datetime.now().isoformat()
            
            return response
            
        except Exception as e:
            logger.error(f"生成个性化响应失败: {e}")
            return AgentResponse(
                text_response=f"抱歉，处理您的请求时发生了错误：{str(e)}",
                confidence=0.0,
                reasoning=["系统错误"],
                sources=[],
                multimodal_analysis={},
                suggestions=["请稍后重试"],
                session_id=context.session_id,
                timestamp=datetime.now().isoformat()
            )
    
    async def _personalize_response(self,
                                  workflow_result: Dict[str, Any],
                                  intent_analysis: Dict[str, Any],
                                  context: ConversationContext) -> str:
        """个性化响应内容"""
        try:
            base_answer = workflow_result.get("answer", "")
            user_profile = context.user_profile
            
            personalization_prompt = f"""
            基于用户画像和意图分析，个性化以下回答：
            
            原始回答: {base_answer}
            
            用户画像: {user_profile}
            用户意图: {intent_analysis.get('primary_intent', '')}
            话题类别: {intent_analysis.get('topic_category', '')}
            复杂度: {intent_analysis.get('complexity_level', '')}
            紧急程度: {intent_analysis.get('urgency', '')}
            
            请根据用户的经验水平、角色、偏好等信息，调整回答的：
            1. 语言风格和专业程度
            2. 详细程度和深度
            3. 实例和案例的选择
            4. 建议的具体性和可操作性
            
            个性化回答:
            """
            
            response = await self.llm.agenerate([personalization_prompt])
            personalized = response.generations[0][0].text.strip()
            
            return personalized if personalized else base_answer
            
        except Exception as e:
            logger.error(f"个性化响应失败: {e}")
            return workflow_result.get("answer", "")
    
    async def _generate_suggestions(self,
                                  input_data: MultiModalInput,
                                  intent_analysis: Dict[str, Any],
                                  context: ConversationContext,
                                  workflow_result: Dict[str, Any]) -> List[str]:
        """生成相关建议"""
        try:
            suggestions_prompt = f"""
            基于当前对话和分析结果，生成3-5个相关的后续建议或问题：
            
            用户意图: {intent_analysis.get('primary_intent', '')}
            话题类别: {intent_analysis.get('topic_category', '')}
            当前话题: {context.current_topic or ''}
            
            建议应该：
            1. 与当前话题相关
            2. 有助于用户深入学习
            3. 具有实用价值
            4. 适合用户的水平
    
(Content truncated due to size limit. Use line ranges to read in chunks)


