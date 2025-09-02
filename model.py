"""
数据模型定义
定义系统中使用的所有数据模型和数据结构
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    MULTIMODAL = "multimodal"


class UserRole(str, Enum):
    """用户角色枚举"""
    SALES_NEWBIE = "销售新手"
    SALES_SPECIALIST = "销售专员"
    SALES_SUPERVISOR = "销售主管"
    SALES_MANAGER = "销售经理"
    SALES_DIRECTOR = "销售总监"


class ExperienceLevel(str, Enum):
    """经验水平枚举"""
    BEGINNER = "0-1年"
    INTERMEDIATE = "1-3年"
    ADVANCED = "3-5年"
    EXPERT = "5年以上"


class QueryType(str, Enum):
    """查询类型枚举"""
    SIMPLE = "simple"
    COMPLEX = "complex"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# 基础模型
class BaseTimestampModel(BaseModel):
    """带时间戳的基础模型"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class UserProfile(BaseModel):
    """用户画像模型"""
    name: Optional[str] = Field(default=None, description="用户姓名")
    role: UserRole = Field(default=UserRole.SALES_NEWBIE, description="用户角色")
    experience: ExperienceLevel = Field(default=ExperienceLevel.BEGINNER, description="经验水平")
    industry: Optional[str] = Field(default=None, description="所属行业")
    specialization: Optional[str] = Field(default=None, description="专业领域")
    goals: List[str] = Field(default=[], description="学习目标")
    preferences: Dict[str, Any] = Field(default={}, description="个人偏好")


class MediaMetadata(BaseModel):
    """媒体元数据模型"""
    filename: Optional[str] = Field(default=None, description="文件名")
    size: Optional[int] = Field(default=None, description="文件大小（字节）")
    format: Optional[str] = Field(default=None, description="文件格式")
    duration: Optional[float] = Field(default=None, description="时长（秒）")
    dimensions: Optional[Dict[str, int]] = Field(default=None, description="尺寸信息")
    encoding: Optional[str] = Field(default="utf-8", description="编码格式")


class ProcessedContent(BaseModel):
    """处理后的内容模型"""
    original_content: Union[str, bytes] = Field(..., description="原始内容")
    processed_text: str = Field(..., description="处理后的文本")
    content_type: ContentType = Field(..., description="内容类型")
    extracted_features: Dict[str, Any] = Field(default={}, description="提取的特征")
    metadata: MediaMetadata = Field(default_factory=MediaMetadata, description="元数据")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="处理置信度")
    processing_time: Optional[float] = Field(default=None, description="处理耗时（秒）")


class KnowledgeSource(BaseModel):
    """知识来源模型"""
    id: str = Field(..., description="来源ID")
    title: str = Field(..., description="标题")
    content: str = Field(..., description="内容")
    content_type: ContentType = Field(..., description="内容类型")
    source_path: str = Field(..., description="来源路径")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性分数")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class RetrievalContext(BaseModel):
    """检索上下文模型"""
    query: str = Field(..., description="查询文本")
    query_type: QueryType = Field(default=QueryType.SIMPLE, description="查询类型")
    sources: List[KnowledgeSource] = Field(default=[], description="检索到的来源")
    total_sources: int = Field(default=0, description="总来源数")
    retrieval_time: Optional[float] = Field(default=None, description="检索耗时（秒）")
    filters: Dict[str, Any] = Field(default={}, description="检索过滤条件")


class ReasoningStep(BaseModel):
    """推理步骤模型"""
    step_id: int = Field(..., description="步骤ID")
    description: str = Field(..., description="步骤描述")
    input_data: Dict[str, Any] = Field(default={}, description="输入数据")
    output_data: Dict[str, Any] = Field(default={}, description="输出数据")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="置信度")
    execution_time: Optional[float] = Field(default=None, description="执行时间（秒）")
    tools_used: List[str] = Field(default=[], description="使用的工具")


class AgentDecision(BaseModel):
    """智能体决策模型"""
    decision_id: str = Field(..., description="决策ID")
    decision_type: str = Field(..., description="决策类型")
    reasoning_chain: List[ReasoningStep] = Field(default=[], description="推理链")
    final_decision: str = Field(..., description="最终决策")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="决策置信度")
    alternatives: List[str] = Field(default=[], description="备选方案")


class ConversationMessage(BaseModel):
    """对话消息模型"""
    message_id: str = Field(..., description="消息ID")
    role: str = Field(..., description="角色（user/assistant）")
    content: str = Field(..., description="消息内容")
    content_type: ContentType = Field(default=ContentType.TEXT, description="内容类型")
    metadata: Dict[str, Any] = Field(default={}, description="消息元数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    processing_info: Optional[ProcessedContent] = Field(default=None, description="处理信息")


class ConversationSession(BaseTimestampModel):
    """对话会话模型"""
    session_id: str = Field(..., description="会话ID")
    user_profile: UserProfile = Field(default_factory=UserProfile, description="用户画像")
    messages: List[ConversationMessage] = Field(default=[], description="消息列表")
    current_topic: Optional[str] = Field(default=None, description="当前话题")
    context: Dict[str, Any] = Field(default={}, description="会话上下文")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="会话状态")
    total_messages: int = Field(default=0, description="消息总数")
    last_activity: datetime = Field(default_factory=datetime.now, description="最后活动时间")


class AgentResponse(BaseModel):
    """智能体响应模型"""
    response_id: str = Field(..., description="响应ID")
    session_id: str = Field(..., description="会话ID")
    text_response: str = Field(..., description="文本响应")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="响应置信度")
    reasoning: List[str] = Field(default=[], description="推理过程")
    sources: List[KnowledgeSource] = Field(default=[], description="知识来源")
    suggestions: List[str] = Field(default=[], description="相关建议")
    multimodal_analysis: Dict[str, Any] = Field(default={}, description="多模态分析结果")
    processing_time: Optional[float] = Field(default=None, description="处理时间（秒）")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    agent_decision: Optional[AgentDecision] = Field(default=None, description="智能体决策")


class SystemMetrics(BaseModel):
    """系统指标模型"""
    metric_id: str = Field(..., description="指标ID")
    metric_name: str = Field(..., description="指标名称")
    metric_value: Union[int, float, str] = Field(..., description="指标值")
    metric_type: str = Field(..., description="指标类型")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    tags: Dict[str, str] = Field(default={}, description="标签")


class KnowledgeBaseStats(BaseModel):
    """知识库统计模型"""
    total_documents: int = Field(default=0, description="文档总数")
    total_chunks: int = Field(default=0, description="文档块总数")
    content_types: Dict[str, int] = Field(default={}, description="内容类型分布")
    total_size_bytes: int = Field(default=0, description="总大小（字节）")
    last_updated: datetime = Field(default_factory=datetime.now, description="最后更新时间")
    index_status: ProcessingStatus = Field(default=ProcessingStatus.COMPLETED, description="索引状态")


class APIRequest(BaseModel):
    """API请求模型"""
    request_id: str = Field(..., description="请求ID")
    endpoint: str = Field(..., description="API端点")
    method: str = Field(..., description="HTTP方法")
    parameters: Dict[str, Any] = Field(default={}, description="请求参数")
    user_agent: Optional[str] = Field(default=None, description="用户代理")
    ip_address: Optional[str] = Field(default=None, description="IP地址")
    timestamp: datetime = Field(default_factory=datetime.now, description="请求时间")


class APIResponse(BaseModel):
    """API响应模型"""
    request_id: str = Field(..., description="对应的请求ID")
    status_code: int = Field(..., description="HTTP状态码")
    response_data: Dict[str, Any] = Field(default={}, description="响应数据")
    processing_time: float = Field(..., description="处理时间（秒）")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")


class FileUpload(BaseTimestampModel):
    """文件上传模型"""
    upload_id: str = Field(..., description="上传ID")
    filename: str = Field(..., description="文件名")
    original_filename: str = Field(..., description="原始文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小（字节）")
    content_type: ContentType = Field(..., description="内容类型")
    mime_type: str = Field(..., description="MIME类型")
    upload_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="上传状态")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="处理状态")
    metadata: Dict[str, Any] = Field(default={}, description="文件元数据")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class WorkflowState(BaseModel):
    """工作流状态模型"""
    workflow_id: str = Field(..., description="工作流ID")
    session_id: str = Field(..., description="会话ID")
    current_step: str = Field(..., description="当前步骤")
    step_history: List[str] = Field(default=[], description="步骤历史")
    state_data: Dict[str, Any] = Field(default={}, description="状态数据")
    execution_context: Dict[str, Any] = Field(default={}, description="执行上下文")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="工作流状态")
    start_time: datetime = Field(default_factory=datetime.now, description="开始时间")
    end_time: Optional[datetime] = Field(default=None, description="结束时间")
    total_execution_time: Optional[float] = Field(default=None, description="总执行时间（秒）")


class ErrorLog(BaseTimestampModel):
    """错误日志模型"""
    error_id: str = Field(..., description="错误ID")
    error_type: str = Field(..., description="错误类型")
    error_message: str = Field(..., description="错误信息")
    stack_trace: Optional[str] = Field(default=None, description="堆栈跟踪")
    context: Dict[str, Any] = Field(default={}, description="错误上下文")
    severity: str = Field(default="ERROR", description="严重程度")
    component: str = Field(..., description="出错组件")
    session_id: Optional[str] = Field(default=None, description="相关会话ID")
    request_id: Optional[str] = Field(default=None, description="相关请求ID")


# 请求和响应模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    text: Optional[str] = Field(default=None, description="文本消息")
    image_base64: Optional[str] = Field(default=None, description="Base64编码的图像")
    audio_base64: Optional[str] = Field(default=None, description="Base64编码的音频")
    video_base64: Optional[str] = Field(default=None, description="Base64编码的视频")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    user_profile: Optional[UserProfile] = Field(default=None, description="用户画像")
    metadata: Dict[str, Any] = Field(default={}, description="请求元数据")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "什么是销售漏斗？",
                "session_id": "session_123",
                "user_profile": {
                    "role": "销售新手",
                    "experience": "0-1年",
                    "industry": "软件"
                }
            }
        }


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[AgentResponse] = Field(default=None, description="响应数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "response_id": "resp_123",
                    "session_id": "session_123",
                    "text_response": "销售漏斗是一个描述客户从初次接触到最终购买的过程模型...",
                    "confidence": 0.95,
                    "suggestions": ["了解更多销售技巧", "查看实际案例"]
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="系统状态")
    components: Dict[str, str] = Field(..., description="组件状态")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    version: str = Field(default="1.0.0", description="系统版本")
    uptime: Optional[float] = Field(default=None, description="运行时间（秒）")


class KnowledgeUploadRequest(BaseModel):
    """知识库上传请求模型"""
    content_type: ContentType = Field(default=ContentType.TEXT, description="内容类型")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    class Config:
        schema_extra = {
            "example": {
                "content_type": "document",
                "metadata": {
                    "category": "sales_training",
                    "priority": "high"
                }
            }
        }


class KnowledgeUploadResponse(BaseModel):
    """知识库上传响应模型"""
    success: bool = Field(..., description="是否成功")
    file_id: Optional[str] = Field(default=None, description="文件ID")
    message: str = Field(..., description="响应消息")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="处理状态")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "file_id": "file_123",
                "message": "文件上传成功，正在处理中...",
                "processing_status": "processing"
            }
        }



