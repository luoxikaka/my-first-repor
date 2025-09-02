"""
检索增强生成 (RAG) 系统实现
支持多模态知识库的构建、检索和生成，包括文档、图像、音频等多种数据类型
"""

import asyncio
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import chromadb
import numpy as np
import pandas as pd
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .mcp_server import process_multimodal_input

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeItem(BaseModel):
    """知识库条目模型"""
    id: str = Field(..., description="唯一标识符")
    content: str = Field(..., description="文本内容")
    content_type: str = Field(..., description="内容类型")
    source: str = Field(..., description="来源文件路径")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    embedding: Optional[List[float]] = Field(default=None, description="向量嵌入")
    created_at: str = Field(..., description="创建时间")


class RetrievalResult(BaseModel):
    """检索结果模型"""
    items: List[KnowledgeItem] = Field(..., description="检索到的知识条目")
    scores: List[float] = Field(..., description="相似度分数")
    query: str = Field(..., description="查询文本")
    total_results: int = Field(..., description="总结果数")


class RAGResponse(BaseModel):
    """RAG响应模型"""
    answer: str = Field(..., description="生成的答案")
    sources: List[KnowledgeItem] = Field(..., description="参考来源")
    confidence: float = Field(..., description="置信度")
    query: str = Field(..., description="原始查询")
    reasoning: Optional[str] = Field(default=None, description="推理过程")


class MultiModalEmbedding:
    """多模态嵌入生成器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.text_model = SentenceTransformer(model_name)
        logger.info(f"加载文本嵌入模型: {model_name}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为向量"""
        return self.text_model.encode(text)
    
    def encode_multimodal(self, content: str, content_type: str) -> np.ndarray:
        """编码多模态内容为向量"""
        # 对于非文本内容，使用其文本描述进行编码
        if content_type == "text":
            return self.encode_text(content)
        else:
            # 对于图像、音频等，使用MCP处理后的文本描述
            return self.encode_text(content)


class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self, 
                 persist_directory: str = "./knowledge_base_db",
                 collection_name: str = "sales_knowledge"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 初始化ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "销售知识库"}
        )
        
        # 初始化嵌入模型
        self.embedding_model = MultiModalEmbedding()
        
        # 文档分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        logger.info(f"知识库初始化完成，存储路径: {persist_directory}")
    
    async def add_document(self, file_path: str, content_type: str = "auto") -> List[str]:
        """添加文档到知识库"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 自动检测内容类型
            if content_type == "auto":
                content_type = self._detect_content_type(file_path)
            
            logger.info(f"添加文档: {file_path}, 类型: {content_type}")
            
            # 根据文件类型加载文档
            documents = await self._load_document(file_path, content_type)
            
            # 分割文档
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(doc_chunks):
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_id": i,
                            "source": str(file_path),
                            "content_type": content_type
                        }
                    ))
            
            # 生成嵌入并存储
            item_ids = []
            for chunk in chunks:
                item_id = str(uuid.uuid4())
                
                # 生成嵌入
                embedding = self.embedding_model.encode_multimodal(
                    chunk.page_content, 
                    content_type
                )
                
                # 存储到ChromaDB
                self.collection.add(
                    ids=[item_id],
                    documents=[chunk.page_content],
                    metadatas=[chunk.metadata],
                    embeddings=[embedding.tolist()]
                )
                
                item_ids.append(item_id)
            
            logger.info(f"成功添加 {len(chunks)} 个文档块")
            return item_ids
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    async def add_multimodal_content(self, 
                                   content: Union[str, bytes], 
                                   content_type: str,
                                   source: str,
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加多模态内容到知识库"""
        try:
            # 使用MCP处理多模态内容
            mcp_response = await process_multimodal_input(
                content=content,
                content_type=content_type,
                metadata=metadata or {}
            )
            
            # 创建知识条目
            item_id = str(uuid.uuid4())
            
            # 生成嵌入
            embedding = self.embedding_model.encode_multimodal(
                mcp_response.processed_content,
                content_type
            )
            
            # 构建元数据
            full_metadata = {
                "source": source,
                "content_type": content_type,
                "confidence": mcp_response.confidence,
                "extracted_features": mcp_response.extracted_features,
                **(metadata or {})
            }
            
            # 存储到ChromaDB
            self.collection.add(
                ids=[item_id],
                documents=[mcp_response.processed_content],
                metadatas=[full_metadata],
                embeddings=[embedding.tolist()]
            )
            
            logger.info(f"成功添加多模态内容: {item_id}, 类型: {content_type}")
            return item_id
            
        except Exception as e:
            logger.error(f"添加多模态内容失败: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               content_type_filter: Optional[str] = None) -> RetrievalResult:
        """搜索知识库"""
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode_text(query)
            
            # 构建过滤条件
            where_clause = {}
            if content_type_filter:
                where_clause["content_type"] = content_type_filter
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # 构建知识条目
            items = []
            scores = []
            
            for i in range(len(results["ids"][0])):
                item = KnowledgeItem(
                    id=results["ids"][0][i],
                    content=results["documents"][0][i],
                    content_type=results["metadatas"][0][i].get("content_type", "text"),
                    source=results["metadatas"][0][i].get("source", "unknown"),
                    metadata=results["metadatas"][0][i],
                    created_at=results["metadatas"][0][i].get("created_at", "unknown")
                )
                items.append(item)
                
                # 转换距离为相似度分数 (距离越小，相似度越高)
                distance = results["distances"][0][i]
                score = 1.0 / (1.0 + distance)
                scores.append(score)
            
            return RetrievalResult(
                items=items,
                scores=scores,
                query=query,
                total_results=len(items)
            )
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise
    
    def _detect_content_type(self, file_path: Path) -> str:
        """检测文件内容类型"""
        suffix = file_path.suffix.lower()
        
        if suffix in [".txt", ".md", ".markdown"]:
            return "text"
        elif suffix in [".pdf"]:
            return "pdf"
        elif suffix in [".doc", ".docx"]:
            return "document"
        elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            return "image"
        elif suffix in [".mp3", ".wav", ".flac", ".aac"]:
            return "audio"
        elif suffix in [".mp4", ".avi", ".mov", ".wmv"]:
            return "video"
        else:
            return "text"  # 默认为文本
    
    async def _load_document(self, file_path: Path, content_type: str) -> List[Document]:
        """加载文档"""
        try:
            if content_type == "text":
                loader = TextLoader(str(file_path), encoding="utf-8")
                return loader.load()
            elif content_type == "pdf":
                loader = PyPDFLoader(str(file_path))
                return loader.load()
            elif content_type == "document":
                loader = UnstructuredWordDocumentLoader(str(file_path))
                return loader.load()
            elif content_type == "markdown":
                loader = UnstructuredMarkdownLoader(str(file_path))
                return loader.load()
            elif content_type in ["image", "audio", "video"]:
                # 对于多媒体文件，读取二进制数据并使用MCP处理
                with open(file_path, "rb") as f:
                    content = f.read()
                
                mcp_response = await process_multimodal_input(
                    content=content,
                    content_type=content_type,
                    metadata={"source": str(file_path)}
                )
                
                return [Document(
                    page_content=mcp_response.processed_content,
                    metadata={
                        "source": str(file_path),
                        "content_type": content_type,
                        "confidence": mcp_response.confidence,
                        "extracted_features": mcp_response.extracted_features
                    }
                )]
            else:
                raise ValueError(f"不支持的内容类型: {content_type}")
                
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            raise


class RAGGenerator:
    """RAG生成器"""
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.1):
        self.knowledge_base = knowledge_base
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # 销售专用提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是一个专业的销售培训助手。请基于以下知识库内容回答用户的问题。

知识库内容:
{context}

用户问题: {question}

请提供准确、专业的回答，并在回答中引用相关的知识来源。如果知识库中没有相关信息，请诚实地说明。

回答格式:
1. 直接回答用户问题
2. 提供详细解释和建议
3. 引用相关知识来源
4. 如果适用，提供实际应用建议

回答:"""
        )
        
        logger.info("RAG生成器初始化完成")
    
    async def generate_answer(self, 
                            query: str, 
                            max_sources: int = 5,
                            content_type_filter: Optional[str] = None) -> RAGResponse:
        """生成答案"""
        try:
            # 检索相关知识
            retrieval_result = self.knowledge_base.search(
                query=query,
                n_results=max_sources,
                content_type_filter=content_type_filter
            )
            
            if not retrieval_result.items:
                return RAGResponse(
                    answer="抱歉，我在知识库中没有找到相关信息来回答您的问题。",
                    sources=[],
                    confidence=0.0,
                    query=query,
                    reasoning="知识库中无相关内容"
                )
            
            # 构建上下文
            context_parts = []
            for i, item in enumerate(retrieval_result.items):
                context_parts.append(f"来源 {i+1} ({item.source}):\n{item.content}\n")
            
            context = "\n".join(context_parts)
            
            # 生成答案
            prompt = self.prompt_template.format(context=context, question=query)
            
            response = await self.llm.agenerate([prompt])
            answer = response.generations[0][0].text.strip()
            
            # 计算置信度（基于检索结果的平均分数）
            confidence = np.mean(retrieval_result.scores) if retrieval_result.scores else 0.0
            
            return RAGResponse(
                answer=answer,
                sources=retrieval_result.items,
                confidence=confidence,
                query=query,
                reasoning=f"基于 {len(retrieval_result.items)} 个知识来源生成答案"
            )
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return RAGResponse(
                answer=f"生成答案时发生错误: {str(e)}",
                sources=[],
                confidence=0.0,
                query=query,
                reasoning="系统错误"
            )


class RAGSystem:
    """完整的RAG系统"""
    
    def __init__(self, 
                 knowledge_base_path: str = "./knowledge_base_db",
                 model_name: str = "gpt-3.5-turbo"):
        self.knowledge_base = KnowledgeBase(persist_directory=knowledge_base_path)
        self.generator = RAGGenerator(
            knowledge_base=self.knowledge_base,
            model_name=model_name
        )
        logger.info("RAG系统初始化完成")
    
    async def initialize_knowledge_base(self, knowledge_dir: str):
        """初始化知识库"""
        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            logger.warning(f"知识库目录不存在: {knowledge_dir}")
            return
        
        logger.info(f"开始初始化知识库: {knowledge_dir}")
        
        # 遍历知识库目录
        for file_path in knowledge_path.rglob("*"):
            if file_path.is_file():
                try:
                    await self.knowledge_base.add_document(str(file_path))
                    logger.info(f"已添加文件: {file_path}")
                except Exception as e:
                    logger.error(f"添加文件失败 {file_path}: {e}")
        
        logger.info("知识库初始化完成")
    
    async def query(self, 
                   question: str, 
                   max_sources: int = 5,
                   content_type_filter: Optional[str] = None) -> RAGResponse:
        """查询RAG系统"""
        return await self.generator.generate_answer(
            query=question,
            max_sources=max_sources,
            content_type_filter=content_type_filter
        )
    
    async def add_knowledge(self, 
                          content: Union[str, bytes],
                          content_type: str,
                          source: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """添加知识到系统"""
        return await self.knowledge_base.add_multimodal_content(
            content=
(Content truncated due to size limit. Use line ranges to read in chunks)