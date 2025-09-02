"""
基于LangGraph的智能体工作流实现
构建复杂的多步推理和决策流程，支持状态管理和条件分支
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import uuid

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from .rag_system import rag_system, RAGResponse
from .mcp_server import process_multimodal_input

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """智能体状态定义"""
    messages: Annotated[List[Any], "对话消息列表"]
    user_query: str
    query_type: str  # "simple", "complex", "multimodal"
    context: Dict[str, Any]
    rag_results: Optional[List[RAGResponse]]
    current_step: str
    reasoning_chain: List[str]
    confidence: float
    final_answer: Optional[str]
    tools_used: List[str]
    session_id: str
    created_at: str


class SalesKnowledgeTool(BaseTool):
    """销售知识查询工具"""
    name = "sales_knowledge_search"
    description = "搜索销售相关知识库，获取产品信息、销售技巧、客户案例等"
    
    def _run(self, query: str, content_type: Optional[str] = None) -> str:
        """同步运行（LangGraph需要）"""
        import asyncio
        return asyncio.run(self._arun(query, content_type))
    
    async def _arun(self, query: str, content_type: Optional[str] = None) -> str:
        """异步运行"""
        try:
            response = await rag_system.query(
                question=query,
                content_type_filter=content_type
            )
            
            result = {
                "answer": response.answer,
                "confidence": response.confidence,
                "sources_count": len(response.sources),
                "reasoning": response.reasoning
            }
            
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"销售知识查询失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class MultiModalAnalysisTool(BaseTool):
    """多模态内容分析工具"""
    name = "multimodal_analysis"
    description = "分析图像、音频、视频等多模态内容，提取关键信息"
    
    def _run(self, content: str, content_type: str, metadata: Optional[Dict] = None) -> str:
        """同步运行"""
        import asyncio
        return asyncio.run(self._arun(content, content_type, metadata))
    
    async def _arun(self, content: str, content_type: str, metadata: Optional[Dict] = None) -> str:
        """异步运行"""
        try:
            response = await process_multimodal_input(
                content=content,
                content_type=content_type,
                metadata=metadata or {}
            )
            
            result = {
                "processed_content": response.processed_content,
                "confidence": response.confidence,
                "extracted_features": response.extracted_features,
                "content_type": response.content_type
            }
            
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"多模态分析失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class SalesAdviceTool(BaseTool):
    """销售建议工具"""
    name = "sales_advice"
    description = "基于客户信息和销售场景提供个性化销售建议"
    
    def _run(self, customer_info: str, sales_scenario: str, product_info: str = "") -> str:
        """同步运行"""
        import asyncio
        return asyncio.run(self._arun(customer_info, sales_scenario, product_info))
    
    async def _arun(self, customer_info: str, sales_scenario: str, product_info: str = "") -> str:
        """异步运行"""
        try:
            # 构建销售建议查询
            query = f"""
            客户信息: {customer_info}
            销售场景: {sales_scenario}
            产品信息: {product_info}
            
            请提供针对性的销售建议和策略。
            """
            
            response = await rag_system.query(question=query)
            
            result = {
                "advice": response.answer,
                "confidence": response.confidence,
                "reasoning": response.reasoning,
                "sources_count": len(response.sources)
            }
            
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"销售建议生成失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


# 初始化工具
tools = [
    SalesKnowledgeTool(),
    MultiModalAnalysisTool(),
    SalesAdviceTool()
]

tool_executor = ToolExecutor(tools)


class SalesAgentWorkflow:
    """销售智能体工作流"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.1
        )
        
        # 创建状态图
        self.workflow = StateGraph(AgentState)
        
        # 添加节点
        self.workflow.add_node("analyze_query", self.analyze_query)
        self.workflow.add_node("simple_response", self.simple_response)
        self.workflow.add_node("complex_reasoning", self.complex_reasoning)
        self.workflow.add_node("multimodal_processing", self.multimodal_processing)
        self.workflow.add_node("knowledge_retrieval", self.knowledge_retrieval)
        self.workflow.add_node("generate_response", self.generate_response)
        self.workflow.add_node("validate_response", self.validate_response)
        
        # 设置入口点
        self.workflow.set_entry_point("analyze_query")
        
        # 添加条件边
        self.workflow.add_conditional_edges(
            "analyze_query",
            self.route_query,
            {
                "simple": "simple_response",
                "complex": "complex_reasoning",
                "multimodal": "multimodal_processing"
            }
        )
        
        # 添加边
        self.workflow.add_edge("simple_response", "validate_response")
        self.workflow.add_edge("complex_reasoning", "knowledge_retrieval")
        self.workflow.add_edge("multimodal_processing", "knowledge_retrieval")
        self.workflow.add_edge("knowledge_retrieval", "generate_response")
        self.workflow.add_edge("generate_response", "validate_response")
        self.workflow.add_edge("validate_response", END)
        
        # 编译工作流
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info("销售智能体工作流初始化完成")
    
    async def analyze_query(self, state: AgentState) -> AgentState:
        """分析用户查询"""
        try:
            query = state["user_query"]
            
            # 使用LLM分析查询类型
            analysis_prompt = f"""
            分析以下用户查询，确定其类型和复杂度：
            
            查询: {query}
            
            请返回JSON格式的分析结果：
            {{
                "query_type": "simple|complex|multimodal",
                "complexity_score": 0.0-1.0,
                "requires_multimodal": true/false,
                "requires_knowledge_search": true/false,
                "main_intent": "查询意图描述",
                "keywords": ["关键词1", "关键词2"]
            }}
            """
            
            response = await self.llm.agenerate([analysis_prompt])
            analysis_text = response.generations[0][0].text.strip()
            
            try:
                analysis = json.loads(analysis_text)
            except:
                # 如果JSON解析失败，使用默认分析
                analysis = {
                    "query_type": "simple",
                    "complexity_score": 0.5,
                    "requires_multimodal": False,
                    "requires_knowledge_search": True,
                    "main_intent": "一般查询",
                    "keywords": [query]
                }
            
            state["query_type"] = analysis["query_type"]
            state["context"].update(analysis)
            state["current_step"] = "analyze_query"
            state["reasoning_chain"].append(f"查询分析: {analysis['main_intent']}")
            
            logger.info(f"查询分析完成: {analysis['query_type']}")
            return state
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            state["query_type"] = "simple"
            state["current_step"] = "analyze_query"
            return state
    
    def route_query(self, state: AgentState) -> str:
        """路由查询到不同的处理分支"""
        query_type = state["query_type"]
        
        # 检查是否包含多模态内容
        if state["context"].get("requires_multimodal", False):
            return "multimodal"
        
        # 根据复杂度路由
        complexity = state["context"].get("complexity_score", 0.5)
        if complexity > 0.7:
            return "complex"
        else:
            return "simple"
    
    async def simple_response(self, state: AgentState) -> AgentState:
        """处理简单查询"""
        try:
            query = state["user_query"]
            
            # 直接使用RAG系统获取答案
            rag_response = await rag_system.query(query, max_sources=3)
            
            state["rag_results"] = [rag_response]
            state["final_answer"] = rag_response.answer
            state["confidence"] = rag_response.confidence
            state["current_step"] = "simple_response"
            state["reasoning_chain"].append("简单查询直接响应")
            state["tools_used"].append("rag_system")
            
            logger.info("简单查询处理完成")
            return state
            
        except Exception as e:
            logger.error(f"简单查询处理失败: {e}")
            state["final_answer"] = f"处理查询时发生错误: {str(e)}"
            state["confidence"] = 0.0
            return state
    
    async def complex_reasoning(self, state: AgentState) -> AgentState:
        """处理复杂推理"""
        try:
            query = state["user_query"]
            context = state["context"]
            
            # 分解复杂查询为子问题
            decomposition_prompt = f"""
            将以下复杂查询分解为多个子问题：
            
            原始查询: {query}
            查询意图: {context.get('main_intent', '')}
            
            请返回JSON格式的分解结果：
            {{
                "sub_questions": ["子问题1", "子问题2", "子问题3"],
                "reasoning_steps": ["推理步骤1", "推理步骤2", "推理步骤3"],
                "required_tools": ["tool1", "tool2"]
            }}
            """
            
            response = await self.llm.agenerate([decomposition_prompt])
            decomposition_text = response.generations[0][0].text.strip()
            
            try:
                decomposition = json.loads(decomposition_text)
            except:
                decomposition = {
                    "sub_questions": [query],
                    "reasoning_steps": ["直接回答"],
                    "required_tools": ["sales_knowledge_search"]
                }
            
            state["context"]["decomposition"] = decomposition
            state["current_step"] = "complex_reasoning"
            state["reasoning_chain"].extend(decomposition["reasoning_steps"])
            
            logger.info(f"复杂推理分解完成: {len(decomposition['sub_questions'])} 个子问题")
            return state
            
        except Exception as e:
            logger.error(f"复杂推理失败: {e}")
            state["context"]["decomposition"] = {
                "sub_questions": [state["user_query"]],
                "reasoning_steps": ["直接回答"],
                "required_tools": ["sales_knowledge_search"]
            }
            return state
    
    async def multimodal_processing(self, state: AgentState) -> AgentState:
        """处理多模态内容"""
        try:
            # 这里假设多模态内容在context中
            multimodal_content = state["context"].get("multimodal_content", {})
            
            if not multimodal_content:
                logger.warning("未找到多模态内容")
                state["current_step"] = "multimodal_processing"
                return state
            
            # 处理多模态内容
            tool_input = ToolInvocation(
                tool="multimodal_analysis",
                tool_input={
                    "content": multimodal_content.get("content", ""),
                    "content_type": multimodal_content.get("type", "text"),
                    "metadata": multimodal_content.get("metadata", {})
                }
            )
            
            result = tool_executor.invoke(tool_input)
            analysis_result = json.loads(result)
            
            state["context"]["multimodal_analysis"] = analysis_result
            state["current_step"] = "multimodal_processing"
            state["reasoning_chain"].append(f"多模态分析: {analysis_result.get('content_type', 'unknown')}")
            state["tools_used"].append("multimodal_analysis")
            
            logger.info("多模态处理完成")
            return state
            
        except Exception as e:
            logger.error(f"多模态处理失败: {e}")
            state["current_step"] = "multimodal_processing"
            return state
    
    async def knowledge_retrieval(self, state: AgentState) -> AgentState:
        """知识检索"""
        try:
            query = state["user_query"]
            decomposition = state["context"].get("decomposition", {})
            sub_questions = decomposition.get("sub_questions", [query])
            
            rag_results = []
            
            # 对每个子问题进行检索
            for sub_q in sub_questions:
                tool_input = ToolInvocation(
                    tool="sales_knowledge_search",
                    tool_input={"query": sub_q}
                )
                
                result = tool_executor.invoke(tool_input)
                result_data = json.loads(result)
                
                # 创建RAGResponse对象
                rag_response = RAGResponse(
                    answer=result_data.get("answer", ""),
                    sources=[],  # 简化处理
                    confidence=result_data.get("confidence", 0.0),
                    query=sub_q,
                    reasoning=result_data.get("reasoning", "")
                )
                
                rag_results.append(rag_response)
            
            state["rag_results"] = rag_results
            state["current_step"] = "knowledge_retrieval"
            state["reasoning_chain"].append(f"检索了 {len(rag_results)} 个知识片段")
            state["tools_used"].append("sales_knowledge_search")
            
            logger.info(f"知识检索完成: {len(rag_results)} 个结果")
            return state
            
        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            state["rag_results"] = []
            return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """生成最终响应"""
        try:
            query = state["user_query"]
            rag_results = state["rag_results"] or []
            reasoning_chain = state["reasoning_chain"]
            
            # 构建综合响应
            context_parts = []
            total_confidence = 0.0
            
            for i, rag_result in enumerate(rag_results):
                context_parts.append(f"知识片段 {i+1}: {rag_result.answer}")
                total_confidence += rag_result.confidence
            
            avg_confidence = total_confidence / len(rag_results) if rag_results else 0.0
            
            # 多模态分析结果
            multimodal_analysis = state["context"].get("multimodal_analysis", {})
            if multimodal_analysis:
                context_parts.append(f"多模态分析: {multimodal_analysis.get('processed_content', '')}")
            
            synthesis_prompt = f"""
            基于以下信息，为用户提供综合性的回答：
            
            用户问题: {query}
            
            推理过程: {' -> '.join(reasoning_chain)}
            
            知识内容:
            {chr(10).join(context_parts)}
            
            请提供一个完整、准确、有用的回答，包括：
            1. 直接回答用户问题
            2. 提供详细解释
            3. 给出实用建议
            4. 如果适用，提供相关案例
            
            回答:
            """
            
            response = await self.llm.agenerate([synthesis_prompt])
            final_answer = response.generations[0][0].text.strip()
  
(Content truncated due to size limit. Use line ranges to read in chunks)


