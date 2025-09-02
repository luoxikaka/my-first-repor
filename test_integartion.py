"""
销售智能问答知识培训智能体 - 集成测试
测试整个系统的端到端功能
"""

import asyncio
import base64
import json
import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

import httpx
from fastapi.testclient import TestClient

# 导入应用模块
from main import app
from app.multimodal_agent import chat_with_agent
from app.rag_system import rag_system, initialize_rag_system
from app.mcp_server import process_multimodal_input
from app.agent_workflow import run_sales_agent


class TestSalesAgentIntegration:
    """销售智能体集成测试类"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """创建测试客户端"""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    async def setup_system(self):
        """设置测试环境"""
        # 创建临时知识库目录
        temp_dir = tempfile.mkdtemp()
        
        # 创建测试文档
        test_doc_path = Path(temp_dir) / "test_sales_guide.txt"
        with open(test_doc_path, "w", encoding="utf-8") as f:
            f.write("""
            销售漏斗基础知识
            
            销售漏斗是一个描述客户从初次接触到最终购买的过程模型。
            它包括以下几个阶段：
            
            1. 认知阶段：客户意识到问题或需求
            2. 兴趣阶段：客户对解决方案产生兴趣
            3. 考虑阶段：客户评估不同的选择
            4. 意向阶段：客户表现出购买意向
            5. 评估阶段：客户进行最终评估
            6. 购买阶段：客户做出购买决定
            
            提高转化率的关键策略：
            - 精准定位目标客户
            - 提供有价值的内容
            - 建立信任关系
            - 及时跟进客户
            - 处理客户异议
            - 优化销售流程
            """)
        
        # 初始化RAG系统
        await initialize_rag_system(temp_dir)
        
        return temp_dir
    
    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """测试根路径接口"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_text_chat(self, client, setup_system):
        """测试文本聊天功能"""
        await setup_system
        
        # 测试简单问题
        chat_request = {
            "text": "什么是销售漏斗？",
            "user_profile": {
                "role": "销售新手",
                "experience": "0-1年",
                "industry": "软件"
            }
        }
        
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        
        response_data = data["data"]
        assert "text_response" in response_data
        assert "confidence" in response_data
        assert "session_id" in response_data
        assert len(response_data["text_response"]) > 0
    
    @pytest.mark.asyncio
    async def test_multimodal_chat(self, client, setup_system):
        """测试多模态聊天功能"""
        await setup_system
        
        # 创建测试图像（简单的base64编码）
        test_image = b"fake_image_data"
        image_base64 = base64.b64encode(test_image).decode()
        
        chat_request = {
            "text": "请分析这张图片中的销售数据",
            "image_base64": image_base64,
            "user_profile": {
                "role": "销售经理",
                "experience": "3-5年",
                "industry": "零售"
            }
        }
        
        response = client.post("/chat", json=chat_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        
        response_data = data["data"]
        assert "multimodal_analysis" in response_data
    
    def test_knowledge_upload(self, client):
        """测试知识库文件上传"""
        # 创建测试文件
        test_content = "这是一个测试销售文档，包含重要的销售技巧和策略。"
        
        files = {
            "file": ("test_doc.txt", test_content, "text/plain")
        }
        
        data = {
            "content_type": "text",
            "metadata": json.dumps({"category": "sales_tips"})
        }
        
        response = client.post("/upload_knowledge", files=files, data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["success"] is True
        assert "file_id" in result
        assert "message" in result
    
    def test_conversation_history(self, client):
        """测试对话历史功能"""
        session_id = "test_session_123"
        
        # 先发送一条消息
        chat_request = {
            "text": "测试消息",
            "session_id": session_id
        }
        
        client.post("/chat", json=chat_request)
        
        # 获取对话历史
        response = client.get(f"/conversation_history/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "history" in data
        assert data["session_id"] == session_id
    
    def test_user_profile_update(self, client):
        """测试用户画像更新"""
        session_id = "test_session_456"
        
        profile_updates = {
            "role": "销售主管",
            "experience": "5年以上",
            "industry": "金融",
            "specialization": "B2B销售"
        }
        
        response = client.post(f"/update_user_profile/{session_id}", json=profile_updates)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "message" in data
    
    def test_session_management(self, client):
        """测试会话管理"""
        session_id = "test_session_789"
        
        # 先创建一个会话
        chat_request = {
            "text": "创建会话测试",
            "session_id": session_id
        }
        
        client.post("/chat", json=chat_request)
        
        # 清除会话
        response = client.delete(f"/clear_session/{session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_mcp_server_functionality(self):
        """测试MCP服务器功能"""
        # 测试文本处理
        text_response = await process_multimodal_input(
            content="这是一个测试文本",
            content_type="text"
        )
        
        assert text_response.processed_content == "这是一个测试文本"
        assert text_response.confidence > 0
        
        # 测试图像处理（模拟）
        fake_image = base64.b64encode(b"fake_image_data").decode()
        image_response = await process_multimodal_input(
            content=fake_image,
            content_type="image"
        )
        
        assert image_response.content_type == "image"
        assert "extracted_features" in image_response.dict()
    
    @pytest.mark.asyncio
    async def test_rag_system_functionality(self, setup_system):
        """测试RAG系统功能"""
        await setup_system
        
        # 测试知识检索
        response = await rag_system.query("销售漏斗的阶段有哪些？")
        
        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.confidence > 0
        assert len(response.sources) > 0
    
    @pytest.mark.asyncio
    async def test_agent_workflow(self, setup_system):
        """测试智能体工作流"""
        await setup_system
        
        # 测试简单查询
        result = await run_sales_agent("如何提高销售转化率？")
        
        assert "answer" in result
        assert "confidence" in result
        assert "reasoning_chain" in result
        assert len(result["answer"]) > 0
        
        # 测试复杂查询
        complex_result = await run_sales_agent(
            "我是一个新手销售，面对B2B客户时经常遇到价格异议，应该如何处理？请提供具体的话术和策略。"
        )
        
        assert complex_result["confidence"] > 0
        assert len(complex_result["reasoning_chain"]) > 1
    
    @pytest.mark.asyncio
    async def test_multimodal_agent_integration(self, setup_system):
        """测试多模态智能体集成"""
        await setup_system
        
        # 测试文本对话
        response = await chat_with_agent(
            text="我需要学习销售技巧",
            user_profile={
                "role": "销售新手",
                "experience": "0-1年"
            }
        )
        
        assert response.text_response is not None
        assert response.confidence > 0
        assert len(response.suggestions) > 0
    
    def test_error_handling(self, client):
        """测试错误处理"""
        # 测试无效的聊天请求
        invalid_request = {}
        
        response = client.post("/chat", json=invalid_request)
        assert response.status_code == 200
        
        data = response.json()
        # 应该能处理空请求，但可能返回错误
        assert "success" in data
    
    def test_api_documentation(self, client):
        """测试API文档可访问性"""
        # 测试OpenAPI文档
        response = client.get("/docs")
        assert response.status_code == 200
        
        # 测试ReDoc文档
        response = client.get("/redoc")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_performance_basic(self, client, setup_system):
        """基础性能测试"""
        await setup_system
        
        import time
        
        # 测试响应时间
        start_time = time.time()
        
        chat_request = {
            "text": "什么是销售？",
            "user_profile": {"role": "测试用户"}
        }
        
        response = client.post("/chat", json=chat_request)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 30  # 响应时间应该在30秒内
    
    def test_concurrent_requests(self, client):
        """测试并发请求处理"""
        import threading
        import time
        
        results = []
        
        def make_request():
            chat_request = {
                "text": "并发测试请求",
                "user_profile": {"role": "测试用户"}
            }
            
            response = client.post("/chat", json=chat_request)
            results.append(response.status_code)
        
        # 创建多个并发请求
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有请求完成
        for thread in threads:
            thread.join()
        
        # 检查所有请求都成功
        assert len(results) == 5
        assert all(status == 200 for status in results)


class TestSystemComponents:
    """系统组件单元测试"""
    
    @pytest.mark.asyncio
    async def test_knowledge_base_operations(self):
        """测试知识库操作"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("测试知识库内容：销售技巧和策略")
            temp_file = f.name
        
        try:
            # 测试添加文档
            item_ids = await rag_system.knowledge_base.add_document(temp_file, "text")
            assert len(item_ids) > 0
            
            # 测试搜索
            search_result = rag_system.knowledge_base.search("销售技巧", n_results=1)
            assert search_result.total_results > 0
            
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_multimodal_processing_pipeline(self):
        """测试多模态处理管道"""
        # 测试文本处理
        text_result = await process_multimodal_input(
            content="销售培训内容",
            content_type="text",
            metadata={"source": "test"}
        )
        
        assert text_result.processed_content == "销售培训内容"
        assert text_result.confidence == 1.0
        
        # 测试图像处理（模拟）
        fake_image_data = base64.b64encode(b"fake_image").decode()
        image_result = await process_multimodal_input(
            content=fake_image_data,
            content_type="image",
            metadata={"format": "png"}
        )
        
        assert image_result.content_type == "image"
        assert "extracted_features" in image_result.dict()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
