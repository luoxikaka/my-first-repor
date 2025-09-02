# 销售智能问答知识培训智能体 - 部署和测试指南

## 目录

1. [系统概述](#系统概述)
2. [环境要求](#环境要求)
3. [快速开始](#快速开始)
4. [详细部署步骤](#详细部署步骤)
5. [配置说明](#配置说明)
6. [测试指南](#测试指南)
7. [监控和维护](#监控和维护)
8. [故障排除](#故障排除)
9. [性能优化](#性能优化)
10. [安全配置](#安全配置)

## 系统概述

本系统是一个基于LangChain框架的多模态销售智能问答知识培训智能体，集成了以下核心技术：

- **Model Context Protocol (MCP)**: 处理多模态输入（文本、图像、音频、视频）
- **检索增强生成 (RAG)**: 从知识库中检索相关信息并生成答案
- **智能代理 (Agent)**: 基于LLM的推理和决策能力
- **LangGraph**: 复杂工作流编排和状态管理
- **多模态融合**: 统一处理不同类型的输入数据

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web 前端      │    │   API 网关      │    │  多模态处理     │
│                 │◄──►│                 │◄──►│   (MCP)        │
│ - 用户界面      │    │ - FastAPI       │    │ - 图像分析      │
│ - 文件上传      │    │ - 路由管理      │    │ - 语音识别      │
│ - 实时对话      │    │ - 认证授权      │    │ - 视频处理      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  智能代理       │    │   RAG 系统      │    │   知识库        │
│                 │◄──►│                 │◄──►│                 │
│ - LangGraph     │    │ - 向量检索      │    │ - 文档存储      │
│ - 工作流编排    │    │ - 语义搜索      │    │ - 向量数据库    │
│ - 状态管理      │    │ - 答案生成      │    │ - 元数据管理    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 环境要求

### 硬件要求

**最低配置：**
- CPU: 4核心
- 内存: 8GB RAM
- 存储: 20GB 可用空间
- 网络: 稳定的互联网连接

**推荐配置：**
- CPU: 8核心或更多
- 内存: 16GB RAM 或更多
- 存储: 50GB SSD
- GPU: 可选，用于本地模型推理

### 软件要求

**必需软件：**
- Python 3.9+
- pip 包管理器
- Git 版本控制

**可选软件：**
- Docker 和 Docker Compose（推荐用于部署）
- Nginx（用于生产环境反向代理）
- PostgreSQL（用于持久化存储）
- Redis（用于缓存）

### API 密钥要求

- **OpenAI API Key**: 必需，用于LLM推理
- **其他LLM提供商**: 可选，如Anthropic、Hugging Face等

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd sales-intelligence-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置您的API密钥
nano .env
```

### 3. 一键部署

```bash
# 使用部署脚本（推荐）
./deploy.sh

# 或者选择特定部署模式
./deploy.sh --docker    # Docker部署
./deploy.sh --local     # 本地部署
./deploy.sh --test      # 只运行测试
```

### 4. 访问系统

部署完成后，您可以通过以下地址访问系统：

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 详细部署步骤

### Docker 部署（推荐）

Docker部署是最简单和可靠的部署方式，适合大多数用户。

#### 1. 安装 Docker

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# CentOS/RHEL
sudo yum install docker docker-compose

# macOS
brew install docker docker-compose
```

#### 2. 启动服务

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f sales-agent
```

#### 3. 服务组件

Docker部署包含以下服务：

- **sales-agent**: 主应用服务
- **postgres**: PostgreSQL数据库
- **redis**: Redis缓存
- **nginx**: 反向代理（可选）
- **prometheus**: 监控服务（可选）
- **grafana**: 可视化仪表板（可选）

### 本地部署

本地部署适合开发和测试环境。

#### 1. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate     # Windows
```

#### 2. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. 初始化知识库

```bash
# 创建知识库目录
mkdir -p knowledge_base/{documents,images,audio}

# 添加示例文档
cp examples/* knowledge_base/documents/
```

#### 4. 启动应用

```bash
# 开发模式
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 生产环境部署

生产环境部署需要考虑安全性、可扩展性和可靠性。

#### 1. 系统服务配置

```bash
# 创建系统用户
sudo useradd -r -s /bin/false sales-agent

# 创建systemd服务文件
sudo tee /etc/systemd/system/sales-agent.service > /dev/null <<EOF
[Unit]
Description=Sales Intelligence Agent
After=network.target

[Service]
Type=exec
User=sales-agent
Group=sales-agent
WorkingDirectory=/opt/sales-agent
Environment=PATH=/opt/sales-agent/venv/bin
ExecStart=/opt/sales-agent/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable sales-agent
sudo systemctl start sales-agent
```

#### 2. Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # 文件上传大小限制
    client_max_body_size 100M;
    
    # 静态文件缓存
    location /static/ {
        alias /opt/sales-agent/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### 3. SSL 证书配置

```bash
# 使用 Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 配置说明

### 环境变量配置

系统通过环境变量进行配置，主要配置项包括：

#### LLM 配置

```bash
# OpenAI 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1
DEFAULT_LLM_MODEL=gpt-3.5-turbo
MAX_TOKENS=2000
TEMPERATURE=0.1

# 其他LLM提供商（可选）
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACE_API_TOKEN=your_hf_token
```

#### 数据库配置

```bash
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost:5432/sales_agent

# Redis
REDIS_URL=redis://localhost:6379/0

# 向量数据库
CHROMA_PERSIST_DIRECTORY=./knowledge_base_db
COLLECTION_NAME=sales_knowledge
```

#### 应用配置

```bash
# 服务器配置
HOST=0.0.0.0
PORT=8000
WORKERS=4
RELOAD=false

# 安全配置
SECRET_KEY=your_secret_key
JWT_SECRET_KEY=your_jwt_secret
CORS_ORIGINS=*

# 文件上传配置
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=pdf,doc,docx,txt,md,png,jpg,jpeg,gif,mp3,wav,mp4,avi
```

#### RAG 系统配置

```bash
# 文档处理
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_RESULTS=5

# 嵌入模型
EMBEDDING_MODEL=all-MiniLM-L6-v2

# 多模态处理
IMAGE_MAX_SIZE=10MB
AUDIO_MAX_SIZE=25MB
VIDEO_MAX_SIZE=100MB
```

### 知识库配置

知识库支持多种文档格式和组织方式：

#### 目录结构

```
knowledge_base/
├── documents/          # 文本文档
│   ├── sales_manual.pdf
│   ├── product_specs.docx
│   └── training_materials.md
├── images/            # 图像资料
│   ├── product_photos/
│   ├── charts/
│   └── presentations/
├── audio/             # 音频文件
│   ├── training_recordings/
│   └── customer_calls/
└── metadata.json      # 元数据配置
```

#### 元数据配置

```json
{
  "collections": {
    "sales_basics": {
      "description": "销售基础知识",
      "priority": 1,
      "tags": ["基础", "入门"]
    },
    "advanced_techniques": {
      "description": "高级销售技巧",
      "priority": 2,
      "tags": ["高级", "技巧"]
    }
  },
  "processing_rules": {
    "pdf": {
      "extract_images": true,
      "ocr_enabled": true
    },
    "audio": {
      "transcription_enabled": true,
      "language": "zh-CN"
    }
  }
}
```

## 测试指南

### 自动化测试

系统包含完整的测试套件，涵盖单元测试、集成测试和端到端测试。

#### 运行所有测试

```bash
# 使用部署脚本运行测试
./deploy.sh --test

# 或者手动运行
python -m pytest tests/ -v --tb=short
```

#### 测试覆盖率

```bash
# 安装覆盖率工具
pip install pytest-cov

# 运行测试并生成覆盖率报告
python -m pytest tests/ --cov=app --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html
```

### 功能测试

#### 1. 基础功能测试

**健康检查测试：**

```bash
curl -X GET http://localhost:8000/health
```

预期响应：
```json
{
  "status": "healthy",
  "components": {
    "api": "healthy",
    "rag_system": "healthy",
    "multimodal_agent": "healthy",
    "mcp_server": "healthy"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**文本对话测试：**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "text": "什么是销售漏斗？",
    "user_profile": {
      "role": "销售新手",
      "experience": "0-1年"
    }
  }'
```

#### 2. 多模态功能测试

**图像上传测试：**

```bash
# 将图像转换为base64
base64_image=$(base64 -w 0 test_image.png)

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"请分析这张销售图表\",
    \"image_base64\": \"$base64_image\"
  }"
```

**知识库上传测试：**

```bash
curl -X POST http://localhost:8000/upload_knowledge \
  -F "file=@test_document.pdf" \
  -F "content_type=pdf" \
  -F "metadata={\"category\": \"sales_training\"}"
```

#### 3. 性能测试

**并发测试：**

```bash
# 使用 Apache Bench 进行并发测试
ab -n 100 -c 10 -T application/json -p test_request.json http://localhost:8000/chat
```

**负载测试：**

```python
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            task = session.post(
                'http://localhost:8000/chat',
                json={'text': f'测试请求 {i}'}
            )
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"100个请求耗时: {end_time - start_time:.2f}秒")
        print(f"平均响应时间: {(end_time - start_time) / 100:.2f}秒")

asyncio.run(load_test())
```

### 手动测试

#### 1. Web界面测试

1. **访问主页**: http://localhost:8000
2. **用户画像设置**: 填写角色、经验、行业信息
3. **文本对话**: 输入销售相关问题
4. **文件上传**: 上传PDF、Word文档到知识库
5. **多模态交互**: 上传图片、音频文件
6. **会话管理**: 测试会话清除和历史导出

#### 2. API 测试

使用 Postman 或类似工具测试所有 API 端点：

- `GET /health` - 健康检查
- `POST /chat` - 聊天接口
- `POST /upload_knowledge` - 知识库上传
- `GET /conversation_history/{session_id}` - 对话历史
- `POST /update_user_profile/{session_id}` - 用户画像更新
- `DELETE /clear_session/{session_id}` - 会话清除

## 监控和维护

### 日志管理

#### 日志配置

系统使用结构化日志记录，支持多种输出格式：

```python
# 日志配置示例
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

#### 日志查看

```bash
# 实时查看日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log

# 查看特定时间段的日志
grep "2024-01-01" logs/app.log
```

### 性能监控

#### Prometheus 指标

系统暴露以下 Prometheus 指标：

- `http_requests_total` - HTTP请求总数
- `http_request_duration_seconds` - 请求响应时间
- `chat_sessions_active` - 活跃会话数
- `knowledge_base_documents_total` - 知识库文档数量
- `rag_retrieval_time_seconds` - RAG检索时间
- `llm_inference_time_seconds` - LLM推理时间

#### Grafana 仪表板

预配置的 Grafana 仪表板包含：

1. **系统概览**
   - 请求量和响应时间
   - 错误率和成功率
   - 系统资源使用情况

2. **业务指标**
   - 用户会话统计
   - 知识库使用情况
   - 多模态内容处理量

3. **性能分析**
   - 各组件响应时间分布
   - 缓存命中率
   - 数据库查询性能

### 数据备份

#### 知识库备份

```bash
#!/bin/bash
# 知识库备份脚本

BACKUP_DIR="/backup/knowledge_base"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份文档文件
tar -czf $BACKUP_DIR/documents_$DATE.tar.gz knowledge_base/

# 备份向量数据库
tar -czf $BACKUP_DIR/vector_db_$DATE.tar.gz knowledge_base_db/

# 备份配置文件
cp .env $BACKUP_DIR/env_$DATE.backup

# 清理旧备份（保留30天）
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

#### 数据库备份

```bash
# PostgreSQL 备份
pg_dump -h localhost -U sales_agent sales_agent > backup_$(date +%Y%m%d).sql

# Redis 备份
redis-cli --rdb backup_$(date +%Y%m%d).rdb
```

### 系统更新

#### 应用更新

```bash
# 1. 备份当前版本
cp -r /opt/sales-agent /opt/sales-agent.backup

# 2. 拉取最新代码
git pull origin main

# 3. 更新依赖
pip install -r requirements.txt

# 4. 运行数据库迁移（如果有）
python manage.py migrate

# 5. 重启服务
sudo systemctl restart sales-agent
```

#### Docker 更新

```bash
# 1. 拉取最新镜像
docker-compose pull

# 2. 重新构建和启动
docker-compose up -d --build

# 3. 清理旧镜像
docker image prune -f
```

## 故障排除

### 常见问题

#### 1. 应用启动失败

**症状**: 应用无法启动或立即退出

**可能原因**:
- 环境变量配置错误
- 依赖包缺失或版本不兼容
- 端口被占用
- 权限问题

**解决方案**:

```bash
# 检查环境变量
cat .env | grep -v "^#"

# 检查依赖
pip check

# 检查端口占用
netstat -tlnp | grep 8000

# 检查权限
ls -la logs/
```

#### 2. API 响应缓慢

**症状**: API请求响应时间过长

**可能原因**:
- LLM API 限流
- 知识库检索效率低
- 数据库查询慢
- 内存不足

**解决方案**:

```bash
# 检查系统资源
htop
free -h
df -h

# 检查数据库性能
# PostgreSQL
SELECT * FROM pg_stat_activity WHERE state = 'active';

# 检查日志中的慢查询
grep "slow" logs/app.log
```

#### 3. 知识库检索不准确

**症状**: RAG系统返回不相关的结果

**可能原因**:
- 文档分块策略不当
- 嵌入模型不适合
- 检索参数设置不当

**解决方案**:

```python
# 调整分块参数
CHUNK_SIZE = 500  # 减小块大小
CHUNK_OVERLAP = 100  # 增加重叠

# 尝试不同的嵌入模型
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 调整检索参数
MAX_RETRIEVAL_RESULTS = 10  # 增加检索结果数量
```

#### 4. 多模态处理失败

**症状**: 图像、音频处理出错

**可能原因**:
- 文件格式不支持
- 文件大小超限
- 处理模型加载失败

**解决方案**:

```bash
# 检查文件格式支持
python -c "from PIL import Image; print(Image.EXTENSION)"

# 检查模型文件
ls -la ~/.cache/huggingface/transformers/

# 重新下载模型
python -c "from transformers import BlipProcessor; BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')"
```

### 调试技巧

#### 1. 启用详细日志

```python
# 在 main.py 中添加
import logging
logging.getLogger("app").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)
```

#### 2. 使用调试模式

```bash
# 启动调试模式
uvicorn main:app --reload --log-level debug
```

#### 3. 性能分析

```python
# 添加性能分析装饰器
import time
import functools

def timing_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper
```

## 性能优化

### 系统级优化

#### 1. 内存优化

```python
# 配置模型缓存
import torch
torch.cuda.empty_cache()  # 清理GPU缓存

# 限制模型并发数
MAX_CONCURRENT_REQUESTS = 10
```

#### 2. 数据库优化

```sql
-- PostgreSQL 索引优化
CREATE INDEX idx_documents_content ON documents USING gin(to_tsvector('chinese', content));
CREATE INDEX idx_sessions_created_at ON sessions(created_at);

-- 查询优化
EXPLAIN ANALYZE SELECT * FROM documents WHERE content ILIKE '%销售%';
```

#### 3. 缓存策略

```python
# Redis 缓存配置
CACHE_CONFIG = {
    "embedding_cache_ttl": 3600,  # 嵌入缓存1小时
    "rag_result_cache_ttl": 1800,  # RAG结果缓存30分钟
    "session_cache_ttl": 7200,    # 会话缓存2小时
}
```

### 应用级优化

#### 1. 异步处理

```python
# 使用异步队列处理耗时任务
import asyncio
from asyncio import Queue

async def background_processor():
    while True:
        task = await task_queue.get()
        await process_task(task)
        task_queue.task_done()
```

#### 2. 批处理优化

```python
# 批量处理嵌入生成
async def batch_embed_documents(documents, batch_size=32):
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_embeddings = await embedding_model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

#### 3. 模型优化

```python
# 模型量化
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("model_name")
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 安全配置

### 认证和授权

#### 1. JWT 认证

```python
# JWT 配置
JWT_CONFIG = {
    "algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "refresh_token_expire_days": 7
}

# 认证中间件
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

#### 2. API 限流

```python
# 使用 slowapi 进行限流
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    # 聊天逻辑
    pass
```

### 数据安全

#### 1. 数据加密

```python
# 敏感数据加密
from cryptography.fernet import Fernet

def encrypt_sensitive_data(data: str) -> str:
    key = os.environ.get("ENCRYPTION_KEY")
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    return encrypted_data.decode()
```

#### 2. 输入验证

```python
# 严格的输入验证
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    text: Optional[str] = None
    
    @validator('text')
    def validate_text(cls, v):
        if v and len(v) > 10000:
            ra
(Content truncated due to size limit. Use line ranges to read in chunks)


