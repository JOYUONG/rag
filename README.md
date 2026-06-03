# RAG v6.0 — 本地知识库问答系统

基于检索增强生成（RAG）的全功能文档问答系统，支持多种文档格式的解析、向量存储、语义检索、重排序和 LLM 生成回答。提供 Gradio Web UI 界面。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **多格式文档解析** | PDF、DOCX、XLSX、PPTX、Markdown、TXT、图片（OCR）、CSV、HTML、JSON、EPUB、音频（Whisper 转写） |
| **多 Embedding 后端** | Jina API（远程） / sentence-transformers（本地） |
| **多 Reranker 后端** | Jina API（远程） / cross-encoder（本地） |
| **多图片描述后端** | Gemini API / Jina API / 本地 CLIP |
| **混合分块策略** | 段落 / 句子 / 滑动窗口 / Markdown 标题 / 固定 Token |
| **向量存储** | HNSW（高性能近似检索） / Flat（暴力精确检索） |
| **MySQL 持久化** | 文档元数据、用户反馈的持久化存储 |
| **多轮对话记忆** | 支持带记忆的连续对话 |
| **反馈学习** | 用户反馈写入数据库，可用于后续优化 |
| **Web UI** | Gradio 界面，支持文档上传、参数调节、反馈提交 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│                   Gradio Web UI                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ 文档上传管理  │  │  对话问答界面  │  │  反馈收集    │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  │
└─────────┼────────────────┼──────────────────┼─────────┘
          │                │                  │
          ▼                ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│ DocumentIngestion│ │ QueryPipeline│ │ MySQLDatabase│
│   Pipeline       │ │              │ │  (持久化)     │
│ ┌─────────────┐ │ │ ┌──────────┐ │ └──────────────┘
│ │ Docling/    │ │ │ │ Vector   │ │
│ │ Unstructured│ │ │ │ Search   │ │
│ │ 解析器       │ │ │ └────┬─────┘ │
│ └──────┬──────┘ │ │      │       │
│        ▼        │ │ ┌────▼─────┐ │
│ ┌─────────────┐ │ │ │ Reranker │ │
│ │ Document    │ │ │ └────┬─────┘ │
│ │ Chunker     │ │ │      ▼       │
│ └──────┬──────┘ │ │ ┌──────────┐ │
│        ▼        │ │ │ LLM 生成  │ │
│ ┌─────────────┐ │ │ └──────────┘ │
│ │ Embedding   │ │ └──────────────┘
│ │ Provider    │ │
│ └──────┬──────┘ │
│        ▼        │
│ ┌─────────────┐ │
│ │ VectorStore │ │
│ │ (HNSW/Flat) │ │
│ └─────────────┘ │
└─────────────────┘
```

---

## 环境变量

在项目根目录创建 `.env` 文件，配置以下变量：

```bash
# ============ 必需 ============
# Jina API Key（使用 Jina Embedding / Reranker / 图片描述时必需）
JINA_API_KEY=your_jina_api_key

# LLM API Key（二选一，用于生成回答）
SILICONFLOW_API_KEY=your_siliconflow_key   # 硅基流动（默认）
OPENAI_API_KEY=your_openai_key             # OpenAI（可选）

# ============ MySQL（可选但推荐） ============
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=rag_db

# ============ 图片描述（可选） ============
GEMINI_API_KEY=your_gemini_key             # Gemini 图片描述
VISION_API_KEY=your_vision_key             # Jina Vision API
VISION_API_BASE_URL=https://api.jina.ai    # Vision API 地址
VISION_MODEL_NAME=jina-embeddings-v3       # Vision 模型名

# ============ 音频转写（可选） ============
WHISPER_API_KEY=your_whisper_key
WHISPER_API_BASE_URL=https://api.example.com
WHISPER_MODEL_NAME=whisper-1
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install gradio python-dotenv requests mysql-connector-python
pip install sentence-transformers  # 本地 Embedding/Reranker
pip install docling                 # 文档解析（推荐）
pip install Pillow                  # 图片处理
```

### 2. 配置环境变量

复制并编辑 `.env` 文件，填入你的 API Key：

```bash
cp .env.example .env
# 编辑 .env 填入实际值
```

### 3. 启动服务

```bash
python rag-v6.0.py
```

启动后访问 `http://127.0.0.1:7860` 打开 Web UI。

---

## 使用指南

### 文档上传与索引

1. 切换到 **📚 文档管理** 标签页
2. 上传一个或多个文档文件
3. 选择文档类型（通用 / 论文 / 代码 / 法律）
4. 点击 **🚀 上传并处理文档**
5. 等待解析、分块、向量化完成

### 知识库问答

1. 切换到 **💬 知识库问答** 标签页
2. 调整检索参数：
   - **检索文档数量**：返回的候选文档片段数量（1-20）
   - **重排序数量**：重排序后保留的片段数量（1-10）
   - **温度**：LLM 生成的随机性（0.0-1.0）
3. 输入问题，点击发送
4. 查看回答及引用来源
5. 对回答给出反馈（👍/👎 + 评论）

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 检索文档数量 | 5 | 初始向量检索返回的候选数量，越多召回率越高 |
| 重排序数量 | 3 | 经 Reranker 精排后保留的数量，越少精度越高 |
| 温度 | 0.7 | LLM 生成温度，越低越确定性，越高越多样 |
| 多轮对话记忆 | 开启 | 启用后会携带历史对话上下文 |
| 文档类型 | 通用 | 影响分块策略和提示词模板的选择 |

---

## 代码结构

### 核心类

```
├── 配置层
│   └── EnvConfig                  # 环境变量加载与管理
│
├── 数据库层
│   └── MySQLDatabase              # MySQL 连接、表创建、CRUD
│
├── Embedding 层
│   └── EmbeddingProvider          # Jina API / 本地 sentence-transformers
│
├── 图片处理层
│   └── ImageDescriber             # Gemini / Jina / 本地 CLIP 图片描述
│
├── Reranker 层
│   └── RerankerProvider           # Jina API / 本地 cross-encoder 重排序
│
├── 分块层
│   └── DocumentChunker            # 5 种分块策略
│
├── 向量存储层
│   └── VectorStore                # HNSW / Flat 索引，余弦相似度检索
│
├── 文档解析层
│   └── DocumentParser             # Docling / Unstructured / 后备解析器
│
├── 管道层
│   ├── DocumentIngestionPipeline  # 文档摄取管道（解析→分块→向量化→存储）
│   └── QueryPipeline              # 查询管道（检索→重排序→LLM 生成）
│
├── Agent 层
│   └── RAGAgent                   # 带记忆的 Agent 封装
│
└── UI 层
    └── Gradio UI                  # Web 界面
```

### 关键方法

| 方法 | 所属类 | 说明 |
|------|--------|------|
| `parse_document()` | DocumentParser | 解析文档为结构化元素（含图片提取） |
| `chunk_document()` | DocumentChunker | 将解析结果按策略分块 |
| `add_texts()` | VectorStore | 添加文本到向量索引 |
| `search()` | VectorStore | 语义相似度检索 |
| `rerank()` | RerankerProvider | 对检索结果精排序 |
| `describe_image()` | ImageDescriber | 生成图片文本描述 |
| `ingest_document()` | DocumentIngestionPipeline | 端到端文档摄取 |
| `query()` | QueryPipeline | 端到端查询（检索+重排序+生成） |
| `chat()` | RAGAgent | 带记忆的多轮对话 |
| `save_feedback()` | MySQLDatabase | 保存用户反馈 |

---

## 分块策略

| 策略 | 适用场景 | 特点 |
|------|----------|------|
| `paragraph` | 通用文档 | 按空行分割，保持段落完整性 |
| `sentence` | 长段落文本 | 按句号/问号/感叹号分割 |
| `sliding_window` | 需要上下文重叠 | 固定窗口大小 + 步长滑动 |
| `markdown_heading` | Markdown 文档 | 按标题层级分割，保留标题作为前缀 |
| `fixed_token` | 精确控制块大小 | 按字符数固定分割 |

默认使用 `paragraph` 策略；Markdown 文件自动切换为 `markdown_heading`。

---

## 向量索引

| 索引类型 | 数据量 | 检索速度 | 精度 |
|----------|--------|----------|------|
| **Flat** | < 10,000 条 | 较慢 | 精确（100%） |
| **HNSW** | ≥ 10,000 条 | 快 | 近似（~95%+） |

系统根据数据量自动选择索引类型。HNSW 参数：`M=16`, `efConstruction=200`, `efSearch=128`。

---

## MySQL 表结构

```sql
-- 文档元数据
CREATE TABLE documents (
    id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(500),
    doc_type VARCHAR(50),
    chunk_count INT,
    upload_time DATETIME,
    file_hash VARCHAR(64),
    status VARCHAR(20) DEFAULT 'active'
);

-- 用户反馈
CREATE TABLE feedbacks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query TEXT,
    answer TEXT,
    feedback_type VARCHAR(20),
    feedback_text TEXT,
    create_time DATETIME
);
```

---

## LLM 支持

| 提供商 | API 格式 | 默认模型 |
|--------|----------|----------|
| 硅基流动（默认） | OpenAI 兼容 | `Qwen/Qwen3-8B` |
| OpenAI | OpenAI | `gpt-3.5-turbo` |

通过 `LLMProvider` 类统一封装，支持任何 OpenAI 兼容 API。

---

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| MySQL 连接失败 | 检查 `.env` 中 MySQL 配置；确保 MySQL 服务已启动 |
| Jina API 报错 | 检查 `JINA_API_KEY` 是否有效；检查网络连接 |
| 本地模型加载慢 | 首次运行需下载模型，后续会从缓存加载 |
| 文档解析失败 | 确认已安装 `docling`；检查文件格式是否支持 |
| 向量检索无结果 | 确认文档已成功索化；检查索引文件是否存在 |
| Ollama 连接失败 | 确保 Ollama 服务运行中：`ollama serve` |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `rag-v6.0.py` | 主程序（全部代码） |
| `.env` | 环境变量配置（需自行创建） |
| `data/vector_indexes/` | 向量索引持久化目录 |
| `data/uploads/` | 上传文件存储目录 |
| `data/images/` | 提取的图片存储目录 |

---

## 版本历史

- **v6.0** — 当前版本
  - 新增 Docling 解析器，文档解析质量大幅提升
  - 新增 HNSW 向量索引，支持大规模数据检索
  - 新增 MySQL 持久化，文档元数据和反馈数据持久存储
  - 新增图片描述功能（Gemini / Jina / 本地 CLIP）
  - 新增 Reranker 精排序（Jina API / 本地 cross-encoder）
  - 新增多轮对话记忆
  - 重构为模块化类设计
