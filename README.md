# 🧠 个人私有知识库 RAG 系统

> 基于 LangChain + 向量检索的本地知识库问答系统，从零搭建一个能读懂你文档的 AI 助手。

大语言模型（LLM）虽然强大，但存在三个核心痛点：

- **知识过时** — 训练数据有截止日期，无法获取最新信息
- **幻觉问题** — 会"一本正经地胡说八道"
- **无法访问私有数据** — 不了解你的个人文档、内部资料

本项目通过 **RAG（Retrieval-Augmented Generation，检索增强生成）** 技术，让你在本地构建一个基于个人文档的智能问答系统，数据全程不出本机，**100% 私密**。

---

## 📌 版本路线图

本项目采用**渐进式迭代**开发，每个版本都是一个完整可运行的脚本，逐步叠加新能力：

| 版本 | 核心特性 | LLM 后端 | 关键技术 |
|:---:|---|---|---|
| **v2.0** | 基线 RAG | DashScope（云端） | 文档加载、文本分割、向量检索、MMR、LCEL 链 |
| **v3.0** | 全本地化 | Ollama（本地） | 零成本、离线运行、本地 LLM 推理 |
| **v4.0** | 精排优化 | Ollama（本地） | CrossEncoder 重排序，两阶段检索（召回→精排） |
| **v5.0** | 全局理解 | Ollama（本地） | RAPTOR 递归摘要树、UMAP 降维、GMM 聚类 |
| **v6.0** | 记忆与个性 | Ollama（本地） | 三层记忆系统、用户画像、动态人格、跨会话记忆 |

---

## 🏗️ 系统架构

```
文档 (PDF/Word/TXT)
       │
       ▼
┌─────────────┐
│  文档加载     │  pypdf / docx2txt / 读取 TXT
└─────┬───────┘
      ▼
┌─────────────┐
│  文本分割     │  中文感知分隔符，chunk_size 可调
└─────┬───────┘
      ▼
┌─────────────┐
│  向量嵌入     │  BAAI/bge-small-zh-v1.5（512维）
└─────┬───────┘
      ▼
┌─────────────┐
│  向量数据库   │  Chroma（持久化存储）
└─────┬───────┘
      ▼
┌─────────────┐     ┌──────────────┐
│  向量检索     │────▶│ 重排序（v4+） │  CrossEncoder 精排
└─────┬───────┘     └──────┬───────┘
      ▼                    ▼
┌─────────────┐     ┌──────────────┐
│ RAPTOR 树   │     │ 记忆系统（v6）│  短期/长期/反思
│  （v5+）     │     └──────┬───────┘
└─────┬───────┘            │
      ▼                    ▼
┌─────────────────────────────┐
│         LLM 生成回答         │  Ollama / DashScope
└─────────────────────────────┘
```

---

## 📁 项目结构

```
llm/
├── rag-v2.0.py              # v2.0 基线 RAG（云端 API）
├── rag-v3.0.py              # v3.0 本地 Ollama（离线运行）
├── rag-v4.0.py              # v4.0 + CrossEncoder 重排序
├── rag-v5.0.py              # v5.0 + RAPTOR 递归摘要树
├── rag-v6.0.py              # v6.0 + 三层记忆 & 动态人格
├── version.py               # GPU/CUDA 检测工具
├── .env                     # API 密钥配置（v2.0 使用）
├── bge-small-zh-v1.5/       # 本地嵌入模型（bge-small-zh）
├── docx/                    # 示例文档
├── chroma_db/               # 向量数据库（v2.0/v4.0）
├── chroma_db_raptor/        # 向量数据库（v5.0 RAPTOR 节点）
├── chroma_db_knowledge/     # 向量数据库（v6.0 知识库）
├── raptor_tree.pkl          # RAPTOR 树缓存（v5.0）
└── memory.db                # 记忆数据库（v6.0 SQLite）
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python** 3.10+
- **Ollama**（v3.0 及以上版本需要）
- **内存** 建议 8GB+（v5.0/v6.0 对内存较敏感）
- **GPU** 可选（支持 CUDA 加速，CPU 亦可运行）

### 2. 安装依赖

```bash
pip install langchain langchain-community langchain-core langchain-text-splitters \
            langchain-ollama langchain-chroma langchain-openai \
            sentence-transformers chromadb pypdf docx2txt \
            python-dotenv openai numpy scikit-learn umap-learn
```

### 3. 安装 Ollama 并拉取模型

```bash
# 安装 Ollama（访问 https://ollama.com 下载）
# 启动 Ollama 服务
ollama serve

# 拉取模型
ollama pull qwen3:8b
```

### 4. 运行

每个版本是独立脚本，可直接运行：

```bash
# v2.0 — 需要 DashScope API 密钥（配置在 .env 中）
python rag-v2.0.py

# v3.0 — 需要 Ollama 运行中
python rag-v3.0.py

# v4.0 — 首次运行会自动下载重排序模型
python rag-v4.0.py

# v5.0 — 首次运行会构建 RAPTOR 树（耗时较长）
python rag-v5.0.py

# v6.0 — 含记忆系统，支持跨会话记忆
python rag-v6.0.py
```

运行后进入交互式问答循环，输入问题即可获得回答，输入 `q`、`quit` 或 `exit` 退出。

### 5. 使用自己的文档

修改各脚本 `main()` 函数中的 `DOC_PATH` 变量，指向你自己的文档路径：

```python
DOC_PATH = "./docx"  # 修改为你的文档目录或文件路径
```

支持的格式：**PDF**、**Word (.docx)**、**TXT**（自动处理 UTF-8/GBK 编码）

---

## ⚙️ 配置参数

各版本顶部的 `RAGConfig` 类包含核心配置项：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `chunk_size` | 500 (v2-4) / 200 (v5-6) | 文本分块大小（字符数） |
| `chunk_overlap` | 50 | 分块重叠字符数 |
| `k_retrieval` | 3~8 | 检索候选数量 |
| `k_rerank` | 3~5 | 精排后保留数量 |
| `persist_directory` | `./chroma_db` | 向量数据库存储路径 |
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |
| `rerank_model` | `BAAI/bge-reranker-base` | 重排序模型（v4+） |
| `temperature` | 0.1 (v2-4) / 0 (v5-6) | LLM 生成温度 |
| `summary_model` | `qwen3:8b` | Ollama 模型名称 |
| `raptor_max_depth` | 2 | RAPTOR 树最大深度（v5） |
| `memory_db_path` | `./memory.db` | 记忆数据库路径（v6） |
| `short_term_window` | 10 | 短期记忆窗口轮次（v6） |
| `reflection_interval` | 10 | 人格反思间隔轮次（v6） |

---

## 🔧 使用的核心模型

| 模型 | 用途 | 说明 |
|---|---|---|
| **BAAI/bge-small-zh-v1.5** | 文本嵌入 | 中文优化，512 维，已内置在项目中 |
| **BAAI/bge-reranker-base** | 重排序 | CrossEncoder，首次使用自动下载 |
| **qwen3:8b** (Ollama) | LLM 推理 | 本地运行，也可换用其他 Ollama 模型 |
| **DashScope/Qwen** | LLM 推理 | v2.0 使用，需 API 密钥 |

---

## 💡 设计亮点

- **🇨🇳 中文优先** — 中文感知分隔符、中文嵌入模型、中文提示词、UTF-8/GBK 自动检测
- **📖 渐进式教学** — 每个版本独立完整，代码注释详尽，适合从零学习 RAG
- **💾 智能缓存** — 向量库持久化、RAPTOR 树序列化、记忆 SQLite 存储，避免重复计算
- **🖥️ 内存适配** — v5.0/v6.0 针对 8GB 内存优化，提供分批处理策略
- **🔒 完全私密** — v3.0+ 全链路本地化，数据不离开你的电脑

---

## 📜 License

本项目仅供学习与研究使用。
