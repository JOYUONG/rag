# -*- coding: utf-8 -*-
"""
RAG 个人私有知识库 v6.0（完整可运行版）
核心升级：三层记忆系统 + 个性化养成
基础特性：零成本、本地大模型（Ollama）、100% 离线
进阶特性：跨会话记忆、用户画像、动态养成
8G内存可运行，新手友好！
所有代码均有详细注释，复制粘贴即可运行，只需修改文档路径（可选）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
os.environ["TRANSFORMERS_OFFLINE"] = "0" # 允许联网下载 
os.environ["HF_HUB_OFFLINE"] = "0" # 然后再import其他模块 
import json
import sqlite3
import numpy as np
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
 
# 文档处理相关（复用v5.0，负责加载PDF/Word/TXT文档）
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
# 向量库相关（负责存储记忆向量，快速检索）
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# 提示词和对话链相关（负责生成个性化回答）
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM as Ollama
# 重排序相关（让检索结果更精准）
from sentence_transformers import CrossEncoder
 
 
# ==================== 配置类（所有可调整的参数都在这里，新手可按需修改） ====================
class RAGConfig:
    """v6.0 统一配置项，新手可以根据自己的电脑配置调整"""
    # 文本拆分参数（文档分块用，不用改）
    chunk_size = 200  # 每块文本的长度
    chunk_overlap = 50  # 块与块之间的重叠部分（避免拆分后语义断裂）
    
    # 检索参数（不用改）
    k_retrieval = 8  # 初始检索数量
    k_rerank = 5     # 重排序后保留的数量
    
    # 向量库参数（不用改）
    persist_directory = "./chroma_db"  # 向量库存储路径
    embedding_model = "BAAI/bge-small-zh-v1.5"  # 向量生成模型（中文友好，轻量）
    
    # 重排序模型（不用改）
    rerank_model = "BAAI/bge-reranker-base"
    
    # 本地大模型参数（新手重点看这里！）
    temperature = 0  # 回答的随机性（0=严谨，1=活泼，建议0）
    summary_model = "qwen3:8b"  # 要和你拉取的Ollama模型一致（比如拉取的是qwen3:1.8b-chat，就改成这个）
    
    # ========== v6.0 新增记忆配置（新手可按需修改） ==========
    # 数据库路径（记忆存储的位置，不用改）
    memory_db_path = "./memory.db"  # 记忆文件，删除这个文件就重置所有记忆
    
    # 记忆参数（不用改）
    short_term_window = 10  # 短期记忆保留最近10轮对话
    long_term_top_k = 5      # 长期记忆每次检索5条
    reflection_interval = 10  # 每10轮对话触发一次反思
    
    # 个性化参数（初始值，不用改，会自动调整）
    personality_dimensions = ["热情度", "专业度", "幽默感", "共情力"]
    default_personality = [50, 70, 30, 50]  # 默认个性：中等热情、较高专业、偏低幽默、中等共情
 
 
# ==================== 记忆数据结构（定义“记忆”的样子，不用改） ====================
@dataclass      # 定义一个数据类，用于存储记忆事件
class MemoryEvent:
    """单次交互记忆（比如你问一句话，AI答一句话，这就是一次交互记忆）"""
    session_id: str  # 会话ID（区分不同的聊天会话）
    user_input: str  # 你说的话
    agent_response: str  # AI说的话
    summary: str = ""  # 交互摘要（简化版，方便存储）
    topics: List[str] = None  # 话题标签（比如“随机森林”“调参”）
    importance: float = 1.0  # 重要性（越高，越优先被检索）
    timestamp: str = ""  # 时间戳（什么时候聊的）
    embedding: List[float] = None  # 记忆向量（用于快速检索）
    
    def __post_init__(self):
        # 自动填充时间戳（不用手动设置）
        if self.timestamp == "":
            self.timestamp = datetime.now().isoformat()
        # 初始化话题标签（空列表）
        if self.topics is None:
            self.topics = []
 
 
@dataclass      # 定义一个数据类，用于存储用户画像
class UserProfile:
    """用户画像（存储你的基本信息，比如名字、专业背景）"""
    user_id: str  # 用户ID（区分不同用户，默认是default_user）
    name: str = ""  # 你的名字（比如张三）
    background: str = ""  # 你的专业背景（比如数据分析师）
    preferences: Dict[str, Any] = None  # 你的偏好（比如喜欢通俗解释）
    interaction_count: int = 0  # 交互次数（聊了多少轮）
    last_active: str = ""  # 最后活跃时间
    created_at: str = ""  # 首次交互时间
    
    def __post_init__(self):
        # 初始化偏好（空字典）
        if self.preferences is None:
            self.preferences = {}
        # 自动填充首次交互时间
        if self.created_at == "":
            self.created_at = datetime.now().isoformat()
        # 自动填充最后活跃时间
        if self.last_active == "":
            self.last_active = datetime.now().isoformat()
 
 
@dataclass      # 定义一个数据类，用于存储个性参数
class Personality:
    """个性化参数（AI的性格，会自动调整）"""
    warmth: float = 50.0      # 热情度（0-100）
    expertise: float = 70.0   # 专业度（0-100）
    humor: float = 30.0       # 幽默感（0-100）
    empathy: float = 50.0     # 共情力（0-100）
    
    def to_dict(self):
        # 把性格参数转换成字典，方便存储和使用
        return {
            "warmth": self.warmth,
            "expertise": self.expertise,
            "humor": self.humor,
            "empathy": self.empathy
        }
    
    def adjust(self, feedback: Dict[str, float]):
        """根据用户反馈调整个性参数（核心：越聊越贴合你）"""
        for key, delta in feedback.items():
            if hasattr(self, key):
                current = getattr(self, key)
                # 限制参数在0-100之间（不能超过100，也不能低于0）
                new_value = max(0, min(100, current + delta))
                setattr(self, key, new_value)
        return self
 
 
# ==================== 记忆管理系统（v6.0核心，三层记忆的实现，不用改） ====================
class MemorySystem:
    """三层记忆管理系统：短期记忆+长期记忆+反思记忆，负责记、存、取记忆"""
    
    def __init__(self, db_path: str = RAGConfig.memory_db_path):
        self.db_path = db_path  # 记忆数据库路径
        self.short_term = []  # 短期记忆（用列表存储，简单高效）
        # 初始化向量生成器（用于给记忆贴“标签”）
        self.embedder = SentenceTransformerEmbeddings(
            model_name=RAGConfig.embedding_model
        )
        # 初始化数据库（创建存储记忆的“表格”）
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库，创建4个表格：记忆事件、用户画像、个性参数、目标跟踪"""
        conn = sqlite3.connect(self.db_path)  # 连接数据库（没有就自动创建）
        cursor = conn.cursor()
        
        # 1. 记忆事件表（存储每一次对话交互）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                user_input TEXT,
                agent_response TEXT,
                summary TEXT,
                topics TEXT,
                importance REAL,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. 用户画像表（存储用户的基本信息）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                background TEXT,
                preferences TEXT,
                interaction_count INTEGER,
                last_active TEXT,
                created_at TEXT
            )
        """)
        
        # 3. 个性参数表（存储AI的性格参数）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personality (
                user_id TEXT PRIMARY KEY,
                warmth REAL,
                expertise REAL,
                humor REAL,
                empathy REAL,
                updated_at TEXT
            )
        """)
        
        # 4. 目标跟踪表（存储用户的长期目标，比如“学会随机森林调参”）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                goal TEXT,
                status TEXT,
                progress REAL,
                last_updated TEXT
            )
        """)
        
        conn.commit()  # 保存修改
        conn.close()   # 关闭连接
        print("✅ 记忆数据库初始化完成（记忆会存在memory.db文件里，删除即重置）")
    
    def add_short_term(self, event: MemoryEvent):
        """添加短期记忆，超出窗口自动丢弃（比如保留最近10轮）"""
        self.short_term.append(event)
        # 保持短期记忆的窗口大小（超过10轮，删除最旧的）
        if len(self.short_term) > RAGConfig.short_term_window:
            self.short_term.pop(0)
    
    def get_short_term(self) -> List[MemoryEvent]:
        """获取短期记忆（返回最近的对话）"""
        return self.short_term.copy()
    
    def add_long_term(self, event: MemoryEvent):
        """添加长期记忆（存入SQLite数据库，同时生成向量标签）"""
        # 生成交互摘要（如果没有，就简单截取前200字）
        if not event.summary:
            event.summary = self._generate_summary(event)
        
        # 生成记忆向量（给记忆贴“标签”，方便后续检索）
        text_to_embed = f"{event.user_input} {event.agent_response} {event.summary}"
        embedding = self.embedder.embed_query(text_to_embed.strip()) if text_to_embed.strip() else [0.0]*768
        event.embedding = embedding
        
        # 把记忆存入SQLite数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memory_events 
            (session_id, timestamp, user_input, agent_response, summary, topics, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.session_id,
            event.timestamp,
            event.user_input,
            event.agent_response,
            event.summary,
            json.dumps(event.topics, ensure_ascii=False),  # 话题标签转JSON存储
            event.importance,
            pickle.dumps(embedding)  # 向量序列化存储
        ))
        conn.commit()
        conn.close()
    
    def retrieve_long_term(self, query: str, top_k: int = 5) -> List[MemoryEvent]:
        """检索长期记忆（根据用户的问题，找到相关的历史记忆）"""
        # 生成查询向量（给用户的问题贴“标签”，方便匹配历史记忆）
        query_embedding = self.embedder.embed_query(query.strip()) if query.strip() else [0.0]*768
        
        # 从数据库获取最近100条记忆（避免检索太多，影响速度）
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT session_id, timestamp, user_input, agent_response, 
                   summary, topics, importance, embedding
            FROM memory_events
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []  # 没有历史记忆，返回空
        
        # 计算用户问题和每条记忆的相似度（余弦相似度，越像分数越高）
        scored_memories = []
        for row in rows:
            embedding = pickle.loads(row[7])  # 反序列化向量
            # 计算余弦相似度（核心：匹配相似记忆）
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            
            # 封装成MemoryEvent对象
            memory = MemoryEvent(
                session_id=row[0],
                timestamp=row[1],
                user_input=row[2],
                agent_response=row[3],
                summary=row[4],
                topics=json.loads(row[5]) if row[5] else [],
                importance=row[6]
            )
            # 相似度 × 重要性 = 最终得分（重要的记忆优先）
            scored_memories.append((similarity * memory.importance, memory))
        
        # 按得分排序，返回前top_k条记忆
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:top_k]]
    
    def _generate_summary(self, event: MemoryEvent) -> str:
        """生成交互摘要（简化记忆内容，节省存储）"""
        combined = f"用户：{event.user_input}\n助手：{event.agent_response}"
        if len(combined) > 200:
            return combined[:200] + "..."  # 超过200字，截取前200字+省略号
        return combined
    
    def update_user_profile(self, user_id: str, profile: UserProfile):
        """更新用户画像（比如你告诉AI你的名字，就更新这里）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO user_profiles 
            (user_id, name, background, preferences, interaction_count, last_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            profile.name,
            profile.background,
            json.dumps(profile.preferences, ensure_ascii=False),
            profile.interaction_count,
            profile.last_active,
            profile.created_at
        ))
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像（比如AI想知道你是谁，就从这里取）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, name, background, preferences, interaction_count, last_active, created_at
            FROM user_profiles
            WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # 把数据库中的数据，封装成UserProfile对象
            return UserProfile(
                user_id=row[0],
                name=row[1] or "",
                background=row[2] or "",
                preferences=json.loads(row[3]) if row[3] else {},
                interaction_count=row[4],
                last_active=row[5],
                created_at=row[6]
            )
        return None  # 没有用户画像，返回空
    
    def get_personality(self, user_id: str) -> Personality:
        """获取AI的个性参数（比如生成回答时，要按什么性格说）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT warmth, expertise, humor, empathy
            FROM personality
            WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # 从数据库获取性格参数
            return Personality(
                warmth=row[0],
                expertise=row[1],
                humor=row[2],
                empathy=row[3]
            )
        return Personality()  # 没有性格参数，返回默认值
    
    def update_personality(self, user_id: str, personality: Personality):
        """更新AI的个性参数（反思后调整，自动更新）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO personality
            (user_id, warmth, expertise, humor, empathy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            personality.warmth,
            personality.expertise,
            personality.humor,
            personality.empathy,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def get_conversation_context(self, query: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """获取完整的对话上下文（整合三层记忆，给AI提供“背景信息”）"""
        # 1. 短期记忆（最近3轮，避免信息太多）
        short_term = self.get_short_term()
        short_term_text = "\n".join([
            f"用户: {m.user_input}\n助手: {m.agent_response}"
            for m in short_term[-3:]
        ])
        
        # 2. 长期记忆（相关的历史记忆）
        long_term = self.retrieve_long_term(query, RAGConfig.long_term_top_k)
        long_term_text = "\n\n".join([
            f"[历史记忆] {m.summary}" for m in long_term
        ])
        
        # 3. 用户画像（你的基本信息）
        profile = self.get_user_profile(user_id)
        profile_text = ""
        if profile:
            profile_text = f"""
用户信息：
- 称呼：{profile.name}
- 背景：{profile.background}
- 偏好：{json.dumps(profile.preferences, ensure_ascii=False)}
- 已交互次数：{profile.interaction_count}
"""
        
        # 4. 个性参数（AI的性格）
        personality = self.get_personality(user_id)
        
        return {
            "short_term": short_term_text,
            "long_term": long_term_text,
            "profile": profile_text,
            "personality": personality.to_dict()
        }
 
 
# ==================== 个性化提示词生成（让AI按你的喜好说话，不用改） ====================
def build_personalized_prompt(personality: Personality, context: Dict[str, Any]) -> str:
    """根据AI的性格参数，生成个性化的提示词（决定AI的说话风格）"""
    
    # 根据性格参数，调整AI的说话风格
    style_instructions = []
    
    if personality.warmth > 70:
        style_instructions.append("语气热情友好，多使用亲切的表达，可以适当加入表情符号（比如😊、～），不用太严肃。")
    elif personality.warmth < 30:
        style_instructions.append("保持简洁直接，不过分热情，专注于回答问题，不用多余的寒暄。")
    
    if personality.expertise > 80:
        style_instructions.append("回答要专业深入，可以包含技术术语和详细解释，适合有一定专业基础的用户。")
    elif personality.expertise < 40:
        style_instructions.append("用通俗易懂的语言解释，避免专业术语，把复杂问题讲简单，适合新手。")
    
    if personality.humor > 60:
        style_instructions.append("适当加入幽默元素，用轻松的语气回答，比如加入简单的调侃、生活化的比喻。")
    
    if personality.empathy > 70:
        style_instructions.append("多关注用户的情绪，表达理解和共情，比如用户说难，就安慰鼓励，不要只讲知识点。")
    
    # 风格要求拼接（没有特殊要求，就保持自然专业）
    style_text = "\n".join(style_instructions) if style_instructions else "保持自然专业的语气，贴合用户的专业背景，回答准确、实用。"
    
    # 提示词模板（AI会根据这个模板，结合记忆和性格，生成回答）
    prompt_template = f"""
你是一个有记忆、有个性的AI知识伴侣，正在与用户进行多轮对话，核心是“记住用户、贴合用户”。
【你的个性特征】
- 热情度：{personality.warmth}/100
- 专业度：{personality.expertise}/100
- 幽默感：{personality.humor}/100
- 共情力：{personality.empathy}/100
【风格要求】
{style_text}
【用户画像】
{{profile}}
【近期对话（短期记忆）】
{{short_term}}
【相关历史记忆（长期记忆）】
{{long_term}}
【当前问题】
{{question}}
请基于以上所有信息，结合知识库内容（如果有），用符合你个性的方式回答用户的问题。
重点注意：
1. 一定要体现你对用户的了解和记忆，比如叫用户的名字，衔接历史话题，不要像第一次聊天；
2. 回答要贴合用户的专业背景，不要讲用户已经知道的内容，也不要讲太超出用户水平的内容；
3. 不用刻意堆砌知识点，实用、易懂为主。
"""
    return prompt_template
 
 
# ==================== 文档处理模块（复用v5.0，负责加载文档，不用改） ====================
def load_documents(file_path: str) -> Optional[List[Document]]:
    """加载文档（支持PDF、Word、TXT），没有文档可以忽略"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}，将只基于记忆对话")
        return None
    
    try:
        # 根据文件格式，选择对应的加载器
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            # 处理编码问题（避免中文乱码）
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="gbk")
        else:
            print("❌ 不支持的文件格式（仅支持PDF、Word、TXT）")
            return None
        
        documents = loader.load()
        print(f"✅ 成功加载文档：{file_path}，共 {len(documents)} 页/段")
        return documents
    except Exception as e:
        print(f"❌ 加载文档失败：{str(e)}")
        return None
 
 
def split_documents(documents: List[Document]) -> List[Document]:
    """文本分块（把文档分成小块，方便检索）"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.chunk_size,
        chunk_overlap=RAGConfig.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 文本拆分完成，共 {len(splits)} 个片段")
    return splits
 
 
def create_knowledge_base(documents: List[Document]):
    """创建知识库向量库（存储文档内容，用于检索）"""
    texts = split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name=RAGConfig.embedding_model)
    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=RAGConfig.persist_directory + "_knowledge"
    )
    vectordb.persist()
    print(f"✅ 知识库创建完成，共 {len(texts)} 个片段")
    return vectordb
 
 
def retrieve_knowledge(vectordb, query: str, k: int = 5) -> List[Document]:
    """从知识库检索相关信息（就像在图书馆找书，找到最相关的几本）"""
    retriever = vectordb.as_retriever(
        search_type="mmr",  # MMR检索：既要相关，又要多样，避免找到的都是重复内容
        search_kwargs={"k": k, "fetch_k": k * 3}
    )
    docs = retriever.invoke(query)
    return docs
 
 
def rerank_documents(query: str, documents: List[Document], top_k: int) -> List[Document]:
    """重排序（像面试官给候选人打分，筛出最匹配的几个）"""
    if not documents:
        return []
    
    rerank_model = CrossEncoder(RAGConfig.rerank_model)
    pairs = [[query, doc.page_content] for doc in documents]
    scores = rerank_model.predict(pairs)
    
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]
 
 
# ==================== 主对话引擎（v6.0的大脑，整合所有模块） ====================
class PersonalizedRAGEngine:
    """v6.0 主引擎：整合记忆 + 知识库 + 个性化，就像一个懂你的老朋友"""
    
    def __init__(self, knowledge_vectordb=None):
        self.memory = MemorySystem()  # 记忆系统（负责记东西）
        self.knowledge_db = knowledge_vectordb  # 知识库（文档内容）
        self.llm = Ollama(
            model=RAGConfig.summary_model,
            temperature=RAGConfig.temperature,
            client_kwargs={"timeout": 1200}  # 超时时间延长，避免大模型思考太久
        )
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")  # 会话ID，区分每次聊天
        self.user_id = "default_user"  # 用户ID，简单起见用默认值（可扩展多用户）
        self.interaction_count = 0  # 聊了多少轮
        
        # 加载用户画像（你是谁）
        self.profile = self.memory.get_user_profile(self.user_id)
        if not self.profile:
            self.profile = UserProfile(user_id=self.user_id)
            self.memory.update_user_profile(self.user_id, self.profile)
        
        # 加载AI的个性参数（它是什么性格）
        self.personality = self.memory.get_personality(self.user_id)
        
        print(f"🔑 会话ID：{self.session_id}")
        print(f"👤 用户：{self.profile.name or '新朋友（快告诉我你的名字吧）'}")
        print(f"🎭 当前个性：{self.personality.to_dict()}")
    
    def process_message(self, user_input: str) -> str:
        """处理单条用户消息（核心：你说一句话，AI怎么回应）"""
        self.interaction_count += 1
        
        # === 第1步：检索记忆（看看你是谁、聊过啥） ===
        memory_context = self.memory.get_conversation_context(
            user_input, self.user_id, self.session_id
        )
        
        # === 第2步：检索知识库（如果有文档，找相关知识） ===
        knowledge_docs = []
        if self.knowledge_db:
            knowledge_docs = retrieve_knowledge(self.knowledge_db, user_input, k=5)
            knowledge_docs = rerank_documents(user_input, knowledge_docs, 3)  # 重排序，筛出最相关的3条
        
        knowledge_text = "\n\n".join([
            f"[知识] {doc.page_content}" for doc in knowledge_docs
        ]) if knowledge_docs else "（当前没有相关知识库，只基于记忆回答）"
        
        # === 第3步：根据个性生成提示词（让AI按你的喜好说话） ===
        prompt_template = build_personalized_prompt(
            self.personality, memory_context
        )
        
        # 填充提示词模板
        formatted_prompt = prompt_template.format(
            profile=memory_context["profile"],
            short_term=memory_context["short_term"],
            long_term=memory_context["long_term"],
            question=user_input
        )
        
        # 加上知识库内容，组成完整输入
        full_prompt = f"{formatted_prompt}\n\n【知识库参考】\n{knowledge_text}\n\n回答："
        
        # === 第4步：调用大模型生成回答 ===
        print("🤔 AI正在思考，结合记忆和知识...")
        response = self.llm.invoke(full_prompt)
        
        # === 第5步：把这次对话记下来（存入记忆） ===
        event = MemoryEvent(
            session_id=self.session_id,
            user_input=user_input,
            agent_response=response,
            importance=1.0  # 默认重要性
        )
        
        # 添加到短期记忆（最近几轮）
        self.memory.add_short_term(event)
        
        # 每隔几轮，存一次长期记忆（避免存太频繁，占空间）
        if self.interaction_count % 2 == 0:  # 每2轮存一次
            self.memory.add_long_term(event)
            print("💾 已将这次对话存入长期记忆")
        
        # === 第6步：更新用户画像（比如你说了名字，就记下来） ===
        self.profile.interaction_count += 1
        self.profile.last_active = datetime.now().isoformat()
        
        # 简单提取用户信息（如果用户说了“我叫xx”或“我是xx”）
        if "我叫" in user_input or "我是" in user_input:
            # 用简单规则提取名字（高级版可以用LLM提取，这里简化）
            import re
            name_match = re.search(r"叫([^，。\s]+)", user_input)
            if name_match and not self.profile.name:
                self.profile.name = name_match.group(1)
                print(f"📝 记住你的名字了：{self.profile.name}")
            
            bg_match = re.search(r"是([^，。\s]+)", user_input)
            if bg_match and not self.profile.background:
                self.profile.background = bg_match.group(1)
                print(f"📝 记住你的背景了：{self.profile.background}")
        
        self.memory.update_user_profile(self.user_id, self.profile)
        
        # === 第7步：定期触发反思（让AI自我进化） ===
        if self.interaction_count % RAGConfig.reflection_interval == 0:
            self._trigger_reflection()
        
        return response
    
    def _trigger_reflection(self):
        """触发自我反思：AI回顾最近的对话，调整自己的性格"""
        print("\n🧠 AI正在进行自我反思，回顾和你的对话...")
        
        # 获取最近的10条记忆用于反思
        recent_memories = self.memory.retrieve_long_term("", 10)
        memories_text = "\n".join([
            f"用户：{m.user_input[:50]}..." for m in recent_memories
        ])
        
        # 反思提示词（让AI分析自己的表现）
        reflection_prompt = f"""
请基于你和用户最近的对话记录，进行自我反思，调整你的个性参数，让自己更贴合用户。
最近对话摘要（共{len(recent_memories)}条）：
{memories_text}
你当前的个性参数：
{self.personality.to_dict()}
请深入分析以下问题：
1. 用户更喜欢热情的回答还是简洁的回答？（比如用户是否用表情、语气词）
2. 用户需要多专业的回答？（用户是追问细节，还是说听不懂）
3. 用户对幽默有反应吗？（用户有没有笑、用轻松语气）
4. 用户表达情绪时，需要你共情吗？（用户有没有倾诉、抱怨）
基于以上分析，输出个性调整量（每个维度-10到+10之间的整数）：
- warmth（热情度）：用户需要更热情就+，需要更冷静就-
- expertise（专业度）：用户需要更专业就+，需要更通俗就-
- humor（幽默感）：用户喜欢幽默就+，不喜欢就-
- empathy（共情力）：用户需要共情就+，不需要就-
输出格式（只输出JSON，不要其他内容）：
{{"warmth": 调整量, "expertise": 调整量, "humor": 调整量, "empathy": 调整量}}
"""
        
        try:
            # 调用大模型生成调整建议
            adjustment_str = self.llm.invoke(reflection_prompt)
            print(f"🧠 反思结果：{adjustment_str}")
            
            # 解析JSON（需要处理可能的格式问题）
            import re
            import json
            adjustment_match = re.search(r'\{.*\}', adjustment_str, re.DOTALL)
            if adjustment_match:
                adjustment = json.loads(adjustment_match.group())
                # 调整个性参数
                self.personality.adjust(adjustment)
                self.memory.update_personality(self.user_id, self.personality)
                print(f"✅ 个性已更新：{self.personality.to_dict()}")
            else:
                print("⚠️ 反思结果格式不对，保持当前个性")
        except Exception as e:
            print(f"⚠️ 反思过程出错：{e}，保持当前个性")
    
    def set_user_info(self, name: str = "", background: str = "", **preferences):
        """手动设置用户信息（如果不自动提取，可以手动设置）"""
        self.profile.name = name or self.profile.name
        self.profile.background = background or self.profile.background
        self.profile.preferences.update(preferences)
        self.memory.update_user_profile(self.user_id, self.profile)
        print(f"✅ 用户信息已更新：{self.profile}")
 
 
# ==================== 主程序（一键运行入口） ====================
def main():
    """主函数：程序从这里开始运行"""
    print("="*80)
    print("🤖 RAG v6.0：永久记忆 + 个性化养成（你的专属AI老朋友）")
    print("="*80)
    print("核心特性：")
    print("  • 三层记忆系统：记住你是谁、聊过啥、喜欢啥")
    print("  • 跨会话永久记忆：退出再进，还记得你")
    print("  • 个性化动态养成：聊得越多，越懂你")
    print("  • 知识库全局理解：结合文档，回答更准")
    print("="*80)
    print()
    
    # === 第1步：加载文档知识库（可选，没有也能运行） ===
    # 修改这里：把路径换成你的文档路径（没有文档就留空或注释掉）
    #KNOWLEDGE_PATH = "./docs/机器学习实战.pdf"  # ← 新手必改：换成你的文档路径
    KNOWLEDGE_PATH = "./docx/rag.txt"
    
    knowledge_db = None
    if os.path.exists(KNOWLEDGE_PATH):
        print("\n📚 正在加载知识库文档...")
        docs = load_documents(KNOWLEDGE_PATH)
        if docs:
            knowledge_db = create_knowledge_base(docs)
            print("✅ 知识库加载完成，可以问文档里的内容了")
    else:
        print("\n⚠️ 未找到知识库文档，将只基于记忆对话（也可以正常聊天）")
        print("   如果想用文档，请把PDF/Word/TXT放在 ./docs/ 文件夹下")
    
    # === 第2步：初始化对话引擎（创建你的AI老朋友） ===
    print("\n🎮 正在启动AI老朋友...")
    engine = PersonalizedRAGEngine(knowledge_db)
    
    # === 第3步：开始对话 ===
    print("\n" + "="*80)
    print("💬 对话开始！输入你的问题，和AI聊天吧～")
    print("提示：")
    print("  • 告诉AI你的名字和背景（比如“我叫张三，是数据分析师”），它会记住你")
    print("  • 可以问文档里的问题（比如“随机森林怎么调参”）")
    print("  • 也可以闲聊（它会根据你的反馈调整个性）")
    print("  • 输入 q/quit/exit 退出程序")
    print("="*80)
    
    while True:
        try:
            user_input = input("\n👤 你：").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\n👋 再见！期待下次和你聊天～")
                print("（你的记忆已经保存，下次启动程序我还记得你）")
                break
            
            if not user_input:
                continue
            
            # 处理消息（核心对话逻辑）
            print("🤖 AI：", end="", flush=True)
            response = engine.process_message(user_input)
            print(response)
            
            # 每隔几轮，显示AI的当前个性（让用户看到变化）
            if engine.interaction_count % 5 == 0:
                p = engine.personality
                print(f"\n📊 当前AI性格：热情度{p.warmth:.0f} 专业度{p.expertise:.0f} 幽默感{p.humor:.0f} 共情力{p.empathy:.0f}")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断对话，再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了：{e}")
            print("💡 建议：检查Ollama是否在运行？模型是否下载好了？")
            continue
 
 
# ==================== 程序入口 ====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序启动失败：{e}")
        print("💡 快速排查：")
        print("   1. 是否安装了所有依赖？运行：pip install -r requirements.txt")
        print("   2. Ollama是否启动？运行：ollama serve")
        print("   3. 模型是否下载？运行：ollama pull qwen:7b-chat")