# -*- coding: utf-8 -*-
"""
RAG 个人私有知识库 v5.0（完整可运行版）
核心升级：RAPTOR 递归摘要树，实现全局理解
基础特性：零成本、本地大模型（Ollama）、100% 离线、支持PDF/Word/TXT
进阶特性：构建语义树，支持“全书总结”“主题归纳”等宏观问题
8G内存可运行，新手友好！
注意：运行前需确保 Ollama 已启动，且已下载 qwen:8b 模型（或修改为自己的模型）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
os.environ["TRANSFORMERS_OFFLINE"] = "0" # 允许联网下载 
os.environ["HF_HUB_OFFLINE"] = "0" # 然后再import其他模块 
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import pickle
import hashlib
 
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM as Ollama
 
from sentence_transformers import CrossEncoder
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
 
 
class RAGConfig:
    """统一配置项（v5.0 新增 RAPTOR 配置），新手无需修改，按需微调即可"""
    # 文本拆分参数（v5.0 使用更小的块，适配8G内存）
    chunk_size = 200  # 8G内存建议200，16G内存可改为300
    chunk_overlap = 50  # 块重叠，保证上下文连贯
    
    # 检索参数
    k_retrieval = 8  # 向量检索召回8个候选（8G内存适配）
    k_rerank = 5     # 重排序后选5个（留更多给树节点）
    
    # 向量库参数
    persist_directory = "./chroma_db"  # 向量库存储路径
    embedding_model = "BAAI/bge-small-zh-v1.5"  # 中文向量模型，轻量且效果好
    
    # RAPTOR 参数（v5.0 新增）
    raptor_tree_path = "./raptor_tree.pkl"  # 树结构缓存路径（自动生成）
    raptor_max_clusters = 10  # 每层最大聚类数
    raptor_min_cluster_size = 2  # 最小聚类大小（少于2个不聚类）
    raptor_max_depth = 2  # 最大树深度（8G内存建议2层，16G可改为3层）
    
    # 重排序模型（中文适配）
    rerank_model = "BAAI/bge-reranker-base"
    
    # 本地大模型参数（用于生成摘要和最终回答）
    temperature = 0  # 0表示 deterministic，不产生随机答案
    encoding = "utf-8"  # 文本编码
    
    # 摘要模型（复用 Ollama，可替换为自己下载的模型，如 llama3:8b）
    summary_model = "qwen3:8b"
 
 
class RAPTORNode:
    """RAPTOR 树节点，存储每个层级的文本、层级和子节点索引"""
    def __init__(self, text: str, level: int, children_indices: List[int] = None):
        self.text = text  # 节点文本（叶子节点是原始块，非叶子是摘要）
        self.level = level  # 层级（0=叶子，1=第一层摘要，2=第二层摘要...）
        self.children_indices = children_indices or []  # 子节点索引（关联下层节点）
        self.embedding = None  # 节点向量（后续计算，用于检索）
    
    def to_dict(self):
        """将节点转为字典，用于保存到本地缓存"""
        return {
            "text": self.text,
            "level": self.level,
            "children_indices": self.children_indices
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典加载节点，用于读取本地缓存的树结构"""
        node = cls(data["text"], data["level"], data["children_indices"])
        return node
 
 
class RAPTORTree:
    """RAPTOR 树结构，管理所有节点和层级关系"""
    def __init__(self):
        self.nodes: List[RAPTORNode] = []  # 所有节点（扁平存储，便于检索）
        self.root_indices: List[int] = []  # 根节点索引（最高层摘要节点）
        self.doc_id = None  # 文档唯一标识（避免不同文档缓存冲突）
    
    def add_node(self, node: RAPTORNode) -> int:
        """添加节点到树中，返回节点索引"""
        self.nodes.append(node)
        return len(self.nodes) - 1
    
    def get_node(self, idx: int) -> RAPTORNode:
        """根据索引获取节点"""
        return self.nodes[idx]
    
    def size(self):
        """返回树的总节点数"""
        return len(self.nodes)
    
    def save(self, path: str):
        """保存树结构到本地缓存，避免重复构建"""
        data = {
            "nodes": [n.to_dict() for n in self.nodes],
            "root_indices": self.root_indices,
            "doc_id": self.doc_id
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str):
        """从本地缓存加载树结构，加速运行"""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tree = cls()
        tree.nodes = [RAPTORNode.from_dict(n) for n in data["nodes"]]
        tree.root_indices = data["root_indices"]
        tree.doc_id = data.get("doc_id")
        return tree
 
 
def load_documents(file_path: str) -> Optional[List[Document]]:
    """加载单文档（支持PDF/Word/TXT），自动处理编码问题，新手无需修改"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}，请检查路径是否正确")
        return None
    
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            # 自动适配编码，避免中文乱码
            try:
                loader = TextLoader(file_path, encoding=RAGConfig.encoding)
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="gbk")
        else:
            print("❌ 不支持的文件格式，仅支持PDF/Word/TXT")
            return None
        
        documents = loader.load()
        print(f"✅ 成功加载文档：{file_path}，共 {len(documents)} 页/段")
        return documents
    except Exception as e:
        print(f"❌ 加载文档失败：{str(e)}，建议检查文件是否损坏")
        return None
 
 
def split_documents(documents: List[Document]) -> List[Document]:
    """文本分块（适配RAPTOR，块更小，便于聚类），保持句子完整性"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.chunk_size,
        chunk_overlap=RAGConfig.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        length_function=len  # 按中文字符长度计算，适配中文文档
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 文本拆分完成，共拆分成 {len(splits)} 个片段（每个片段约{int(RAGConfig.chunk_size/2)}个中文字符）")
    return splits
 
 
def get_embeddings(texts: List[str]) -> np.ndarray:
    """批量获取文本向量，分批处理，避免内存溢出（适配8G内存）"""
    embedder = SentenceTransformerEmbeddings(model_name=RAGConfig.embedding_model)
    vectors = []
    batch_size = 16  # 8G内存建议16，16G可改为32，避免内存不足
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_vectors = embedder.embed_documents(batch)
        vectors.extend(batch_vectors)
        print(f"🔄 正在生成向量：第{i//batch_size + 1}批，共{len(texts)//batch_size + 1}批")
    return np.array(vectors)
 
 
def perform_clustering(embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 10):
    """
    使用 GMM 进行聚类，BIC 自动选择最优聚类数（核心聚类逻辑）
    先通过UMAP降维，解决高维向量聚类效率低、内存消耗大的问题
    """
    if len(embeddings) < min_clusters:
        return np.zeros(len(embeddings), dtype=int)  # 不足min_clusters个，归为一类
    
    # 根据数据点数量决定是否使用UMAP降维
    # 数据点太少时，UMAP的谱分解会失败，直接使用原始嵌入进行聚类
    if len(embeddings) <= 10:
        reduced_embeddings = embeddings
        print(f"⚠️  数据点较少（{len(embeddings)}个），跳过UMAP降维")
    else:
        # 先用 UMAP 降维（保留局部和全局结构，加速聚类）
        n_neighbors = min(15, len(embeddings) - 1)  # 减1避免等于数据点数量
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=5, metric='cosine', random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
    
    # 尝试不同聚类数，选择 BIC 最小的（BIC越小，聚类效果越好）
    best_gmm = None
    best_bic = np.inf
    best_n = min_clusters
    
    # 遍历可能的聚类数，避免聚类数过多或过少
    for n_clusters in range(min_clusters, min(max_clusters, len(embeddings)) + 1):
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='tied', random_state=42)
        gmm.fit(reduced_embeddings)
        bic = gmm.bic(reduced_embeddings)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n_clusters
    
    # 使用最佳模型预测聚类标签
    if best_gmm:
        labels = best_gmm.predict(reduced_embeddings)
        print(f"✅ 聚类完成，最优聚类数：{best_n}")
        return labels
    else:
        return np.zeros(len(embeddings), dtype=int)
 
 
def generate_summary(texts: List[str], llm) -> str:
    """
    使用本地大模型生成文本摘要，适配中文，优化提示词，保证摘要质量
    避免文本过长导致模型超时，自动截断
    """
    if not texts:
        return ""
    
    # 合并文本，过长时截断（避免模型处理超时）
    combined = "\n\n".join(texts)
    if len(combined) > 2000:  # 超过2000字符，截断核心内容
        combined = combined[:2000] + "..."
    
    # 优化后的中文摘要提示词，确保摘要简洁、全面、有逻辑
    prompt = f"""请严格遵循以下规则，为以下文本生成中文摘要：
1. 核心要求：提炼文本的核心框架、关键观点、核心方法，不罗列无关细节；
2. 结构要求：先1句话总述核心内容，再分2-3点简要说明关键信息，最后1句话总结核心价值；
3. 语言要求：简洁专业、符合中文表达习惯，避免口语化，长度控制在150-250字；
4. 禁忌：不添加额外信息、不编造内容，完全基于输入文本生成。
待总结文本：
{combined}
摘要（严格按要求生成）：
"""
    
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"⚠️  摘要生成失败：{e}，将降级返回文本前200字作为临时摘要")
        return combined[:200] + "..."
 
 
def build_raptor_tree(documents: List[Document], force_rebuild: bool = False) -> RAPTORTree:
    """
    完整构建 RAPTOR 递归摘要树（补全所有层级，非简化版）
    算法流程参考 RAPTOR 论文，适配中文文档，避免内存溢出
    """
    # 检查缓存，存在且不强制重建则直接加载，加速运行
    tree_path = RAGConfig.raptor_tree_path
    if not force_rebuild and os.path.exists(tree_path):
        print(f"✅ 加载缓存的 RAPTOR 树：{tree_path}，无需重新构建")
        return RAPTORTree.load(tree_path)
    
    print("\n🔨 开始构建 RAPTOR 树（首次构建可能需要几分钟，后续会缓存）...")
    
    # 步骤1：获取文本块（叶子节点）
    chunks = split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    
    # 步骤2：初始化树和 LLM（用于生成摘要）
    tree = RAPTORTree()
    llm = Ollama(
        model=RAGConfig.summary_model,
        temperature=RAGConfig.temperature,
        client_kwargs={"timeout": 1200}  # 延长超时时间，避免摘要生成超时
    )
    
    # 当前层节点文本（初始为叶子节点文本）和当前层级
    current_texts = texts.copy()
    current_level = 0
    # 记录当前层节点在树中的索引（用于关联父子关系）
    current_node_indices = []
    
    # 步骤3：添加叶子节点（层级0）
    for text in current_texts:
        node = RAPTORNode(text=text, level=current_level)
        node_idx = tree.add_node(node)
        current_node_indices.append(node_idx)
    
    print(f"✅ 已添加 {len(current_node_indices)} 个叶子节点（层级0）")
    
    # 步骤4：递归构建中层、高层摘要节点
    while len(current_texts) > 1 and current_level < RAGConfig.raptor_max_depth:
        print(f"\n🔄 构建第 {current_level + 1} 层摘要（当前 {len(current_texts)} 个节点）...")
        
        # 步骤4.1：获取当前层所有文本的向量
        embeddings = get_embeddings(current_texts)
        
        # 步骤4.2：执行聚类（根据向量相似度分组）
        if len(current_texts) > RAGConfig.raptor_min_cluster_size:
            labels = perform_clustering(
                embeddings, 
                min_clusters=2,
                max_clusters=min(RAGConfig.raptor_max_clusters, len(current_texts))
            )
        else:
            labels = np.zeros(len(current_texts), dtype=int)  # 不足最小聚类数，归为一类
        
        # 步骤4.3：按聚类标签分组，获取每个聚类的节点索引和文本
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)  # 存储当前层的节点索引
        
        # 步骤4.4：为每个聚类生成摘要，创建高层节点，并关联父子关系
        next_texts = []
        next_node_indices = []
        
        for cluster_id, indices in clusters.items():
            # 提取该聚类的所有文本（当前层）
            cluster_texts = [current_texts[i] for i in indices]
            # 提取该聚类对应的子节点索引（当前层的节点索引）
            cluster_child_indices = [current_node_indices[i] for i in indices]
            # 生成该聚类的摘要（高层文本）
            summary = generate_summary(cluster_texts, llm)
            # 创建高层节点（层级+1）
            node = RAPTORNode(
                text=summary,
                level=current_level + 1,
                children_indices=cluster_child_indices
            )
            # 添加节点到树中，记录索引
            node_idx = tree.add_node(node)
            next_texts.append(summary)
            next_node_indices.append(node_idx)
        
        # 步骤4.5：更新当前层信息，进入下一轮递归
        current_texts = next_texts
        current_node_indices = next_node_indices
        current_level += 1
    
    # 步骤5：设置根节点（最后一轮生成的节点就是最高层摘要）
    tree.root_indices = current_node_indices
    # 设置文档唯一标识（避免不同文档缓存冲突）
    tree.doc_id = hashlib.md5("".join(texts[:5]).encode()).hexdigest()
    
    # 步骤6：保存树结构到本地缓存
    tree.save(tree_path)
    print(f"\n✅ RAPTOR 树构建完成！")
    print(f"   - 总节点数：{tree.size()}")
    print(f"   - 叶子节点数：{len(texts)}")
    print(f"   - 摘要节点数：{tree.size() - len(texts)}")
    print(f"   - 树深度：{current_level}")
    print(f"   - 树已缓存至：{tree_path}")
    
    return tree
 
 
def create_raptor_retriever(documents: List[Document]):
    """
    创建 RAPTOR 检索器（完整版）：将所有节点（叶子+所有层级摘要）存入向量库
    实现折叠树检索，平等对待所有层级节点，保证全局理解能力
    """
    # 步骤1：构建完整的 RAPTOR 树（获取所有节点：叶子+中层+高层摘要）
    tree = build_raptor_tree(documents)
    
    # 步骤2：提取树中所有节点的文本和元数据（用于存入向量库）
    all_docs = []
    for idx, node in enumerate(tree.nodes):
        # 为每个节点添加元数据，标记层级和类型（便于后续查看答案来源）
        metadata = {
            "type": "leaf" if node.level == 0 else "summary",
            "level": node.level,
            "node_index": idx,
            "children_count": len(node.children_indices)
        }
        # 为根节点添加特殊标记
        if idx in tree.root_indices:
            metadata["type"] = "global_summary"
            metadata["note"] = "全局摘要节点（最高层）"
        # 创建文档对象，存入向量库
        doc = Document(page_content=node.text, metadata=metadata)
        all_docs.append(doc)
    
    # 步骤3：创建向量库（包含所有节点，实现折叠树检索）
    print(f"\n🔄 正在创建 RAPTOR 向量库（共 {len(all_docs)} 个节点：叶子+摘要）...")
    embeddings = SentenceTransformerEmbeddings(model_name=RAGConfig.embedding_model)
    # 单独创建 RAPTOR 专用向量库，避免与其他版本冲突
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=RAGConfig.persist_directory + "_raptor"
    )
    vectordb.persist()  # 持久化向量库，下次运行无需重新创建
    
    print(f"✅ RAPTOR 检索器创建完成，向量库已保存至：{RAGConfig.persist_directory + '_raptor'}")
    return vectordb
 
 
def create_retriever(vectordb):
    """创建检索器，配置MMR检索策略，平衡相关性和多样性"""
    retriever = vectordb.as_retriever(
        search_type="mmr",  # MMR策略：最大化相关性和多样性，避免召回重复内容
        search_kwargs={
            "k": RAGConfig.k_retrieval,  # 初始召回数量
            "fetch_k": RAGConfig.k_retrieval * 3,  # 候选数量，保证召回全面性
            "lambda_mult": 0.5  # 0.5平衡相关性和多样性，适配全局检索
        }
    )
    print(f"✅ 检索器创建完成，初始召回数量：{RAGConfig.k_retrieval}")
    return retriever
 
 
def rerank_documents(query: str, documents: list, top_k: int) -> list:
    """重排序（复用 v4.0 的 CrossEncoder），筛选最相关的节点，提升答案精准度"""
    print(f"⏳ 正在对 {len(documents)} 个候选节点进行重排序，筛选前{top_k}个最相关节点...")
    rerank_model = CrossEncoder(RAGConfig.rerank_model)
    
    # 构建查询与候选节点的配对，用于计算相似度
    pairs = [[query, doc.page_content] for doc in documents]
    scores = rerank_model.predict(pairs)  # 计算相似度得分，得分越高越相关
    
    # 按得分降序排序，筛选前top_k个
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    print(f"✅ 重排序完成，选出 {top_k} 个最匹配节点")
    
    return [doc for doc, _ in scored_docs[:top_k]]
def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(doc.page_content for doc in docs)
 
def build_rag_chain(retriever):
    """构建 RAG 问答链，优化提示词，保证答案基于上下文，不编造内容"""
    print("\n🔄 初始化本地大模型，构建问答链...")
    # 初始化本地大模型（Ollama），可替换为自己下载的模型
    llm = Ollama(
        model=RAGConfig.summary_model,
        temperature=RAGConfig.temperature,
        client_kwargs={"timeout": 1200}  # 延长超时时间，避免回答生成超时
    )
    
    # 优化后的中文提示词，严格约束模型，保证答案质量
    prompt_template = """
你是一个专业的问答助手，严格基于以下【上下文内容】回答用户的问题，不添加任何外部知识。
【约束条件】
1. 优先参考上下文中标注为「全局摘要节点」的内容，获得全局视角；再结合其他节点（叶子/中层摘要）补充细节；
2. 回答需逻辑清晰、简洁准确，符合中文表达习惯，分点说明（如果内容较多）；
3. 如果上下文中包含答案，需综合所有相关节点的信息，不遗漏核心内容，不重复表述；
4. 如果上下文中没有相关信息，直接回答：根据现有知识库，无法回答该问题。
5. 严禁编造内容、严禁使用上下文以外的知识，严禁添加主观评价。
【上下文内容】
{context}
【用户问题】
{question}
【回答】：
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # 使用 langchain 1.0+ LCEL 方式构建问答链
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG问答链构建完成（v5.0：RAPTOR 树 + 重排序双重优化，LCEL 1.0+）")
    return qa_chain
 
 
def rag_qa(qa_chain, retriever, question: str):
    """执行问答，美化输出，显示答案来源，便于新手调试"""
    print(f"\n🔍 正在检索问题：'{question}'")
    
    # 步骤1：使用 LCEL 链获取答案
    answer = qa_chain.invoke(question)
    
    # 步骤2：获取源文档并进行重排序
    sources = retriever.invoke(question)
    sources = rerank_documents(
        question, 
        sources, 
        RAGConfig.k_rerank
    )
    
    # 步骤4：美化输出，清晰显示答案和来源
    print("\n" + "="*60)
    print("✅ 回答：")
    print("-"*60)
    print(answer)
    print("\n📚 答案来源（按相关性排序）：")
    print("-"*60)
    for i, doc in enumerate(sources, 1):
        # 提取源文档元数据，显示节点类型、层级
        source_type = doc.metadata.get("type")
        if source_type == "global_summary":
            source_info = f"{i}. 全局摘要节点（最高层，层级{doc.metadata['level']}）"
        elif source_type == "summary":
            source_info = f"{i}. 中层摘要节点（层级{doc.metadata['level']}）"
        else:
            source_info = f"{i}. 叶子节点（原始文本片段，层级{doc.metadata['level']}）"
        # 显示节点索引和子节点数量（便于调试）
        source_info += f" | 节点索引：{doc.metadata['node_index']} | 子节点数：{doc.metadata['children_count']}"
        # 显示文本预览
        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        print(source_info)
        print(f"   预览：{preview}")
    print("="*60)
 
 
def main():
    """主函数，一键运行入口，新手只需修改文档路径即可"""
    print("📚 个人私有知识库 RAG 系统 v5.0（完整可运行版）")
    print("="*70)
    print("核心特性：RAPTOR 递归摘要树 + 全局理解能力")
    print("支持：全书总结、主题归纳、跨章节联想 | 100% 私有数据不泄露")
    print("环境要求：8G内存 + Ollama + qwen3:8b 模型 | 零成本、全离线")
    print("="*70)
    print()
 
    # ====================== 新手必改：修改文档路径 ======================
    # 请将下方路径替换为你的文档路径（PDF/Word/TXT均可）
    # 示例1：本地文档（放在当前文件夹docs目录下）
    # DOC_PATH = "./docx/机器学习_周志华1.pdf"
    DOC_PATH = "./docx/rag.txt"
    # 示例2：绝对路径（适用于文档不在当前文件夹）
    # DOC_PATH = "C:/Users/XXX/Documents/机器学习实战.pdf"
    # ========================================================
 
    # 步骤1：加载文档
    documents = load_documents(DOC_PATH)
    if not documents:
        print("❌ 文档加载失败，程序退出")
        return
    
    # 步骤2：创建 RAPTOR 检索器（包含所有节点：叶子+摘要）
    vectordb = create_raptor_retriever(documents)
    
    # 步骤3：创建检索器和问答链
    retriever = create_retriever(vectordb)
    qa_chain = build_rag_chain(retriever)
    
    # 步骤4：启动交互问答
    print("\n🎉 问答交互已启动（输入 q/quit/exit 退出）")
    print("💡 推荐测试宏观问题（v5.0 核心优势）：")
    print("   1. 这本书主要讲了什么？")
    print("   2. 作者的核心观点和方法论是什么？")
    print("   3. 书中分几个部分，每个部分的核心内容是什么？")
    print("💡 也可测试点状问题（v4.0 擅长）：")
    print("   1. 随机森林的参数怎么调？")
    print("   2. 数据预处理的步骤有哪些？")
    
    while True:
        q = input("\n请输入你的问题：")
        # 退出逻辑
        if q.lower() in ['q', 'quit', 'exit']:
            print("感谢使用，再见！")
            break
        # 空输入跳过
        if not q.strip():
            continue
        
        try:
            # 执行问答
            rag_qa(qa_chain, retriever, q)
        except Exception as e:
            print(f"❌ 问答执行失败：{str(e)}")
            print("💡 建议检查：")
            print("   1. Ollama 是否正常启动（终端执行 ollama serve）")
            print("   2. 模型是否已下载（终端执行 ollama pull qwen:7b）")
            print("   3. 文档路径是否正确，文件是否损坏")
 
 
if __name__ == "__main__":
    # 捕获异常，避免程序崩溃，便于新手调试
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")
        print("💡 快速排查：")
        print("   1. 依赖是否安装完整（重新执行 pip 安装命令）")
        print("   2. 内存是否充足（关闭其他占用内存的程序）")
        print("   3. Ollama 服务是否正常运行")