# -*- coding: utf-8 -*-
"""
个人私有知识库 RAG 系统
支持：PDF / Word / TXT、向量库缓存、MMR 检索、中文优化、异常处理、编码兼容
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
os.environ["TRANSFORMERS_OFFLINE"] = "0" # 允许联网下载 
os.environ["HF_HUB_OFFLINE"] = "0" # 然后再import其他模块 
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings 
from langchain_ollama import OllamaLLM as Ollama
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder  # 标准导入，与向量模型风格一致


class RAGConfig:
    """统一配置项"""
    chunk_size = 500
    chunk_overlap = 50
    k_retrieval = 6  # 向量检索召回6个候选片段（原v3.0为3）
    k_rerank = 3     # 重排序后最终选取3个最匹配片段（和原v3.0一致）
    retrieval_score_threshold = 0.5
    persist_directory = "./chroma_db"
    #embedding_model = "all-MiniLM-L6-v2"
    embedding_model = "BAAI/bge-small-zh-v1.5"
    rerank_model = "BAAI/bge-reranker-base"     # 轻量中文重排序模型（v4.0新增）
    temperature = 0.1
    encoding = "utf-8"
 
 
def load_documents(file_path: str) -> Optional[List[Document]]:
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return None
 
    try:
        if file_path.endswith('.pdf'):
            print("检测到 PDF 文件")
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            print("检测到 Word 文件")
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.txt'):
            print("检测到 TXT 文件")
            try:
                loader = TextLoader(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                loader = TextLoader(file_path, encoding="gbk")
        else:
            print("不支持的文件格式")
            print("支持格式：.pdf, .docx, .txt")
            return None
 
        documents = loader.load()
        print(f"成功加载文档：{file_path}")
        total_len = sum(len(d.page_content) for d in documents)
        print(f"文档信息：共 {len(documents)} 页/段，总字符数：{total_len}")
        return documents
 
    except Exception as e:
        print(f"加载文档失败：{str(e)}")
        return None
 
 
def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.chunk_size,
        chunk_overlap=RAGConfig.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    print("文本拆分完成")
    print(f"拆分信息：共拆分成 {len(splits)} 个片段")
    return splits
 
 
def get_or_create_vector_db(splits: List[Document]) -> Tuple[Chroma, SentenceTransformerEmbeddings]:
    print(f"\n初始化向量化模型：{RAGConfig.embedding_model}")
 
    embeddings = SentenceTransformerEmbeddings(model_name=RAGConfig.embedding_model)
 
    if os.path.exists(RAGConfig.persist_directory) and os.listdir(RAGConfig.persist_directory):
        print(f"检测到已存在的向量库，直接加载：{RAGConfig.persist_directory}")
        vectordb = Chroma(
            persist_directory=RAGConfig.persist_directory,
            embedding_function=embeddings
        )
        print(f"向量库信息：包含 {vectordb._collection.count()} 个向量")
    else:
        print(f"未检测到向量库，新建并保存：{RAGConfig.persist_directory}")
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=RAGConfig.persist_directory
        )
        vectordb.persist()
        print(f"向量库信息：包含 {vectordb._collection.count()} 个向量")
 
    return vectordb, embeddings
 
 
def create_retriever(vectordb: Chroma):
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RAGConfig.k_retrieval,
            "fetch_k": RAGConfig.k_retrieval * 3,
            "lambda_mult": 0.7
        }
    )
    print(f"检索器创建完成，检索策略：MMR，召回数量：{RAGConfig.k_retrieval}")
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
    print("\n初始化大模型...")
    # llm = ChatOpenAI(
    #     model_name=RAGConfig.llm_model,
    #     temperature=RAGConfig.temperature
    # )
    
    # 第 2 处修改：替换成本地 Ollama 模型（qwen:8b ）
    llm = Ollama(
        model="qwen3:8b",  # 模型名称，和我们下载的一致
        temperature=RAGConfig.temperature  # 温度不变，0 表示回答更精准
    )
    # 第 3 处修改：可选，增加超时设置（避免模型加载慢导致报错）
    llm.client.timeout = 600
 
    prompt_template = """
你是一个专业的问答助手，请严格基于以下【上下文内容】回答用户的问题。
【约束条件】
1. 如果上下文中包含答案，请用简洁准确的语言回答，并指出信息在文档中的位置。
2. 如果上下文中没有相关信息，请直接回答：根据现有知识库，无法回答该问题。
3. 严禁编造或使用外部知识。
4. 回答保持客观，不添加个人观点。
【上下文内容】
{context}
【用户问题】
{question}
【回答】：
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
 
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG问答链构建完成")
    return qa_chain
 
 
def rag_qa(qa_chain, retriever, question: str):
    """执行问答，美化输出格式（v4.0新增重排序调用）"""
    print(f"\n正在检索：'{question}'")
    # 1. 向量检索（召回6个候选片段）
    # 步骤1：使用 LCEL 链获取答案
    answer = qa_chain.invoke(question)
    
    # 步骤2：获取源文档并进行重排序
    sources = retriever.invoke(question)
    sources = rerank_documents(
        question, 
        sources, 
        RAGConfig.k_rerank
    )
 
    # 美化输出（和v3.0完全一致，无需修改）
    print("\n" + "="*60)
    print("回答：")
    print("-"*60)
    print(answer)
    print("\n答案来源：")
    print("-"*60)
    for i, doc in enumerate(sources, 1):
        source_info = f"{i}. {doc.metadata.get('source', '未知来源')}"
        if 'page' in doc.metadata:
            source_info += f" (第 {doc.metadata['page'] + 1} 页)"
        preview = doc.page_content[:120] + "..." if len(doc.page_content) > 120 else doc.page_content
        print(source_info)
        print(f"   预览：{preview}")
    print("="*60)
 
 
def load_folder_documents(folder_path: str) -> List[Document]:
    if not os.path.exists(folder_path):
        print(f"文件夹不存在：{folder_path}")
        return []
 
    all_docs = []
    supported = ('.pdf', '.docx', '.txt')
    print(f"\n扫描文件夹：{folder_path}")
    for file in os.listdir(folder_path):
        fp = os.path.join(folder_path, file)
        if os.path.isfile(fp) and fp.lower().endswith(supported):
            print(f"发现文档：{file}")
            docs = load_documents(fp)
            if docs:
                all_docs.extend(docs)
    print(f"\n文件夹加载完成，共加载 {len(all_docs)} 页/段")
    return all_docs
 
 
def main():
    print("个人私有知识库 RAG 系统 v4.0")
    print("="*60)
    print()
 
    # DOC_PATH = "./docs/RAG教程.pdf"
    DOC_PATH = "./docx/简介.txt"
    documents = load_documents(DOC_PATH)
    # documents = load_folder_documents("./docs")
 
    if not documents:
        print("文档加载失败，程序退出")
        return
 
    splits = split_documents(documents)
    vectordb, embeddings = get_or_create_vector_db(splits)
    retriever = create_retriever(vectordb)
    qa_chain = build_rag_chain(retriever)
 
    print("\n问答交互已启动（输入 q 退出）")
    while True:
        q = input("\n请输入你的问题：")
        if q.lower() in ['q', 'quit', 'exit']:
            print("感谢使用，再见！")
            break
        if not q.strip():
            print("问题不能为空，请重新输入")
            continue
        rag_qa(qa_chain, retriever, q)
 
 
if __name__ == "__main__":
    main()
