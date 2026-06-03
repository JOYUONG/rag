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
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()
api_key = ""
database_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#database_url = "https://dashscope.aliyuncs.com/api/v1"
class RAGConfig:
    """统一配置项"""
    chunk_size = 500
    chunk_overlap = 50
    k_retrieval = 3
    retrieval_score_threshold = 0.5
    persist_directory = "./chroma_db"
    embedding_model = "BAAI/bge-small-zh-v1.5"
    llm_model = "qwen-plus"
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
def format_docs(docs):
    """格式化检索到的文档"""
    return "\n\n".join(doc.page_content for doc in docs)
 
def build_rag_chain(retriever):
    print("\n初始化大模型...")
    
    # llm = ChatOpenAI(
    #     model_name=RAGConfig.llm_model,
    #     temperature=RAGConfig.temperature
    # )
    from langchain_qwq import ChatQwen

    llm = ChatQwen(
        model_name="qwen-vl-plus-2025-05-07",
        temperature=0.3,
        timeout=10,
        max_tokens=1000
    )

    print("大模型初始化完成")
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

        # 构建 LCEL 风格的链（langchain 1.0+ 推荐方式）
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG问答链构建完成")
    return qa_chain
 
 
def rag_qa(qa_chain, retriever, question: str):
    """执行问答，美化输出格式（和 v2.0 一致，新手易读）"""
    print(f"\n正在检索：'{question}'")
    sources = retriever.invoke(question)
    answer = qa_chain.invoke(question)

    # 美化输出，和 v2.0 保持一致，方便对比
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
        # 片段预览，避免过长
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
    print("个人私有知识库 RAG 系统 v2.0")
    print("="*60)
    print()
 
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
