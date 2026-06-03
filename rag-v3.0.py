# -*- coding: utf-8 -*-
"""
RAG 个人私有知识库 v3.0
核心升级：零成本接入本地大模型（Ollama），彻底告别 OpenAI
支持：PDF / Word / TXT、向量库缓存、MMR 检索、中文优化、编码兼容、100% 离线运行
适合无外网、无 API Key
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 
os.environ["TRANSFORMERS_OFFLINE"] = "0" # 允许联网下载 
os.environ["HF_HUB_OFFLINE"] = "0" # 然后再import其他模块 
from typing import List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings 
from langchain_ollama import OllamaLLM as Ollama
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
 
 
class RAGConfig:
    """统一配置项（v3.0 优化中文适配）"""
    # 文本拆分参数
    chunk_size = 500  # 每个片段字符数（中文约250字）
    chunk_overlap = 50  # 片段重叠字符数
    # 检索参数
    k_retrieval = 3  # 召回最相关的片段数
    retrieval_score_threshold = 0.5  # 相似度阈值
    # 向量库参数
    persist_directory = "./chroma_db"  # 向量库存储路径
    embedding_model = "quentinz/bge-large-zh-v1.5"  # 中文向量模型（效果更好）
    # 大模型参数（本地模型，无需配置 API Key）
    temperature = 0.1  # 温度越低，回答越精准
    # 文件编码
    encoding = "utf-8"
 
 
def load_documents(file_path: str) -> Optional[List[Document]]:
    """加载单文档，支持 PDF / Word / TXT，自动兼容编码"""
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
            # 自动兼容 utf-8 / gbk，解决中文乱码问题
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
    """文本分块，中文友好，避免拆分到句子中间"""
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
 
 
def get_or_create_vector_db(splits: List[Document]) -> Tuple[Chroma, OllamaEmbeddings]:
    """向量库创建或加载（带缓存，避免重复向量化）"""
    print(f"\n初始化向量化模型：{RAGConfig.embedding_model}")
    print("首次运行会自动下载模型，耐心等待...")

    embeddings = OllamaEmbeddings(model=RAGConfig.embedding_model)

    # 检查是否存在已有的向量库
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
        print(f"向量库信息：包含 {vectordb._collection.count()} 个向量")

    return vectordb, embeddings
 
 
def create_retriever(vectordb: Chroma):
    """MMR 检索器，兼顾相关性和结果多样性，避免重复回答"""
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
    """构建 RAG 问答链（v3.0 核心：接入本地 Ollama 模型，langchain 1.0+）"""
    print("\n初始化本地大模型...")
    # 配置本地大模型（Ollama），模型名称和下载的一致
    llm = Ollama(
        model="qwen:8b",  # 这里换成你下载的模型（qwen:4b / qwen:7b 等）
        temperature=RAGConfig.temperature,
        client_kwargs={"timeout": 600}  # 超时设置，避免加载慢报错
    )

    prompt_template = """
你是一个专业的问答助手，请严格基于以下【上下文内容】回答用户的问题。
【约束条件】
1. 如果上下文中包含答案，请用简洁准确的语言回答，并指出信息在文档中的位置。
2. 如果上下文中没有相关信息，请直接回答：根据现有知识库，无法回答该问题。
3. 严禁编造或使用外部知识。
4. 回答保持客观，不添加个人观点，语言通俗易懂。
【上下文内容】
{context}
【用户问题】
{question}
【回答】：
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 构建 LCEL 风格的链（langchain 1.0+ 推荐方式）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG问答链构建完成（本地模型已接入，可离线运行）")
    return rag_chain
 
 
def rag_qa(qa_chain, retriever, question: str):
    """执行问答，美化输出格式（和 v2.0 一致）"""
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
    """批量加载文件夹内所有支持的文档（批量处理，新手可选）"""
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
    """主函数，一键运行，新手直接执行即可"""
    print("个人私有知识库 RAG 系统 v3.0")
    print("="*60)
    print("核心特性：零成本、本地大模型、100% 离线、中文友好")
    print("="*60)
    print()
 
    # ====================== 新手必改：修改你的文档路径 ======================
    # 单文件（推荐新手先试单文件，容易成功）
    DOC_PATH = "./docs/RAG教程.pdf"  # 替换成你的本地文档路径（PDF/Word/TXT）
    documents = load_documents(DOC_PATH)
 
    # 批量文件夹（熟练后再试，注释掉上面一行，取消下面一行注释）
    # documents = load_folder_documents("./docs")  # 批量加载 docs 文件夹下的所有文档
    # ======================================================================
 
    if not documents:
        print("文档加载失败，程序退出")
        return
 
    # 执行完整流程（和 v2.0 一致，无需修改）
    splits = split_documents(documents)
    vectordb, embeddings = get_or_create_vector_db(splits)
    retriever = create_retriever(vectordb)
    qa_chain = build_rag_chain(retriever)

    # 问答交互（和 v2.0 一致）
    print("\n问答交互已启动（输入 q 退出，断网也能使用）")
    while True:
        q = input("\n请输入你的问题：")
        if q.lower() in ['q', 'quit', 'exit']:
            print("感谢使用，再见！")
            break
        if not q.strip():
            print("问题不能为空，请重新输入")
            continue
        # 执行问答，捕获异常，避免崩溃
        try:
            rag_qa(qa_chain, retriever, q)
        except Exception as e:
            print(f"问答执行失败：{str(e)}")
            print("建议：检查 Ollama 是否正常运行，或模型是否下载成功")
 
 
if __name__ == "__main__":
    main()