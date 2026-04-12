"""RAG功能实现"""
from langchain_core import ChatPromptTemplate, RunnablePassthrough, StrOutputParser
from app.llm import get_llm
from app.vectorstore import load_vector_store


def build_rag_chain():
    """构建RAG链"""
    vector_store = load_vector_store()
    if vector_store is None:
        raise ValueError("向量存储不存在，请先运行build_index.py构建索引")
    
    retriever = vector_store.as_retriever()
    llm = get_llm()
    
    template = """
    你是一个基于文档的问答助手，根据提供的上下文回答用户问题。
    请严格基于上下文信息回答，不要添加超出上下文的内容。
    
    上下文:
    {context}
    
    问题:
    {question}
    
    回答:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def ask_question(question):
    """使用RAG链回答问题"""
    rag_chain = build_rag_chain()
    return rag_chain.invoke(question)