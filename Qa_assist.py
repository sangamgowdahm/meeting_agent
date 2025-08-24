from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_deepseek import ChatDeepSeek

def create_qa_chain(vectorstore, api_key):
    """Create a QA chain for answering questions"""

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Context: {context}
        Question: {question}
        Answer:
        """,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatDeepSeek(
        model="deepseek-chat",      # DeepSeek V3 model
        api_key=api_key,
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )
