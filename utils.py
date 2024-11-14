import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.vectorstores import Chroma

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def init_conversation(file_path: str):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(data, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Reference: https://towardsdatascience.com/enhancing-csv-file-query-performance-in-chatgpt-3e2b67a5f867
    system_template = """
    The provided {context} is a tabular dataset containing products along with their image links crawled from the Amazon website.
    The dataset includes the following columns:
    'product': name of the product,
    'image_link': link to the image of the product,
    ----------------
    {context}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        memory=memory,
        verbose=False
    )
    return qa

def chat_with_llm(qa, query: str):
    return qa.run({"question": query})
