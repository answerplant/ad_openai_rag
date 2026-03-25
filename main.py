#!/usr/bin/python
# Set OpenAI key as an environmental variable with $Env:OPENAI_API_KEY = 'your_key_here'
# Echo the key with $Env:OPENAI_API_KEY
# https://docs.langchain.com/oss/python/integrations/embeddings/openai

import os
import getpass
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
)

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0
)

texts = [
    "Answer Digital is a technology consultancy in Leeds, West Yorkshire, UK.",
    "Answer Digital was founded in 1998.",
    "Answer Digital's favourite colour is yellow.",
    "Answer Digital has 100 employees."
]

vectorstore = InMemoryVectorStore.from_texts(
    texts,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know.\n\n"
    "{context}"
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("Where is Answer Digital located and when was it founded?")
print(response)