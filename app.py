# import the modules and methods
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
import bs4

# data sources
data_1 = "https://www.who.int/health-topics/mpox#tab=tab_1"     # overview, symptoms, treatment and prevention
data_2 = "https://www.who.int/news-room/fact-sheets/detail/mpox" # facts sheet
data_3 = "https://www.who.int/news-room/questions-and-answers/item/mpox" # questions and answers
data_4 = "https://www.who.int/news-room/questions-and-answers/item/testing-for-mpox--health-workers" # faq_health_workers
data_5 = "https://www.who.int/news-room/questions-and-answers/item/testing-for-mpox--individuals-and-communities" # faq_communities 
# data_6 = "https://worldhealthorg.shinyapps.io/mpx_global/#1_Overview"   # trend overview
data_7 = "https://www.who.int/news/item/13-09-2024-who-and-partners-establish-an-access-and-allocation-mechanism-for-mpox-vaccines--treatments--tests"  # vaccine
data_8 = "https://www.who.int/news/item/13-09-2024-who-prequalifies-the-first-vaccine-against-mpox"  # vaccine 
data_9 = "https://www.who.int/news/item/31-08-2024-unicef-issues-emergency-tender-to-secure-mpox-vaccines-for-crisis-hit-countries-in-collaboration-with-africa-cdc--gavi-and-who"   # vaccine
data_10 = "https://www.unicef.org/stories/mpox-and-children"    # mpox and children
data_11 = "https://mpoxvaccine.cdc.gov/" # vaccine cdc
data_12 = "https://www.cdc.gov/mpox/vaccines/index.html"   # vaccine cdc
data_13 = "https://africacdc.org/news-item/africa-cdc-declares-mpox-a-public-health-emergency-of-continental-security-mobilizing-resources-across-the-continent/"  # africa cdc mpox
data_14 = "https://africacdc.org/mpox/"  # africa cdc

# load the data
loader = WebBaseLoader(
    web_paths=(data_1, data_2, data_3, data_4, data_5, data_7, data_8, data_9, data_10, data_11, data_12, data_13, data_14),
bs_kwargs=dict(
        parse_only=bs4.SoupStrainer()
    )
)
docs = loader.load()
# print(docs)

# remove the \n, \t, \r
def clean_docs(docs):
    for doc in docs:
        doc.page_content = doc.page_content.replace("\n", "").replace("\t", "").replace("\r", "")
    return docs


cleaned_docs = clean_docs(docs)

# print(cleaned_docs)

# split documents into chunks
def split_docs(doc_to_split):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
  )
  return text_splitter.split_documents(doc_to_split)


#len(splitted_docs) helps to see length of splitted documents
spliitted_docs = split_docs(cleaned_docs)
# print(spliitted_docs)

# load the environment variable
load_dotenv()
GOOGLE_API_KEY = os.getenv("google_api_key")

# create and store the docs as embeddings
def store_embeddings(docs_to_embed):
  vectorstore = Chroma.from_documents(
    documents=docs_to_embed,
    embedding=GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001"),
  )

  return vectorstore


vectorstore = store_embeddings(spliitted_docs)

# retrieve documents
retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
        )
# user_query = "what is mpox?"


# retrieved_docs = retriever.invoke(user_query)
# print(retrieved_docs)

# set the large language model
llm = GoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash",
    temperature=0.1
)

# this prompt helps to formulate query for retrieving documents with past conversation in mind
history_retriever_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",  "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# use the formulated query for retrieving from past conversation and original documents
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    history_retriever_prompt
)

# this prompt tells the llm to answer questions based on a users question and retrieved context with history if there is
main_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
    You are mpoxGPT, an AI assistant specialized in answering questions and offering understanding about mpox or monkey pox.
    You were developed by Simeon Krah as a RAG system based on google's gemini models.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    {context}
    """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]

)

# create a chain to specify how the retrieved context is fed into prompt
question_answer_chain = create_stuff_documents_chain(llm, main_prompt)

# create a retrieval chain (putting all together)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# function to store session messages
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# conversation chain that takes session id into consideration and interact with history in mind
final_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# call the chain to answer questions
operation = final_rag_chain.invoke(
    {"input": "what is mpox"},
    config={
        "configurable": {"session_id": "any"}
    },  # constructs a key "any" in `store`.
)

# operation returns dictionary with answer, context, history and input keys
print(operation["answer"])

