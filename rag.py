import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize Streamlit app
st.title("Welcome! How can I help you today?")

# Chat history initialization
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat history
st.subheader("Chat History:")
with st.container():
    for qa in st.session_state["chat_history"]:
        st.markdown(f"**You:** {qa['question']}")
        st.markdown(f"**Assistant:** {qa['answer']}")
    st.markdown("---")

# Input text box
question = st.text_area("Enter your question:", placeholder="Type your question here...")

if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Generating answer..."):
            answer = ''
            file_paths = [
                'combined_text.txt',
            ]

            # Load and split text
            docs = [TextLoader(file_path).load() for file_path in file_paths]
            docs_list = [item for sublist in docs for item in sublist]

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=0
            )

            doc_splits = text_splitter.split_documents(docs_list)

            # Embeddings for deep vector representation using FAISS
            class HuggingFaceEmbeddings(Embeddings):
                def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, texts):
                    return self.model.encode(texts)

                def embed_query(self, text):
                    return self.model.encode([text])[0]

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # Look for cached vectorstore and use if seen, else generate
            index_filepath = "faiss_index"

            if os.path.exists(index_filepath):
                vectorstore = FAISS.load_local(index_filepath, embedding_model, allow_dangerous_deserialization=True)
            else:
                vectorstore = FAISS.from_documents(doc_splits, embedding_model)
                vectorstore.save_local(index_filepath)

            retriever = vectorstore.as_retriever(k=4)

            # Prompt engineering, the core of the behavior of the model
            prompt = PromptTemplate(
                template="""You are an assistant for question-answering tasks.
                Use the following documents to answer the question.
                If you don't know the answer, just say that you don't know.
                Use as many sentences as you want but be accurate and detailed to some degree:
                Question: {question}
                Documents: {documents}
                Answer:
                """,
                input_variables=["question", "documents"],
            )

            # Temp = 0 for deterministic responses, ollama server SHOULD be running in background
            llm = ChatOllama(
                model="llama3.1",  # <--- Base Model from ollama
                temperature=0,
            )

            # Pipeline for RAG
            rag_chain = prompt | llm | StrOutputParser()

            class RAGApplication:
                def __init__(self, retriever, rag_chain):
                    self.retriever = retriever
                    self.rag_chain = rag_chain

                def run(self, question):
                    documents = self.retriever.invoke(question)
                    doc_texts = "\n".join([doc.page_content for doc in documents])
                    answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
                    return answer

            # Actually running the RAG
            rag_application = RAGApplication(retriever, rag_chain)
            answer = rag_application.run(question)
        
        # Save to chat history
        st.session_state["chat_history"].append({"question": question, "answer": answer})
        
        # Display the answer
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a question before clicking the button.")
