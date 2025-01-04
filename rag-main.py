import os

import numpy as np
import requests
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Web Search Function
def web_search(query, num_results=3):
    search_api_key = ""
    search_engine_id = ""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={search_api_key}&cx={search_engine_id}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("items", [])
        return [
            {"title": item["title"], "link": item["link"]}
            for item in results[:num_results]
        ]
    else:
        return []


# Response Validation Function
def validate_response(answer, documents, embedding_model):
    doc_embeddings = embedding_model.embed_documents(
        [doc.page_content for doc in documents]
    )
    answer_embedding = embedding_model.embed_query(answer)
    similarities = cosine_similarity([answer_embedding], doc_embeddings)[0]
    avg_similarity = np.mean(similarities)
    return avg_similarity >= 0.5


# Hallucination Percentage Calculation
def calculate_hallucination_percentage(answer, sources, embedding_model):
    if not sources:
        return 100
    source_texts = [
        source.page_content if hasattr(source, "page_content") else source
        for source in sources
    ]
    source_embeddings = embedding_model.embed_documents(source_texts)
    answer_embedding = embedding_model.embed_query(answer)
    similarities = cosine_similarity([answer_embedding], source_embeddings)[0]
    avg_similarity = np.mean(similarities)
    hallucination_percentage = (1 - avg_similarity) * 100
    return max(0, min(hallucination_percentage, 100))


# Embedding Model
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]


# RAG Application
class RAGApplication:
    def __init__(self, retriever, llm, embedding_model):
        self.retriever = retriever
        self.llm = llm
        self.embedding_model = embedding_model

    def run(self, question):
        # Retrieve documents from the vectorstore
        documents = self.retriever.invoke(question)

        if not documents:  # Fallback to web search if no documents are found
            web_results = web_search(question)
            if web_results:
                web_texts = "\n".join([result["title"] for result in web_results])
                # Generate answer using web results
                prompt_with_web = PromptTemplate(
                    template="""You are a model designed to provide precise and contextually relevant answers based on web search results.
                                Question: {question}
                                Web Results: {web_results}
                                Answer:""",
                    input_variables=["question", "web_results"],
                )
                rag_chain_with_web = prompt_with_web | self.llm | StrOutputParser()
                answer = rag_chain_with_web.invoke(
                    {"question": question, "web_results": web_texts}
                )
                hallucination_percentage = calculate_hallucination_percentage(
                    answer, web_results, self.embedding_model
                )
                return {
                    "answer": answer,
                    "sources": web_results,
                    "hallucination_percentage": hallucination_percentage,
                }
            else:
                return {
                    "answer": "No reliable information found from web search.",
                    "sources": [],
                    "hallucination_percentage": 100,
                }

        # Pro
        doc_texts = "\n".join([doc.page_content for doc in documents])
        prompt_with_docs = PromptTemplate(
            template=""""You are a model designed to provide precise, structured, and contextually relevant answers to legal and jurisprudential questions. Your responses must be:

    Direct: Start immediately with the explanation or answer.
    In the first sentence itself, the answer to the question must be there. dont do any formatting except bold in the first sentence.
    Structured: Use clear numbering, headings, or bullet points for readability.
    Accurate: Cite concepts, laws, or principles explicitly and without overgeneralization.
    Concise but Comprehensive: Avoid verbose explanations while ensuring the response fully addresses the query.

Example Questions and Expected Answers:

    Question: According to Bentham, “every law may be considered in eight different aspects.” Discuss.
    Answer:
        Source: The will of the sovereign, who issues or adopts laws.
        Subjects: Persons or entities to whom the law applies, either as active or passive subjects.
        Objects: The goals or objectives of the law.
        Extent: The scope of the law, either geographical (direct) or relational (indirect).
        Aspects: Directive and sanctional components of the sovereign's will.
        Force: The motivation to comply with the law.
        Remedial Appendage: Subsidiary laws for judicial remedies.
        Expression: The articulation of the sovereign's intent through the law.

    Question: Article 16 qualifies equality of opportunity in matters of public employment. However, there are certain exceptions to it. Discuss.
    Answer:
        Article 16(1): Guarantees equality of opportunity in public employment.
        Article 16(2): Prohibits discrimination based on religion, race, caste, sex, descent, place of birth, or residence.
        Exceptions:
            Article 16(3): Parliament may require residence as a qualification for specific employment.
            Article 16(4): Reservation for backward classes underrepresented in state services.
            Article 16(4A): Reservation in promotions for Scheduled Castes and Tribes.
            Article 16(4B): Carrying forward unfilled reserved vacancies.
            Article 16(5): Religious institutions can prescribe faith-based qualifications for certain roles.
            Article 16(6): Reservation for economically weaker sections up to 10%.
                        Question: {question}
                        Documents: {documents}
                        Answer:""",
            input_variables=["question", "documents"],
        )
        rag_chain_with_docs = prompt_with_docs | self.llm | StrOutputParser()
        answer = rag_chain_with_docs.invoke(
            {"question": question, "documents": doc_texts}
        )

        # Validate response
        if not validate_response(answer, documents, self.embedding_model):
            web_results = web_search(question)
            if web_results:
                web_texts = "\n".join([result["title"] for result in web_results])
                # Generate answer using web results
                prompt_with_web = PromptTemplate(
                    template="""You are a model designed to provide precise and contextually relevant answers based on web search results.
                                Question: {question}
                                Web Results: {web_results}
                                Answer:""",
                    input_variables=["question", "web_results"],
                )
                rag_chain_with_web = prompt_with_web | self.llm | StrOutputParser()
                answer = rag_chain_with_web.invoke(
                    {"question": question, "web_results": web_texts}
                )
                hallucination_percentage = calculate_hallucination_percentage(
                    answer, web_results, self.embedding_model
                )
                return {
                    "answer": answer,
                    "sources": web_results,
                    "hallucination_percentage": hallucination_percentage,
                }
            else:
                return {
                    "answer": "The generated answer does not align well with available sources, and no web search results were found.",
                    "sources": [],
                    "hallucination_percentage": 100,
                }

        hallucination_percentage = calculate_hallucination_percentage(
            answer, documents, self.embedding_model
        )
        return {
            "answer": answer,
            "sources": [doc.page_content for doc in documents],
            "hallucination_percentage": hallucination_percentage,
        }


# Streamlit App
st.set_page_config(page_title='ILLegaLlaMa', page_icon='⚖', layout='wide')
st.title("Conversational AI with Chat Management")

# Initialize chat management
if "chats" not in st.session_state:
    st.session_state["chats"] = {}
if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = "Chat 1"
if st.session_state["current_chat"] not in st.session_state["chats"]:
    st.session_state["chats"][st.session_state["current_chat"]] = []

# Sidebar for chat management
with st.sidebar:
    st.header("Chats")
    for chat_name, chat_history in st.session_state["chats"].items():
        summary = chat_history[0][0] if chat_history else "No messages"
        if st.button(f"{chat_name}: {summary[:20]}...", key=f"switch_{chat_name}"):
            st.session_state["current_chat"] = chat_name
    if st.button("Start New Chat"):
        new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
        st.session_state["chats"][new_chat_name] = []
        st.session_state["current_chat"] = new_chat_name
    if st.button("Clear Current Chat"):
        st.session_state["chats"][st.session_state["current_chat"]] = []

# Display chat history
st.write(f"### {st.session_state['current_chat']}")
chat_history = st.session_state["chats"][st.session_state["current_chat"]]
for user_query, bot_response in chat_history:
    st.markdown(f"**You:** {user_query}")
    st.markdown(f"**AI:** {bot_response}")

# Chat input
question = st.text_area("Ask a question:")
if st.button("Submit"):
    if question.strip():
        with st.spinner("Generating answer..."):
            file_paths = ["combined_text.txt"]
            docs = [TextLoader(file_path).load() for file_path in file_paths]
            docs_list = [item for sublist in docs for item in sublist]
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(docs_list)
            embedding_model = HuggingFaceEmbeddings()
            index_filepath = "faiss_index"
            if os.path.exists(index_filepath):
                vectorstore = FAISS.load_local(
                    index_filepath,
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
            else:
                vectorstore = FAISS.from_documents(doc_splits, embedding_model)
                vectorstore.save_local(index_filepath)
            retriever = vectorstore.as_retriever(k=4)
            llm = ChatOllama(model="llama3.1", temperature=0)
            rag_application = RAGApplication(retriever, llm, embedding_model)
            result = rag_application.run(question)
            st.session_state["chats"][st.session_state["current_chat"]].append(
                (question, result["answer"])
            )
            st.markdown(f"**You:** {question}")
            st.markdown(f"**AI:** {result['answer']}")

            st.subheader("Hallucination Percentage")
            st.write(f"{result['hallucination_percentage']:.2f}%")
            st.subheader("Sources")
            if result["sources"]:
                for source in result["sources"]:
                    if (
                        isinstance(source, dict)
                        and "link" in source
                        and "title" in source
                    ):
                        st.markdown(f"- [{source['title']}]({source['link']})")
                    else:
                        st.write(f"- {source}")
            else:
                st.write("No sources available.")
    else:
        st.error("Please enter a question before clicking the button.")
