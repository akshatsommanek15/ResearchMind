import tempfile
from pathlib import Path
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="ResearchMind",
    page_icon="📚",
    layout="wide"
)

st.title("📚 ResearchMind")
st.write(
    "Your AI-powered research assistant. "
    "Upload research papers and ask questions about them."
)

groq_api_key = os.getenv("GROQ_API_KEY" or st.secrets.get("GROQ_API_KEY"))
if not groq_api_key:
    st.error("GROQ_API_KEY was not found in your .env file.")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="openai/gpt-oss-20b"
)

if "store" not in st.session_state:
    st.session_state["store"] = {}

if "vectors" not in st.session_state:
    st.session_state["vectors"] = None

if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

with st.sidebar:
    st.header("Research Papers")

    session_id = st.text_input(
        "Session ID",
        value="default_session"
    )

    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_clicked = st.button(
        "Process PDFs",
        disabled=not uploaded_files
    )

if process_clicked:
    try:
        with st.spinner("Processing research papers..."):
            documents = []

            with tempfile.TemporaryDirectory() as temp_dir:
                for index, uploaded_file in enumerate(uploaded_files):
                    temppdf = Path(temp_dir) / f"paper_{index}.pdf"
                    temppdf.write_bytes(uploaded_file.getvalue())

                    loader = PyPDFLoader(str(temppdf))
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(documents)

            st.session_state["vectors"] = FAISS.from_documents(
                documents=chunks,
                embedding=st.session_state["embeddings"]
            )

            st.session_state["processed_files"] = [
                file.name for file in uploaded_files
            ]

        st.success(f"Processed {len(uploaded_files)} PDF(s) successfully.")

    except Exception as e:
        st.error(f"Could not process PDFs: {e}")



def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

if st.session_state["vectors"] is None:
   st.info("Upload your PDFs and click 'Process PDFs' to begin.")       

else:
    retriever = st.session_state["vectors"].as_retriever(search_kwargs={"k": 4})
    contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
)
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)    

    system_prompt = (
        """
        You are a research paper assistant.

        Answer the question using only the provided context.

        Instructions:

        1. Give a clear and accurate answer.
        2. Explain technical concepts in simple language.
        3. Do not invent facts or information.
        4. If the answer is not in the context, respond exactly:
        "The information is not available in the uploaded documents."
        5. Do not use LaTeX. Write equations in readable plain text.
        6. ALWAYS format your answer as bullet points.
        7. Do not write long paragraphs.
        8. Start every point with "- ".
        10. Do not include an introductory paragraph before the bullet points.
        11. Output only the answer, without greetings or introductory phrases.
        12. Include only claims supported by the provided context.
        13. Do not add specific details, examples, layer numbers, or implementation
            steps unless they are supported by the context.
        14. Do not repeat the same information.

        Context:
        {context}

        Answer (bullet points only):
        """)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    history = get_session_history(session_id)

    user_input = st.chat_input("Ask about your research papers")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
        st.write("Answer:")
        st.write(response['answer'])
        st.write("Sources")

        with st.expander("Sources and citations"):
                seen = set()

                for doc in response["context"]:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page_label")

                    if page is None:
                        page_index = doc.metadata.get("page")
                        page = (
                            int(page_index) + 1
                            if page_index is not None
                            else "Unknown"
                        )

                    citation = (source, page)

                    if citation not in seen:
                        st.markdown(f"- **{source} — Page {page}**")
                        seen.add(citation)


with st.sidebar:
    st.subheader("Previous Questions")

    history = get_session_history(session_id)

    for message in history.messages:
        if isinstance(message, HumanMessage):
            st.write("•", message.content)

