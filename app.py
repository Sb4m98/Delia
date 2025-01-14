import openai
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import fitz
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from streamlit_pdf_viewer import pdf_viewer
import os
from typing import List, Tuple, Dict, Any, Optional
from langchain.prompts import PromptTemplate
import shutil
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import re
from langchain.chains.base import Chain
from langchain.chains import LLMChain
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class DocumentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('italian') + stopwords.words('english'))
    
    def analyze_text(self, text: str) -> Dict:
        # Tokenizzazione e pulizia
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Analisi
        word_count = len(words)
        unique_words = len(set(words))
        word_freq = Counter(words).most_common(10)
        
        # Calcolo della complessità del testo (lunghezza media delle parole)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        return {
            'word_count': word_count,
            'unique_words': unique_words,
            'word_freq': word_freq,
            'avg_word_length': avg_word_length,
            'vocabulary_density': unique_words / word_count if word_count > 0 else 0
        }
    
class PDFHighlighter:
    def __init__(self):
        self.highlight_color = (1, 0.85, 0)  # Yellow highlight

    def highlight_pdf(self, pdf_path: str, source_docs: List[Any]) -> str:
        """Create a new PDF with highlighted sections."""
        doc = fitz.open(pdf_path)
        
        for source_doc in source_docs:
            # Ensure page number is valid and zero-based
            page_num = source_doc.metadata.get('page', 1)
            if isinstance(page_num, str):
                try:
                    page_num = int(page_num)
                except ValueError:
                    page_num = 1
            
            # Convert to zero-based index and validate
            page_idx = page_num - 1 if page_num > 0 else 0
            if page_idx >= doc.page_count:
                page_idx = doc.page_count - 1
            
            chunk_text = source_doc.page_content
            
            page = doc[page_idx]
            # Get all instances of the text in the page
            text_instances = self.find_text_on_page(page, chunk_text)
            
            # Highlight each instance
            for rect_list in text_instances:
                quads = [rect.quad for rect in rect_list if hasattr(rect, 'quad')]
                
                if quads:  # Only add highlight if we found matches
                    annot = page.add_highlight_annot(quads)
                    annot.set_colors(stroke=self.highlight_color)
                    annot.update()
        
        # Save highlighted PDF
        output_path = f"{pdf_path}_highlighted.pdf"
        doc.save(output_path)
        doc.close()
        return output_path

    def find_text_on_page(self, page: fitz.Page, search_text: str) -> List[List[fitz.Rect]]:
        """Find all instances of text on the page and return their rectangles."""
        # Clean and prepare the search text
        search_text = self._prepare_text_for_search(search_text)
        
        # Split into sentences for more accurate matching
        sentences = nltk.sent_tokenize(search_text)
        all_matches = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 4:  # Skip very short sentences
                continue
                
            # Search for text instances on the page
            matches = page.search_for(sentence.strip())
            if matches:
                all_matches.append(matches)
        
        return all_matches

    def _prepare_text_for_search(self, text: str) -> str:
        """Clean and prepare text for searching."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove special characters that might interfere with search
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    def extract_text_from_pdf(self, pdf_doc: Any, pdf_index: int) -> List[Tuple[str, Dict[str, Any]]]:
        texts_with_metadata = []
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"temp_{pdf_index}.pdf")
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(pdf_doc.getvalue())
            
            doc = fitz.open(temp_path)
            pdf_info = {
                'filename': pdf_doc.name,
                'pdf_index': pdf_index,
                'total_pages': doc.page_count
            }
            
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                text = page.get_text("text", sort=True).strip()
                if text:
                    text = self._preprocess_text(text)
                    metadata = {
                        'page': page_number + 1,  # Store as 1-based page numbers
                        'pdf_info': pdf_info,
                        'word_locations': page.get_text("words")
                    }
                    texts_with_metadata.append((text, metadata))
            
            doc.close()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return texts_with_metadata

    def _preprocess_text(self, text: str) -> str:
        text = ' '.join(text.split())
        text = ''.join(char for char in text if char.isprintable())
        return text

    def create_chunks(self, texts_with_metadata: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        chunks_with_metadata = []
        for text, metadata in texts_with_metadata:
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': f"chunk_{metadata['pdf_info']['pdf_index']}_{metadata['page']}_{i+1}",
                    'text_length': len(chunk)
                })
                chunks_with_metadata.append((chunk, chunk_metadata))
        return chunks_with_metadata
    
    def process_and_analyze(self, pdf_doc: Any, pdf_index: int) -> Tuple[List[Tuple[str, Dict]], Dict]:
        texts_with_metadata = self.extract_text_from_pdf(pdf_doc, pdf_index)
        
        # Analisi del documento
        full_text = ' '.join([text for text, _ in texts_with_metadata])
        analyzer = DocumentAnalyzer()
        analysis_results = analyzer.analyze_text(full_text)
        
        return texts_with_metadata, analysis_results

class RelevanceScorer:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Calcola la similarità semantica tra due testi."""
        emb1 = self.embeddings.embed_query(text1)
        emb2 = self.embeddings.embed_query(text2)
        return cosine_similarity([emb1], [emb2])[0][0]

class EnhancedConversationalRetrievalChain(Chain):
    """Versione migliorata della catena di recupero conversazionale."""
    
    retriever: Any
    llm_chain: Any
    question_generator: Any
    memory: Any
    scorer: RelevanceScorer
    min_similarity_threshold: float = 0.75
    
    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer", "source_documents", "relevance_scores"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history = self.memory.chat_memory.messages if self.memory else []
        
        # Generate question considering context
        if chat_history:
            chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in chat_history])
            generated_question = self.question_generator.run(
                question=question,
                chat_history=chat_history_str
            )
        else:
            generated_question = question
        
        # Retrieve documents and calculate relevance
        docs = self.retriever.get_relevant_documents(generated_question)
        
        # Prepare context combining documents
        context = "\n\n".join([d.page_content for d in docs])
        
        # Get response using LLMChain
        response = self.llm_chain.run(
            question=question,
            chat_history=chat_history_str if chat_history else "",
            context=context
        )
        
        # Calculate and validate relevance scores
        relevance_scores = {}
        filtered_docs = []
        
        for doc in docs:
            # Calculate similarity with both question and answer
            question_similarity = self.scorer.compute_similarity(question, doc.page_content)
            answer_similarity = self.scorer.compute_similarity(response, doc.page_content)
            
            # Use weighted average with more weight on answer similarity
            combined_score = (0.3 * question_similarity + 0.7 * answer_similarity)
            
            # Update metadata with validated page information
            if 'page' in doc.metadata:
                # Ensure page numbers are valid
                pdf_info = doc.metadata.get('pdf_info', {})
                total_pages = pdf_info.get('total_pages', 1)
                current_page = doc.metadata['page']
                
                # Adjust page number if it's out of bounds
                if current_page < 1:
                    current_page = 1
                elif current_page > total_pages:
                    current_page = total_pages
                
                doc.metadata['page'] = current_page
            
            doc.metadata['relevance_score'] = combined_score
            relevance_scores[doc.metadata.get('chunk_id', f'chunk_{len(relevance_scores)}')] = {
                'combined_score': combined_score,
                'question_similarity': question_similarity,
                'answer_similarity': answer_similarity,
                'page': doc.metadata.get('page', 1)
            }
            
            if combined_score >= self.min_similarity_threshold:
                filtered_docs.append(doc)
        
        # Sort by combined relevance score
        filtered_docs.sort(key=lambda x: x.metadata['relevance_score'], reverse=True)
        
        # Take only the top most relevant documents
        top_docs = filtered_docs[:3]
        
        return {
            "answer": response,
            "source_documents": top_docs,
            "relevance_scores": relevance_scores
        }

    @classmethod
    def from_llm(
        cls,
        llm: Any,
        retriever: Any,
        memory: Any = None,
        **kwargs: Any
    ) -> "EnhancedConversationalRetrievalChain":
        """Costruisce una nuova istanza della catena migliorata."""
        # Create question generator
        question_prompt = PromptTemplate(
            input_variables=["question", "chat_history"],
            template="""Data la seguente conversazione e una domanda di follow-up, riformula 
            la domanda di follow-up per essere autonoma.
            
            Chat History:
            {chat_history}
            
            Follow Up Input: {question}
            
            Domanda autonoma:"""
        )
        question_generator = LLMChain(llm=llm, prompt=question_prompt)
        
        # Create main response prompt
        response_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""Usa il seguente contesto per rispondere alla domanda. 
            Se non sei sicuro della risposta, dillo onestamente.
            Sii specifico e cita parti rilevanti del contesto quando possibile.
            
            Contesto: {context}
            
            Chat History: {chat_history}
            Domanda: {question}
            
            Risposta dettagliata:"""
        )
        
        # Create the main LLM chain
        llm_chain = LLMChain(llm=llm, prompt=response_prompt)
        
        return cls(
            retriever=retriever,
            llm_chain=llm_chain,
            question_generator=question_generator,
            memory=memory,
            scorer=RelevanceScorer(),
        )

class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    def create_vectorstore(self, chunks_with_metadata: List[Tuple[str, Dict]]) -> FAISS:
        texts = [chunk[0] for chunk in chunks_with_metadata]
        metadatas = [chunk[1] for chunk in chunks_with_metadata]
        return FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)

class PDFManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pdf_files = {}
    
    def save_pdf(self, pdf_doc: Any, pdf_index: int) -> str:
        temp_path = os.path.join(self.temp_dir, f"pdf_{pdf_index}.pdf")
        with open(temp_path, 'wb') as f:
            f.write(pdf_doc.getvalue())
        self.pdf_files[pdf_index] = {
            'path': temp_path,
            'name': pdf_doc.name
        }
        return temp_path
    
    def get_pdf_path(self, pdf_index: int) -> str:
        return self.pdf_files.get(pdf_index, {}).get('path')
    
    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/dPQxgfx/robo.png" alt="Bot">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/NW2NRnb/dxc-logo.png" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>'''

def init_css():
    st.markdown("""
        <style>
        /* Sfondo principale */
        .main {
            background-color: rgb(243, 238, 255); /* Sfondo chiaro coerente con la home */
            padding: 2rem;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: rgb(245, 243, 255); /* Sfondo sidebar */
        }
        
        /* Pulsanti nella sidebar */
        .css-1d391kg .stButton>button {
            background-color: #7C3AED; /* Viola chiaro */
            color: white; /* Testo bianco */
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            transition: background-color 0.3s, color 0.3s;
        }
        
        .css-1d391kg .stButton>button:hover {
            background-color: #581C87; /* Viola scuro per hover */
        }
        
        /* Titoli specifici per la home */
        .home-title {
            color: #581C87; /* Viola scuro */
            font-size: 3.5rem;
            margin-bottom: 2rem;
        }
        /* Titoli */
        h1, h2, h3 {
            color: #581C87; /* Viola scuro */
            font-weight: bold;
        }
        
        /* Paragrafi */
        p {
            color: ##000000; /* Viola chiaro */
        }
        
        /* Pulsanti globali */
        .stButton>button {
            font-size: 1rem;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            background-color: #7C3AED; /* Viola chiaro */
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #581C87; /* Viola scuro per hover */
        }
        
        /* Pulsanti secondari */
        .secondary-button {
            border: 2px solid #7C3AED; /* Viola chiaro */
            color: #7C3AED; /* Viola chiaro */
            background-color: transparent;
        }
        
        .secondary-button:hover {
            background-color: #7C3AED; /* Viola chiaro per hover */
            color: white; /* Testo bianco */
        }

        /* Chat Messages */
        .chat-message {
            padding: 1.5rem; 
            border-radius: 1rem; 
            margin-bottom: 1rem; 
            display: flex;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .chat-message.user {
            background-color: #e0d4fc !important;
            color: #3a007d !important;
        }

        .chat-message.bot {
            background-color: #fff !important;
            color: ##000000 !important;
        }

        /* Avatar Styling */
        .chat-message .avatar {
            width: 12%;
        }

        .chat-message .avatar img {
            max-width: 58px;
            max-height: 58px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 1rem
        }

        </style>
    """, unsafe_allow_html=True)

def home_page():
    # Custom CSS for the layout and logo
    st.markdown("""
        <style>
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2rem auto;
            max-width: 400px;
            padding: 1rem;
        }
        .logo-container svg {
            width: 100%;
            height: auto;
        }
        .subtitle {
            text-align: center;
            color: #4B5563;
            font-size: 13rem;
            margin-bottom: 3rem;
        }
        .feature-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            transition: transform 0.2s ease-in-out;
            border: 1px solid rgba(124, 58, 237, 0.1);
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(124, 58, 237, 0.1);
        }
        .step-circle {
            background-color: #7C3AED;
            color: white;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem auto;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Choose which logo to use by uncommenting one of these sections:

    # OPTION 1: Tech Style Logo
    st.markdown("""
        <div class="logo-container">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 80">
                <rect x="10" y="10" width="50" height="50" transform="rotate(45 35 35)" fill="#7C3AED"/>
                <text x="22" y="50" font-family="Arial" font-weight="bold" font-size="36" fill="white">D</text>
                <text x="70" y="50" font-family="Arial" font-weight="bold" font-size="36" fill="#7C3AED">elia</text>
                <path d="M150 25 L160 25 L160 35" stroke="#7C3AED" stroke-width="2" fill="none"/>
                <circle cx="160" cy="35" r="2" fill="#7C3AED"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)

    # Subtitle
    st.markdown("""
        <p class="subtitle">Document Enhanced Learning Intelligent Assistant</p>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <h3 style='color: #6600A9;'>📄 Analisi Documenti</h3>
                <p>Carica i tuoi PDF e ottieni analisi dettagliate sul contenuto, 
                incluse statistiche su parole chiave e densità del vocabolario.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <h3 style='color: #6600A9;'>💬 Chat Intelligente</h3>
                <p>Interagisci con i tuoi documenti attraverso una chat AI avanzata
                che comprende il contesto e fornisce risposte precise.</p>
            </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("<h2 style='text-align: center; color: #1F2937; margin: 3rem 0 2rem;'>Come Funziona</h2>", 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="text-align: center;">
                <div class="step-circle">1</div>
                <h4 font-weight: bold; margin-bottom: 0.5rem;">Carica</h4>
                <p>Carica i tuoi documenti PDF nel sistema</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="text-align: center;">
                <div class="step-circle">2</div>
                <h4 font-weight: bold; margin-bottom: 0.5rem;">Analizza</h4>
                <p>Il sistema analizza e processa i contenuti</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="text-align: center;">
                <div class="step-circle">3</div>
                <h4 font-weight: bold; margin-bottom: 0.5rem;">Interagisci</h4>
                <p>Chatta con i tuoi documenti e ottieni risposte</p>
            </div>
        """, unsafe_allow_html=True)

def display_source_documents(col2, source_docs, relevance_scores):
    with col2:
        if source_docs:
            st.markdown("### 📑 Fonti di Riferimento")
            
            docs_by_pdf = {}
            for doc in source_docs:
                pdf_index = doc.metadata['pdf_info']['pdf_index']
                pdf_info = doc.metadata['pdf_info']
                relevance_score = doc.metadata.get('relevance_score', 0)
                
                if relevance_score >= 0.75:
                    if pdf_index not in docs_by_pdf:
                        docs_by_pdf[pdf_index] = {
                            'docs': [],
                            'scores': [],
                            'relevant_pages': set(),
                            'relevant_sections': [],
                            'total_score': 0,
                            'total_pages': pdf_info.get('total_pages', 1)
                        }
                    
                    # Ensure page number is valid
                    page = doc.metadata.get('page', 1)
                    if isinstance(page, str):
                        try:
                            page = int(page)
                        except ValueError:
                            page = 1
                    
                    # Validate page number against total pages
                    if page < 1:
                        page = 1
                    elif page > docs_by_pdf[pdf_index]['total_pages']:
                        page = docs_by_pdf[pdf_index]['total_pages']
                    
                    docs_by_pdf[pdf_index]['docs'].append(doc)
                    docs_by_pdf[pdf_index]['scores'].append(relevance_score)
                    docs_by_pdf[pdf_index]['relevant_pages'].add(page)
                    docs_by_pdf[pdf_index]['relevant_sections'].append({
                        'content': doc.page_content,
                        'page': page,
                        'score': relevance_score
                    })
                    docs_by_pdf[pdf_index]['total_score'] += relevance_score
            
            highlighter = PDFHighlighter()
            
            # Sort PDFs by total relevance score
            sorted_pdfs = sorted(docs_by_pdf.items(), 
                               key=lambda x: x[1]['total_score'], 
                               reverse=True)
            
            for pdf_index, pdf_data in sorted_pdfs:
                if pdf_data['docs']:  # Only show if we have relevant docs
                    pdf_info = pdf_data['docs'][0].metadata['pdf_info']
                    avg_score = sum(pdf_data['scores']) / len(pdf_data['scores'])
                    
                    with st.expander(
                        f"📄 {pdf_info['filename']} (Rilevanza: {avg_score:.2%})", 
                        expanded=True
                    ):
                        try:
                            # Create highlighted version of the PDF
                            pdf_path = st.session_state.pdf_manager.get_pdf_path(pdf_index)
                            if pdf_path:
                                highlighted_pdf = highlighter.highlight_pdf(
                                    pdf_path, 
                                    pdf_data['docs']
                                )
                                
                                # Display relevant sections first
                                st.markdown("#### 🔍 Sezioni Rilevanti:")
                                for section in sorted(pdf_data['relevant_sections'], 
                                                    key=lambda x: x['score'], 
                                                    reverse=True):
                                    st.markdown(f"""
                                    ---
                                    **Pagina {section['page']}** (Rilevanza: {section['score']:.2%})
                                    
                                    "{section['content']}"
                                    """)
                                
                                # Display highlighted PDF
                                st.markdown("#### 📑 Documento con Evidenziazioni:")
                                pdf_container = st.container(height=600)  # Imposta l'altezza desiderata
                                with pdf_container:
                                    pdf_viewer(
                                        highlighted_pdf,
                                        pages_to_render=sorted(pdf_data['relevant_pages']),
                                        width=600,
                                        key=f"viewer_{pdf_index}_{'-'.join(map(str, pdf_data['relevant_pages']))}"
                                    )
                                
                                # Display detailed relevance information
                                st.info(f"""
                                📌 Pagine rilevanti: {', '.join(map(str, sorted(pdf_data['relevant_pages'])))}
                                📊 Punteggio medio di rilevanza: {avg_score:.2%}
                                🎯 Numero di sezioni rilevanti: {len(pdf_data['relevant_sections'])}
                                """)
                                
                        except Exception as e:
                            st.error(f"Errore durante l'elaborazione del PDF: {str(e)}")
        else:
            st.info("Nessuna fonte di riferimento trovata per questa risposta.")

def display_document_statistics(analysis_results: Dict, filename: str):
    st.markdown(f"Statistiche per {filename}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Parole totali", analysis_results['word_count'])
    with col2:
        st.metric("Parole Uniche", analysis_results['unique_words'])
    with col3:
        st.metric("Densità del vocabolario", f"{analysis_results['vocabulary_density']:.2%}")
    
    # Word Frequency Graph
    word_freq_df = pd.DataFrame(analysis_results['word_freq'], columns=['Parole', 'Totale'])
    fig = px.bar(
        word_freq_df,
        x='Parole',
        y='Totale',
        title='Parole più comuni',
        template='plotly_white'
    )
    fig.update_traces(marker_color='#02ab21')
    st.plotly_chart(fig, use_container_width=True)

def initialize_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False
    if "pdf_manager" not in st.session_state:
        st.session_state.pdf_manager = PDFManager()

def reset_session():
    """Reset all session state variables and clear uploaded files."""
    # Clear uploaded files
    st.session_state.uploaded_files = []
    st.session_state.files_processed = False
    st.session_state.analyzed_files = []  
    
    # Clear chat history and conversation
    if "chat_history" in st.session_state:
        del st.session_state.chat_history
    if "conversation" in st.session_state:
        del st.session_state.conversation
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    if "analyses" in st.session_state:
        del st.session_state.analyses
    if "current_sources" in st.session_state:
        del st.session_state.current_sources
    # Reset PDF manager
    if "pdf_manager" in st.session_state:
        st.session_state.pdf_manager.cleanup()
        st.session_state.pdf_manager = PDFManager()

def upload_page():
    initialize_session_state()
    if "analyzed_files" not in st.session_state:
        st.session_state.analyzed_files = []
    if "processing_state" not in st.session_state:
        st.session_state.processing_state = False

    # Custom CSS for the enhanced upload page
    st.markdown("""
        <style>
        .upload-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        .file-list {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .file-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .file-item {
            display: flex;
            align-items: center;
            padding: 0.5rem;
            background: white;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .upload-header {
            background: linear-gradient(135deg, #7C3AED 0%, #581C87 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .analysis-button {
            background: #7C3AED;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .analysis-button:hover {
            background: #581C87;
            transform: translateY(-2px);
        }
        .reset-button {
            background: #DC2626;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .reset-button:hover {
            background: #B91C1C;
        }
        .success-message {
            background: #F0FDF4;
            border-left: 4px solid #22C55E;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        .analysis-results {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class='upload-header'>
            <h1 style='font-size: 2.5rem; font-weight: 300; margin-bottom: 1rem;'>Analisi Dei Documenti</h1>
            <p style='font-size: 1.1rem; opacity: 0.9;'>Carica i tuoi PDF per iniziare l'analisi</p>
        </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "📂 Seleziona o trascina qui i tuoi file PDF",
        accept_multiple_files=True,
        type=['pdf'],
        key="pdf_uploader"
    )

    current_files = {f.name for f in uploaded_files} if uploaded_files else set()
    analyzed_files = set(st.session_state.analyzed_files)
    new_files_exist = bool(current_files - analyzed_files)

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Analysis button centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🚀 Avvia Analisi",
                use_container_width=True,
                type="primary"
            )

    st.markdown("</div>", unsafe_allow_html=True)
    if uploaded_files and analyze_button and (new_files_exist or not st.session_state.files_processed):
        with st.spinner("📊 Analisi dei documenti in corso..."):
            processor = DocumentProcessor()
            all_chunks = []
            all_analyses = {}
            
            # Preserve existing analyses if any
            if hasattr(st.session_state, 'analyses'):
                all_analyses.update(st.session_state.analyses)
            
            progress_bar = st.progress(0)
            
            # Only process new files
            files_to_process = [pdf for pdf in uploaded_files if pdf.name not in analyzed_files]
            
            for i, pdf in enumerate(files_to_process):
                temp_path = st.session_state.pdf_manager.save_pdf(pdf, len(analyzed_files) + i)
                texts, analysis = processor.process_and_analyze(pdf, len(analyzed_files) + i)
                chunks = processor.create_chunks(texts)
                
                all_chunks.extend(chunks)
                all_analyses[pdf.name] = analysis
                
                progress_bar.progress((i + 1) / len(files_to_process))
            
            # Update or create vectorstore
            if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore is not None:
                # Add new documents to existing vectorstore
                vectorstore = VectorStoreManager().create_vectorstore(all_chunks)
                st.session_state.vectorstore.merge_from(vectorstore)
            else:
                # Create new vectorstore
                st.session_state.vectorstore = VectorStoreManager().create_vectorstore(all_chunks)
            
            st.session_state.files_processed = True
            st.session_state.analyses = all_analyses
            st.session_state.analyzed_files.extend([f.name for f in files_to_process])
            
            st.markdown("""
                <div class='success-message'>
                    <h4>✅ Analisi completata con successo!</h4>
                    <p>Tutti i documenti sono stati elaborati e sono pronti per la consultazione.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.rerun()

    # Display analysis results and file list
    if st.session_state.files_processed and hasattr(st.session_state, 'analyses'):
        # File list with reset button
        st.markdown("""
            <div class='file-list'>
                <div class='file-header'>
                    <h3>📚 File Analizzati</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for file list and reset button
        file_col, reset_col = st.columns([4, 1])
        
        with file_col:
            # Mostra solo i file che sono stati effettivamente analizzati
            for file_name in st.session_state.analyses.keys():
                st.markdown(f"""
                    <div class='file-item'>
                        <span>📄 {file_name}</span>
                    </div>
                """, unsafe_allow_html=True)
        
        with reset_col:
            if st.button("🔄 Reset", type="secondary"):
                reset_session()
                st.rerun()

        # Analysis Results
        st.markdown("<div class='analysis-results'>", unsafe_allow_html=True)
        st.markdown("## 📊 Risultati dell'analisi")
        tabs = st.tabs([f"📄 {filename}" for filename in st.session_state.analyses.keys()])
        
        for tab, (filename, analysis) in zip(tabs, st.session_state.analyses.items()):
            with tab:
                display_document_statistics(analysis, filename)
        st.markdown("</div>", unsafe_allow_html=True)

    # Process files if analysis button was clicked
    if uploaded_files and not st.session_state.files_processed and 'analyze_button' in locals() and analyze_button:
        with st.spinner("📊 Analisi dei documenti in corso..."):
            processor = DocumentProcessor()
            all_chunks = []
            all_analyses = {}
            
            progress_bar = st.progress(0)
            
            for i, pdf in enumerate(st.session_state.uploaded_files):
                temp_path = st.session_state.pdf_manager.save_pdf(pdf, i)
                texts, analysis = processor.process_and_analyze(pdf, i)
                chunks = processor.create_chunks(texts)
                
                all_chunks.extend(chunks)
                all_analyses[pdf.name] = analysis
                
                progress_bar.progress((i + 1) / len(st.session_state.uploaded_files))
            
            # Create vectorstore
            vectorstore = VectorStoreManager().create_vectorstore(all_chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.files_processed = True
            st.session_state.analyses = all_analyses
            
            # Aggiorna la lista dei file analizzati
            st.session_state.analyzed_files = list(all_analyses.keys())
            
            st.markdown("""
                <div class='success-message'>
                    <h4>✅ Analisi completata con successo!</h4>
                    <p>Tutti i documenti sono stati elaborati e sono pronti per la consultazione.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.rerun()
                            
def chat_page():
    initialize_session_state()
    
    # Check if files are processed before allowing chat
    if not st.session_state.files_processed:
        st.warning("⚠️ Per favore, carica e analizza prima i documenti nella sezione 'Carica File'")
        return
    # Apply custom CSS
    st.markdown('''
    <style>
    /* Remove the default padding that was for Streamlit avatars */
    .stChatMessage {
        padding: 0 !important;
        margin-bottom: 1rem !important;
    }

    /* Global Background */
    body, .main, .stApp {
        background-color: #f4f4f9 !important;
        color: #3a007d !important;
        font-family: 'Georgia', serif;
    }

    /* Custom Chat Message Container */
    .chat-message {
        padding: 1.5rem; 
        border-radius: 1rem; 
        margin-bottom: 1rem; 
        display: flex;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stChatMessage.user .chat-message {
        background-color: #e0d4fc !important;
        color: #3a007d !important;
    }

    .stChatMessage.assistant .chat-message {
        background-color: #fff !important;
        color: #3a007d !important;
    }

    /* Avatar Styling */
    .chat-message .avatar {
        width: 15%;
    }
                
    /* Resize chat container to be larger */
    .stContainer {
        height: 700px !important;  /* Increase the height of the chat container */
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }

    /* Message Text */
    .chat-message .message {
        width: 85%;
        padding: 0 1.5rem;
        word-wrap: break-word;
        font-size: 1rem;
    }

    /* Button Styling */
    .stButton>button {
        background-color: white !important;
        color: #5c2af5 !important;
        border: 2px solid #5c2af5 !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-size: 1rem !important;
        cursor: pointer;
    }
    /* Remove default left padding from chat container */
    .element-container:has(.stChatMessage) {
        padding-left: 0 !important;
    }
    </style>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Chatta con i tuoi documenti")
        
        # Initialize conversation and chat history if they don't exist
        if "conversation" not in st.session_state:
            llm = ChatOpenAI(temperature=0.3, model="gpt-4-turbo-preview")
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer',
                input_key='question'
            )
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 7, "fetch_k": 14, "lambda_mult": 0.7}
            )
            st.session_state.conversation = EnhancedConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                verbose=True
            )
            # Add welcome message when conversation is initialized
            st.session_state.chat_history = [
                AIMessage(content="Ciao! Sono qui per aiutarti ad analizzare i tuoi documenti. Come posso esserti utile?")
            ]
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Ciao! Sono qui per aiutarti ad analizzare i tuoi documenti. Come posso esserti utile?")
            ]
        
        # Chat container with scrollbar
        chat_container = st.container(height=750)
        
        # Your existing templates
        bot_template = '''
        <div class="chat-message">
            <div class="avatar">
                <img src="https://i.ibb.co/dPQxgfx/robo.png" alt="Bot">
            </div>
            <div class="message">{}</div>
        </div>
        '''

        user_template = '''
        <div class="chat-message">
            <div class="avatar">
                <img src="https://i.ibb.co/NW2NRnb/dxc-logo.png" alt="User">
            </div>    
            <div class="message">{}</div>
        </div>
        '''
        
        # Display chat messages
        for message in st.session_state.chat_history:
            with chat_container:
                if isinstance(message, HumanMessage):
                    message_placeholder = st.chat_message("user", avatar="https://i.ibb.co/GtMvMfP/Immagine-2025-01-08-113111.png")
                    message_placeholder.markdown(user_template.format(message.content), unsafe_allow_html=True)
                else:
                    message_placeholder = st.chat_message("assistant", avatar="https://i.ibb.co/GtMvMfP/Immagine-2025-01-08-113111.png")
                    message_placeholder.markdown(bot_template.format(message.content), unsafe_allow_html=True)
        
        # Input box
        if query := st.chat_input("Fai una domanda sui documenti:"):
            # Immediately display the user's question
            with chat_container:
                message_placeholder = st.chat_message("user", avatar="https://i.ibb.co/GtMvMfP/Immagine-2025-01-08-113111.png")
                message_placeholder.markdown(user_template.format(query), unsafe_allow_html=True)
            
            st.session_state.chat_history.append(HumanMessage(content=query))
            
            with st.spinner("Elaborazione della risposta..."):
                response = st.session_state.conversation({"question": query})
                answer = response["answer"]
                source_docs = response["source_documents"]
                
                st.session_state.chat_history.append(AIMessage(content=answer))
                st.session_state.current_sources = source_docs
                
                # Display the bot's response immediately
                with chat_container:
                    message_placeholder = st.chat_message("assistant", avatar="https://i.ibb.co/GtMvMfP/Immagine-2025-01-08-113111.png")
                    message_placeholder.markdown(bot_template.format(answer), unsafe_allow_html=True)
    
    with col2:
        if "current_sources" in st.session_state and st.session_state.current_sources:
            st.markdown("### 📑 Documenti di Supporto")
            
            docs_by_pdf = {}
            for doc in st.session_state.current_sources:
                pdf_index = doc.metadata['pdf_info']['pdf_index']
                relevance_score = doc.metadata.get('relevance_score', 0)
                
                if relevance_score >= 0.65:  # Mostra solo documenti veramente rilevanti
                    if pdf_index not in docs_by_pdf:
                        docs_by_pdf[pdf_index] = {
                            'docs': [],
                            'max_score': 0,
                            'relevant_pages': set(),
                            'relevant_sections': []  # Aggiungiamo le sezioni rilevanti
                        }
                    
                    docs_by_pdf[pdf_index]['docs'].append(doc)
                    docs_by_pdf[pdf_index]['max_score'] = max(
                        docs_by_pdf[pdf_index]['max_score'],
                        relevance_score
                    )
                    docs_by_pdf[pdf_index]['relevant_pages'].add(doc.metadata['page'])
                    docs_by_pdf[pdf_index]['relevant_sections'].append(doc.page_content)
            
            # Initialize PDF highlighter
            highlighter = PDFHighlighter()
            
            # Display PDFs with highlights
            for pdf_index, pdf_data in docs_by_pdf.items():
                if pdf_data['max_score'] >= 0.6:  # Relevance threshold
                    pdf_info = pdf_data['docs'][0].metadata['pdf_info']
                    with st.expander(
                        f"📄 {pdf_info['filename']} (Relevance: {pdf_data['max_score']:.2f})", 
                        expanded=True
                    ):
                        try:
                            # Create highlighted version of the PDF
                            pdf_path = st.session_state.pdf_manager.get_pdf_path(pdf_index)
                            if pdf_path:
                                highlighted_pdf = highlighter.highlight_pdf(
                                    pdf_path, 
                                    pdf_data['docs']
                                )
                                
                                # Display highlighted PDF in a scrollable container
                                pdf_container = st.container(height=750)
                                with pdf_container:
                                    pdf_viewer(
                                        highlighted_pdf,
                                        pages_to_render=sorted(pdf_data['relevant_pages']),
                                        width=600,
                                        key=f"viewer_{pdf_index}_{'-'.join(map(str, pdf_data['relevant_pages']))}"
                                    )
                                # Display relevance information
                                st.info(f"""
                                📌 Pagine rilevanti: {', '.join(map(str, sorted(pdf_data['relevant_pages'])))}
                                📊 Punteggio di rilevanza: {pdf_data['max_score']:.2f}
                                """)
                        except Exception as e:
                            st.error(f"Errore durante l'evidenziazione del PDF: {str(e)}")
                            # Fallback to normal PDF display
                            pdf_viewer(
                                pdf_path,
                                pages_to_render=sorted(pdf_data['relevant_pages']),
                                width=600,
                                key=f"viewer_{pdf_index}_{'-'.join(map(str, pdf_data['relevant_pages']))}"
                            )

def create_sidebar():
    with st.sidebar:
        # Menu di navigazione con stili personalizzati
        selected = option_menu(
            "Menu",
            ["Home", "Carica File", "Chat"],
            icons=["house", "folder", "chat"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "1rem",
                    "background-color": "rgb(245, 243, 255)"  # Sfondo sidebar coerente
                },
                "icon": {
                    "color": "#581C87",  # Viola scuro per le icone
                    "font-size": "1.2rem"
                },
                "nav-link": {
                    "color": "#581C87",  # Viola scuro per i link non selezionati
                    "font-size": "1rem",
                    "text-align": "left",
                    "padding": "0.75rem 1rem",
                    "border-radius": "0.5rem",
                    "margin": "0.25rem 0",
                    "transition": "background-color 0.3s, color 0.3s",  # Effetto hover
                },
                "nav-link-hover": {
                    "background-color": "rgb(230, 223, 250)",  # Sfondo hover (viola chiaro trasparente)
                    "color": "#581C87"  # Testo viola scuro in hover
                },
                "nav-link-selected": {
                    "background-color": "#7C3AED",  # Viola chiaro per link selezionato
                    "color": "white"  # Testo bianco per il link selezionato
                }
            }
        )

        # Separatore e informazioni aggiuntive
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "Questa app ti consente di analizzare documenti PDF e chattare con i loro contenuti "
            "utilizzando l'intelligenza artificiale."
        )
        
        return selected

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Delia", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_css()
    initialize_session_state()
    # Create sidebar and get selected page
    selected = create_sidebar()


   # Route to appropriate page

    if selected == "Home":
        home_page()
    elif selected == "Carica File":
        upload_page()
    elif selected == "Chat":
        chat_page()
if __name__ == "__main__":
    main()
