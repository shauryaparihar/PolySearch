"""
Streamlit Web Application for Cross-Lingual Retrieval Demo

Interactive interface for searching documents across multiple languages.

Author: Shaurya Parihar
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dense_retrieval import DenseRetriever


# Page config
st.set_page_config(
    page_title="Cross-Lingual Document Retrieval",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .lang-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .lang-en { background-color: #e3f2fd; color: #1976d2; }
    .lang-es { background-color: #fff3e0; color: #ef6c00; }
    .lang-fr { background-color: #f3e5f5; color: #7b1fa2; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever(model_name):
    """Load and cache the retriever model."""
    retriever = DenseRetriever(model_name=model_name)
    retriever.load_model()
    retriever.load_documents()
    retriever.encode_documents(batch_size=32, use_cache=True)
    retriever.build_index(index_type='flat')
    return retriever


def get_lang_color(lang):
    """Get color class for language badge."""
    colors = {
        'en': 'lang-en',
        'es': 'lang-es',
        'fr': 'lang-fr'
    }
    return colors.get(lang, 'lang-en')


def get_lang_name(lang):
    """Get full language name."""
    names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French'
    }
    return names.get(lang, lang)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Cross-Lingual Document Retrieval</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Search across English, Spanish, and French documents seamlessly</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        [
            "sentence-transformers/LaBSE",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        ],
        help="Choose the multilingual embedding model"
    )
    
    num_results = st.sidebar.slider(
        "Number of Results",
        min_value=5,
        max_value=20,
        value=10,
        help="How many results to display"
    )
    
    # Load retriever
    with st.spinner(f"Loading model: {model_choice.split('/')[-1]}..."):
        try:
            retriever = load_retriever(model_choice)
            st.sidebar.success("✓ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
    # Main content
    st.markdown("---")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your search query (in any language)",
            placeholder="e.g., climate change solutions, cambio climático, technologies renouvelables",
            help="Type your query in English, Spanish, or French"
        )
    
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown("""
        Try these example queries:
        - **English:** "artificial intelligence", "renewable energy", "global warming"
        - **Spanish:** "inteligencia artificial", "energía renovable", "calentamiento global"
        - **French:** "intelligence artificielle", "énergie renouvelable", "réchauffement climatique"
        """)
    
    # Search results
    if query and (search_button or query):
        st.markdown("---")
        st.markdown("### 🎯 Search Results")
        
        with st.spinner("Searching across all languages..."):
            try:
                results = retriever.search(query, top_k=num_results)
                
                if results:
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    
                    lang_counts = {}
                    for _, _, lang, _ in results:
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    
                    with col1:
                        st.metric("Total Results", len(results))
                    with col2:
                        st.metric("Languages Found", len(lang_counts))
                    with col3:
                        avg_score = sum(score for _, _, _, score in results) / len(results)
                        st.metric("Avg Similarity", f"{avg_score:.3f}")
                    
                    st.markdown("---")
                    
                    # Display results
                    for i, (doc_id, title, lang, score) in enumerate(results, 1):
                        # Get document details
                        doc = retriever.documents[retriever.documents['doc_id'] == doc_id].iloc[0]
                        
                        # Result box
                        st.markdown(f"""
                        <div class="result-box">
                            <h4 style="margin:0;">
                                {i}. {title}
                            </h4>
                            <p style="margin:0.5rem 0;">
                                <span class="lang-badge {get_lang_color(lang)}">{get_lang_name(lang)}</span>
                                <span style="color: #666;">Similarity: {score:.4f}</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable summary
                        with st.expander(f"📄 View Summary"):
                            st.write(doc['summary'])
                            st.markdown(f"**URL:** [{doc['url']}]({doc['url']})")
                            st.markdown(f"**Categories:** {doc['categories']}")
                    
                else:
                    st.warning("No results found. Try a different query.")
                
            except Exception as e:
                st.error(f"Search error: {e}")
    
    # Sidebar - About
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Stats")
    
    if retriever:
        total_docs = len(retriever.documents)
        lang_counts = retriever.documents['language'].value_counts()
        
        st.sidebar.markdown(f"**Total Documents:** {total_docs:,}")
        st.sidebar.markdown("**Language Distribution:**")
        for lang, count in lang_counts.items():
            percentage = (count / total_docs) * 100
            st.sidebar.markdown(f"- {get_lang_name(lang)}: {count:,} ({percentage:.1f}%)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 Technology")
    st.sidebar.markdown("""
    - **Embeddings:** Multilingual Sentence Transformers
    - **Search:** FAISS Vector Similarity
    - **Framework:** Streamlit
    - **Languages:** EN, ES, FR
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Cross-Lingual Dense Retrieval System</p>
        <p>© 2026 Shaurya Parihar | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
