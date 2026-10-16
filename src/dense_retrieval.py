"""
Dense Retrieval Implementation for Cross-Lingual Retrieval System

Uses multilingual sentence embeddings (LaBSE, mBERT, XLM-R) for
semantic search across languages.

"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import pickle
from typing import List, Tuple, Dict
import time


class DenseRetriever:
    """
    Implements dense retrieval using multilingual sentence embeddings.
    """
    
    def __init__(self, model_name='sentence-transformers/LaBSE', data_dir='data/processed'):
        """
        Initialize dense retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            data_dir: Directory containing processed data
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.embeddings_dir = Path('data/embeddings')
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.documents = None
        self.doc_embeddings = None
        self.index = None
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the multilingual sentence transformer model."""
        print(f"\nLoading model: {self.model_name}")
        print("(This may take a few minutes on first run...)")
        
        self.model = SentenceTransformer(self.model_name)
        self.model = self.model.to(self.device)
        
        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def load_documents(self):
        """Load all documents."""
        print("\nLoading documents...")
        
        train_file = self.data_dir / 'all_train.csv'
        
        if not train_file.exists():
            raise FileNotFoundError(f"Data not found: {train_file}")
        
        self.documents = pd.read_csv(train_file)
        print(f"✓ Loaded {len(self.documents):,} documents")
        
        # Show distribution
        lang_counts = self.documents['language'].value_counts()
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count:,} documents")
    
    def encode_documents(self, batch_size=32, use_cache=True):
        """
        Encode all documents into dense vectors.
        
        Args:
            batch_size: Batch size for encoding
            use_cache: Whether to use cached embeddings if available
        """
        cache_file = self.embeddings_dir / f'doc_embeddings_{self.model_name.replace("/", "_")}.pkl'
        
        # Check cache
        if use_cache and cache_file.exists():
            print(f"\n✓ Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
            print(f"  Loaded {len(self.doc_embeddings)} embeddings")
            return
        
        print(f"\nEncoding {len(self.documents):,} documents...")
        print(f"Batch size: {batch_size}")
        print("⏰ This will take some time. Good moment for a coffee break!")
        
        # Get document texts (we'll encode title + content)
        # Combining title and content gives better retrieval
        doc_texts = [
            f"{row['title']}. {row['summary']}"  # Title + summary for efficiency
            for _, row in self.documents.iterrows()
        ]
        
        # Encode in batches with progress bar
        start_time = time.time()
        self.doc_embeddings = self.model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device=self.device
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Encoding complete!")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Shape: {self.doc_embeddings.shape}")
        print(f"  Docs/second: {len(self.documents)/elapsed:.1f}")
        
        # Save to cache
        print(f"\nSaving embeddings to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.doc_embeddings, f)
        print(f"✓ Saved to {cache_file}")
    
    def build_index(self, index_type='flat'):
        """
        Build FAISS index for fast similarity search.
        
        Args:
            index_type: Type of FAISS index ('flat' or 'ivf')
        """
        print(f"\nBuilding FAISS index ({index_type})...")
        
        if self.doc_embeddings is None:
            raise ValueError("Embeddings not loaded. Run encode_documents first.")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        
        dimension = self.doc_embeddings.shape[1]
        
        if index_type == 'flat':
            # Flat index: exact search (slower but accurate)
            self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine sim)
        elif index_type == 'ivf':
            # IVF index: approximate search (faster for large datasets)
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(self.doc_embeddings)
            self.index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        self.index.add(self.doc_embeddings)
        
        print(f"✓ Index built successfully")
        print(f"  Total vectors: {self.index.ntotal:,}")
        print(f"  Index type: {type(self.index).__name__}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, str, float]]:
        """
        Search for similar documents given a query.
        
        Args:
            query: Search query (in any language)
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, title, language, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Run build_index first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            device=self.device
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get document details
        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc = self.documents.iloc[idx]
            results.append((
                doc['doc_id'],
                doc['title'],
                doc['language'],
                float(score)
            ))
        
        return results
    
    def evaluate(self, queries_df: pd.DataFrame, top_k: int = 10):
        """
        Evaluate dense retrieval on test queries.
        
        Args:
            queries_df: DataFrame with test queries
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating Dense Retrieval...")
        print(f"Model: {self.model_name}")
        print(f"Test queries: {len(queries_df):,}")
        
        results = []
        
        for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Evaluating"):
            query = row['query_text']
            query_lang = row['query_language']
            relevant_doc = row['relevant_doc_id']
            
            # Search
            retrieved = self.search(query, top_k=top_k)
            
            # Check if relevant document was retrieved
            retrieved_ids = [doc_id for doc_id, _, _, _ in retrieved]
            
            # Calculate metrics
            if relevant_doc in retrieved_ids:
                rank = retrieved_ids.index(relevant_doc) + 1
                reciprocal_rank = 1.0 / rank
                recall = 1.0
                
                # Calculate DCG and nDCG
                dcg = 1.0 / np.log2(rank + 1)
                idcg = 1.0 / np.log2(2)  # Perfect ranking
                ndcg = dcg / idcg
            else:
                reciprocal_rank = 0.0
                recall = 0.0
                ndcg = 0.0
            
            results.append({
                'query_id': row['query_id'],
                'query': query,
                'query_lang': query_lang,
                'relevant_doc': relevant_doc,
                'found': relevant_doc in retrieved_ids,
                'reciprocal_rank': reciprocal_rank,
                'recall': recall,
                'ndcg': ndcg
            })
        
        # Calculate aggregate metrics
        mrr = np.mean([r['reciprocal_rank'] for r in results])
        recall_at_k = np.mean([r['recall'] for r in results])
        ndcg_at_k = np.mean([r['ndcg'] for r in results])
        
        # Per-language breakdown
        results_df = pd.DataFrame(results)
        per_lang_mrr = results_df.groupby('query_lang')['reciprocal_rank'].mean()
        per_lang_recall = results_df.groupby('query_lang')['recall'].mean()
        
        metrics = {
            'method': 'dense_retrieval',
            'model': self.model_name,
            'top_k': top_k,
            'total_queries': len(queries_df),
            'mrr': mrr,
            f'recall@{top_k}': recall_at_k,
            f'ndcg@{top_k}': ndcg_at_k,
            'per_language_mrr': per_lang_mrr.to_dict(),
            'per_language_recall': per_lang_recall.to_dict(),
            'results': results
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics."""
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - DENSE RETRIEVAL")
        print("="*60)
        print(f"Model: {metrics['model']}")
        print(f"Total queries: {metrics['total_queries']:,}")
        print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
        print(f"Recall@{metrics['top_k']}: {metrics['recall@' + str(metrics['top_k'])]:.4f}")
        print(f"nDCG@{metrics['top_k']}: {metrics['ndcg@' + str(metrics['top_k'])]:.4f}")
        
        print("\nPer-language MRR:")
        for lang, mrr in metrics['per_language_mrr'].items():
            print(f"  {lang}: {mrr:.4f}")
        
        print("\nPer-language Recall:")
        for lang, recall in metrics['per_language_recall'].items():
            print(f"  {lang}: {recall:.4f}")
        
        print("="*60 + "\n")
    
    def save_results(self, metrics: Dict, output_dir: str = 'results/dense'):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_safe_name = self.model_name.replace('/', '_')
        output_file = output_dir / f"{model_safe_name}_results.json"
        
        # Convert to serializable format
        save_data = {
            'method': metrics['method'],
            'model': metrics['model'],
            'top_k': metrics['top_k'],
            'total_queries': metrics['total_queries'],
            'mrr': float(metrics['mrr']),
            f"recall@{metrics['top_k']}": float(metrics[f"recall@{metrics['top_k']}"]),
            f"ndcg@{metrics['top_k']}": float(metrics[f"ndcg@{metrics['top_k']}"]),
            'per_language_mrr': {k: float(v) for k, v in metrics['per_language_mrr'].items()},
            'per_language_recall': {k: float(v) for k, v in metrics['per_language_recall'].items()},
            'results': metrics['results']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Dense retrieval for cross-lingual search'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['encode', 'index', 'search', 'evaluate', 'all'],
        default='all',
        help='Operation mode (default: all)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/LaBSE',
        help='Sentence transformer model (default: LaBSE)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Search query (for search mode)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )
    parser.add_argument(
        '--test-queries',
        type=int,
        default=500,
        help='Number of test queries for evaluation (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = DenseRetriever(model_name=args.model)
    
    # Load model and documents
    retriever.load_model()
    retriever.load_documents()
    
    if args.mode in ['encode', 'all']:
        # Encode documents
        retriever.encode_documents(batch_size=args.batch_size)
    
    if args.mode in ['index', 'search', 'evaluate', 'all']:
        # Load embeddings if not already encoded
        if retriever.doc_embeddings is None:
            retriever.encode_documents(batch_size=args.batch_size, use_cache=True)
        
        # Build index
        retriever.build_index(index_type='flat')
    
    if args.mode == 'search':
        # Interactive search
        if not args.query:
            print("\nEnter search query (or 'quit' to exit):")
            while True:
                query = input(">> ")
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                results = retriever.search(query, top_k=args.top_k)
                
                print(f"\nTop {args.top_k} results:")
                for i, (doc_id, title, lang, score) in enumerate(results, 1):
                    print(f"{i}. [{lang}] {title}")
                    print(f"   Score: {score:.4f} | ID: {doc_id}")
                print()
        else:
            # Single query
            results = retriever.search(args.query, top_k=args.top_k)
            
            print(f"\nQuery: {args.query}")
            print(f"\nTop {args.top_k} results:")
            for i, (doc_id, title, lang, score) in enumerate(results, 1):
                print(f"{i}. [{lang}] {title}")
                print(f"   Score: {score:.4f} | ID: {doc_id}")
    
    if args.mode in ['evaluate', 'all']:
        # Load test queries
        queries_file = Path('data/processed') / 'all_queries.csv'
        queries_df = pd.read_csv(queries_file)
        
        # Sample if specified
        if args.test_queries < len(queries_df):
            queries_df = queries_df.sample(n=args.test_queries, random_state=42)
        
        # Evaluate
        metrics = retriever.evaluate(queries_df, top_k=args.top_k)
        retriever.print_metrics(metrics)
        retriever.save_results(metrics)
    
    print("\n✅ Dense retrieval complete!\n")


if __name__ == '__main__':
    main()
