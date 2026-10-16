"""
Baseline Retrieval Methods for Cross-Lingual Retrieval System

Implements two baselines:
1. BM25 (keyword-based, monolingual)
2. Translate-Test (translate query, then search)

"""

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from deep_translator import GoogleTranslator
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
import time


class BaselineRetriever:
    """
    Implements baseline retrieval methods for comparison.
    """
    
    def __init__(self, data_dir='data/processed'):
        """
        Initialize baseline retriever.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.documents = None
        self.bm25_index = {}  # Separate BM25 index per language
        self.translators = {}
        
        # Initialize translators
        self._init_translators()
    
    def _init_translators(self):
        """Initialize Google Translators for all language pairs."""
        print("Initializing translators...")
        
        # We need to translate FROM any query language TO any document language
        lang_pairs = [
            ('en', 'es'), ('en', 'fr'),  # English to Spanish/French
            ('es', 'en'), ('es', 'fr'),  # Spanish to English/French
            ('fr', 'en'), ('fr', 'es'),  # French to English/Spanish
        ]
        
        for source, target in lang_pairs:
            key = f'{source}_{target}'
            self.translators[key] = GoogleTranslator(source=source, target=target)
        
        print(f"✓ Initialized {len(self.translators)} translators")
    
    def load_documents(self):
        """Load all documents from processed data."""
        print("\nLoading documents...")
        
        # Load combined dataset (all languages)
        train_file = self.data_dir / 'all_train.csv'
        
        if not train_file.exists():
            raise FileNotFoundError(f"Processed data not found: {train_file}")
        
        self.documents = pd.read_csv(train_file)
        print(f"✓ Loaded {len(self.documents):,} documents")
        
        # Show language distribution
        lang_counts = self.documents['language'].value_counts()
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count:,} documents")
    
    def build_bm25_index(self):
        """Build BM25 index for each language separately."""
        print("\nBuilding BM25 indexes...")
        
        for lang in ['en', 'es', 'fr']:
            # Get documents for this language
            lang_docs = self.documents[self.documents['language'] == lang]
            
            # Tokenize documents (simple word splitting)
            tokenized_docs = [
                doc.lower().split() 
                for doc in tqdm(lang_docs['content'], desc=f"Tokenizing {lang}")
            ]
            
            # Build BM25 index
            self.bm25_index[lang] = BM25Okapi(tokenized_docs)
            
            # Store document IDs for this language
            self.bm25_index[f'{lang}_ids'] = lang_docs['doc_id'].tolist()
            
            print(f"✓ Built BM25 index for {lang}: {len(tokenized_docs):,} documents")
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text  # No translation needed
        
        translator_key = f'{source_lang}_{target_lang}'
        
        try:
            translated = self.translators[translator_key].translate(text)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails
    
    def search_bm25(self, query: str, query_lang: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25 (monolingual - only searches in query language).
        
        Args:
            query: Search query
            query_lang: Language of the query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index[query_lang].get_scores(tokenized_query)
        
        # Get top-k document IDs and scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        doc_ids = self.bm25_index[f'{query_lang}_ids']
        
        results = [(doc_ids[i], scores[i]) for i in top_indices]
        
        return results
    
    def search_translate_test(self, query: str, query_lang: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using Translate-Test approach.
        Translates query to all languages, searches in each, combines results.
        
        Args:
            query: Search query
            query_lang: Language of the query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        all_results = []
        
        # Search in each language
        for target_lang in ['en', 'es', 'fr']:
            # Translate query if needed
            if query_lang != target_lang:
                translated_query = self.translate_text(query, query_lang, target_lang)
                time.sleep(0.1)  # Small delay to avoid rate limiting
            else:
                translated_query = query
            
            # Search with BM25 in target language
            results = self.search_bm25(translated_query, target_lang, top_k=top_k)
            all_results.extend(results)
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results[:top_k]
    
    def evaluate(self, queries_df: pd.DataFrame, method: str = 'bm25', top_k: int = 10):
        """
        Evaluate retrieval method on test queries.
        
        Args:
            queries_df: DataFrame with test queries
            method: 'bm25' or 'translate'
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {method.upper()} method...")
        print(f"Test queries: {len(queries_df):,}")
        
        results = []
        
        for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Evaluating"):
            query = row['query_text']
            query_lang = row['query_language']
            relevant_doc = row['relevant_doc_id']
            
            # Search
            if method == 'bm25':
                retrieved = self.search_bm25(query, query_lang, top_k=top_k)
            elif method == 'translate':
                retrieved = self.search_translate_test(query, query_lang, top_k=top_k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Check if relevant document was retrieved
            retrieved_ids = [doc_id for doc_id, score in retrieved]
            
            # Calculate metrics
            if relevant_doc in retrieved_ids:
                rank = retrieved_ids.index(relevant_doc) + 1
                reciprocal_rank = 1.0 / rank
                recall = 1.0
            else:
                reciprocal_rank = 0.0
                recall = 0.0
            
            results.append({
                'query_id': row['query_id'],
                'query': query,
                'query_lang': query_lang,
                'relevant_doc': relevant_doc,
                'found': relevant_doc in retrieved_ids,
                'reciprocal_rank': reciprocal_rank,
                'recall': recall
            })
        
        # Calculate aggregate metrics
        mrr = np.mean([r['reciprocal_rank'] for r in results])
        recall_at_k = np.mean([r['recall'] for r in results])
        
        # Per-language breakdown
        results_df = pd.DataFrame(results)
        per_lang_mrr = results_df.groupby('query_lang')['reciprocal_rank'].mean()
        
        metrics = {
            'method': method,
            'top_k': top_k,
            'total_queries': len(queries_df),
            'mrr': mrr,
            f'recall@{top_k}': recall_at_k,
            'per_language_mrr': per_lang_mrr.to_dict(),
            'results': results
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics."""
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - {metrics['method'].upper()}")
        print("="*60)
        print(f"Total queries: {metrics['total_queries']:,}")
        print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
        print(f"Recall@{metrics['top_k']}: {metrics['recall@' + str(metrics['top_k'])]:.4f}")
        
        print("\nPer-language MRR:")
        for lang, mrr in metrics['per_language_mrr'].items():
            print(f"  {lang}: {mrr:.4f}")
        
        print("="*60 + "\n")
    
    def save_results(self, metrics: Dict, output_dir: str = 'results/baseline'):
        """Save evaluation results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{metrics['method']}_results.json"
        
        # Convert to serializable format
        save_data = {
            'method': metrics['method'],
            'top_k': metrics['top_k'],
            'total_queries': metrics['total_queries'],
            'mrr': float(metrics['mrr']),
            f"recall@{metrics['top_k']}": float(metrics[f"recall@{metrics['top_k']}"]),
            'per_language_mrr': {k: float(v) for k, v in metrics['per_language_mrr'].items()},
            'results': metrics['results']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Baseline retrieval methods for cross-lingual search'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['bm25', 'translate', 'both'],
        default='both',
        help='Retrieval method to evaluate (default: both)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with processed data (default: data/processed)'
    )
    parser.add_argument(
        '--test-queries',
        type=int,
        default=100,
        help='Number of test queries to use (default: 100)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = BaselineRetriever(data_dir=args.data_dir)
    
    # Load documents and build index
    retriever.load_documents()
    retriever.build_bm25_index()
    
    # Load test queries
    queries_file = Path(args.data_dir) / 'all_queries.csv'
    queries_df = pd.read_csv(queries_file)
    
    # Sample queries if specified
    if args.test_queries < len(queries_df):
        queries_df = queries_df.sample(n=args.test_queries, random_state=42)
    
    print(f"\nUsing {len(queries_df):,} test queries")
    
    # Evaluate methods
    methods_to_eval = ['bm25', 'translate'] if args.method == 'both' else [args.method]
    
    for method in methods_to_eval:
        metrics = retriever.evaluate(queries_df, method=method, top_k=args.top_k)
        retriever.print_metrics(metrics)
        retriever.save_results(metrics)
    
    print("\n✅ Baseline evaluation complete!")
    print("📁 Results saved in: results/baseline/\n")


if __name__ == '__main__':
    main()
