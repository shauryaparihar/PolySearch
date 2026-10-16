"""
Data Preprocessing Script for Cross-Lingual Retrieval System

This script cleans and prepares Wikipedia articles for retrieval.
- Removes noise and special characters
- Creates train/test splits
- Generates query-document pairs for evaluation

"""

import json
import pandas as pd
import re
import argparse
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict, Tuple


class DataPreprocessor:
    
    def __init__(self, input_dir='data/raw', output_dir='data/processed'):
        """
        Initialize preprocessor.
        
        Args:
            input_dir: Directory containing raw JSON files
            output_dir: Directory to save processed data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.languages = ['en', 'es', 'fr']
    
    def clean_text(self, text: str) -> str:
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"áéíóúñÁÉÍÓÚÑàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛçÇ]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove extra whitespace
        text = text.strip()
        
        return text
    
    def load_articles(self, language: str) -> List[Dict]:

        input_file = self.input_dir / f'wikipedia_{language}.json'
        
        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        return articles
    
    def process_articles(self, articles: List[Dict]) -> pd.DataFrame:
        
        processed = []
        
        for article in tqdm(articles, desc="Processing articles"):
            # Clean title and content
            clean_title = self.clean_text(article['title'])
            clean_content = self.clean_text(article['content'])
            clean_summary = self.clean_text(article['summary'])
            
            # Filter: Skip if content is too short after cleaning
            if len(clean_content) < 300:
                continue
            
            # Filter: Skip if content is too long (likely list pages)
            if len(clean_content) > 50000:
                continue
            
            processed.append({
                'doc_id': f"{article['language']}_{len(processed)}",
                'title': clean_title,
                'content': clean_content,
                'summary': clean_summary,
                'language': article['language'],
                'url': article['url'],
                'length': len(clean_content),
                'categories': ', '.join(article['categories'][:5])  # Top 5 categories
            })
        
        return pd.DataFrame(processed)
    
    def create_train_test_split(self, df: pd.DataFrame, test_size=0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
       
        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df_shuffled) * (1 - test_size))
        train_df = df_shuffled[:split_idx]
        test_df = df_shuffled[split_idx:]
        
        return train_df, test_df
    
    def generate_test_queries(self, test_df: pd.DataFrame, num_queries=500) -> pd.DataFrame:
       
        queries = []
        
        # Sample articles for queries
        sampled = test_df.sample(n=min(num_queries, len(test_df)), random_state=42)
        
        for idx, row in sampled.iterrows():
            # Create query from title
            query_from_title = {
                'query_id': f"q_title_{len(queries)}",
                'query_text': row['title'],
                'query_language': row['language'],
                'relevant_doc_id': row['doc_id'],
                'query_type': 'title'
            }
            queries.append(query_from_title)
            
            # Create query from summary (first sentence)
            summary_sentences = row['summary'].split('.')
            if len(summary_sentences) > 0:
                first_sentence = summary_sentences[0].strip()
                if len(first_sentence) > 20:  # Only if meaningful
                    query_from_summary = {
                        'query_id': f"q_summary_{len(queries)}",
                        'query_text': first_sentence,
                        'query_language': row['language'],
                        'relevant_doc_id': row['doc_id'],
                        'query_type': 'summary'
                    }
                    queries.append(query_from_summary)
        
        return pd.DataFrame(queries)
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          queries_df: pd.DataFrame, language: str):
        """
        Save processed data to CSV files.
        """
        # Save train and test sets
        train_file = self.output_dir / f'{language}_train.csv'
        test_file = self.output_dir / f'{language}_test.csv'
        queries_file = self.output_dir / f'{language}_queries.csv'
        
        train_df.to_csv(train_file, index=False, encoding='utf-8')
        test_df.to_csv(test_file, index=False, encoding='utf-8')
        queries_df.to_csv(queries_file, index=False, encoding='utf-8')
        
        print(f"\n✓ Saved {language} data:")
        print(f"  Train: {len(train_df):,} documents → {train_file}")
        print(f"  Test: {len(test_df):,} documents → {test_file}")
        print(f"  Queries: {len(queries_df):,} queries → {queries_file}")
    
    def process_all_languages(self):
        """Process articles for all languages."""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        all_stats = {}
        
        for lang in self.languages:
            print(f"\nProcessing {lang.upper()} articles...")
            
            # Load raw articles
            articles = self.load_articles(lang)
            print(f"  Loaded: {len(articles):,} raw articles")
            
            # Process articles
            df = self.process_articles(articles)
            print(f"  After cleaning: {len(df):,} articles")
            
            # Create train/test split
            train_df, test_df = self.create_train_test_split(df, test_size=0.1)
            
            # Generate test queries
            queries_df = self.generate_test_queries(test_df, num_queries=500)
            
            # Save processed data
            self.save_processed_data(train_df, test_df, queries_df, lang)
            
            # Store stats
            all_stats[lang] = {
                'raw': len(articles),
                'processed': len(df),
                'train': len(train_df),
                'test': len(test_df),
                'queries': len(queries_df),
                'avg_length': df['length'].mean()
            }
        
        # Create combined dataset (all languages together)
        self._create_combined_dataset()
        
        # Print summary
        self._print_summary(all_stats)
    
    def _create_combined_dataset(self):
        """Combine all language datasets into single files."""
        print("\n" + "-"*60)
        print("Creating combined dataset (all languages)...")
        
        all_train = []
        all_test = []
        all_queries = []
        
        for lang in self.languages:
            train_df = pd.read_csv(self.output_dir / f'{lang}_train.csv')
            test_df = pd.read_csv(self.output_dir / f'{lang}_test.csv')
            queries_df = pd.read_csv(self.output_dir / f'{lang}_queries.csv')
            
            all_train.append(train_df)
            all_test.append(test_df)
            all_queries.append(queries_df)
        
        # Combine
        combined_train = pd.concat(all_train, ignore_index=True)
        combined_test = pd.concat(all_test, ignore_index=True)
        combined_queries = pd.concat(all_queries, ignore_index=True)
        
        # Save
        combined_train.to_csv(self.output_dir / 'all_train.csv', index=False, encoding='utf-8')
        combined_test.to_csv(self.output_dir / 'all_test.csv', index=False, encoding='utf-8')
        combined_queries.to_csv(self.output_dir / 'all_queries.csv', index=False, encoding='utf-8')
        
        print(f"✓ Combined train: {len(combined_train):,} documents")
        print(f"✓ Combined test: {len(combined_test):,} documents")
        print(f"✓ Combined queries: {len(combined_queries):,} queries")
    
    def _print_summary(self, stats: Dict):
        """Print preprocessing summary."""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        
        for lang, stat in stats.items():
            print(f"\n{lang.upper()}:")
            print(f"  Raw articles: {stat['raw']:,}")
            print(f"  After cleaning: {stat['processed']:,}")
            print(f"  Train set: {stat['train']:,}")
            print(f"  Test set: {stat['test']:,}")
            print(f"  Test queries: {stat['queries']:,}")
            print(f"  Avg length: {stat['avg_length']:,.0f} chars")
        
        total_train = sum(s['train'] for s in stats.values())
        total_test = sum(s['test'] for s in stats.values())
        total_queries = sum(s['queries'] for s in stats.values())
        
        print(f"\n{'='*60}")
        print(f"Total training documents: {total_train:,}")
        print(f"Total test documents: {total_test:,}")
        print(f"Total test queries: {total_queries:,}")
        print(f"{'='*60}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Preprocess Wikipedia articles for cross-lingual retrieval'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory with raw JSON files (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed CSV files (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process all languages
    preprocessor.process_all_languages()
    
    print("\n✅ Preprocessing complete!")
    print(f"📁 Files saved in: {args.output_dir}\n")


if __name__ == '__main__':
    main()
