"""
Data Collection Script for Cross-Lingual Retrieval System

This script downloads Wikipedia articles in English, Spanish, and French.
Target: 5,000-10,000 articles per language.

"""

import wikipedia
import json
import os
import argparse
from tqdm import tqdm
import time
import random
from pathlib import Path


class WikipediaDataCollector:
    """
    Collects Wikipedia articles across multiple languages.
    """
    
    def __init__(self, output_dir='data/raw'):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save downloaded articles
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Language codes for Wikipedia
        self.lang_codes = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French'
        }
        
        # Categories to ensure diverse topics
        # These are popular categories that exist across all languages
        self.seed_categories = {
            'en': [
                'Science', 'Technology', 'History', 'Geography', 'Politics',
                'Sports', 'Arts', 'Literature', 'Medicine', 'Environment',
                'Economics', 'Philosophy', 'Music', 'Film', 'Biology',
                'Physics', 'Chemistry', 'Mathematics', 'Astronomy', 'Psychology'
            ],
            'es': [
                'Ciencia', 'Tecnología', 'Historia', 'Geografía', 'Política',
                'Deportes', 'Arte', 'Literatura', 'Medicina', 'Medio ambiente',
                'Economía', 'Filosofía', 'Música', 'Cine', 'Biología',
                'Física', 'Química', 'Matemáticas', 'Astronomía', 'Psicología'
            ],
            'fr': [
                'Science', 'Technologie', 'Histoire', 'Géographie', 'Politique',
                'Sport', 'Art', 'Littérature', 'Médecine', 'Environnement',
                'Économie', 'Philosophie', 'Musique', 'Cinéma', 'Biologie',
                'Physique', 'Chimie', 'Mathématiques', 'Astronomie', 'Psychologie'
            ]
        }
    
    def collect_articles(self, language, num_articles=5000):
        """
        Collect Wikipedia articles for a specific language.
        
        Args:
            language: Language code ('en', 'es', 'fr')
            num_articles: Target number of articles to collect
        
        Returns:
            List of article dictionaries
        """
        wikipedia.set_lang(language)
        articles = []
        seen_titles = set()
        
        print(f"\n{'='*60}")
        print(f"Collecting {num_articles} {self.lang_codes[language]} articles...")
        print(f"{'='*60}\n")
        
        # Use seed categories to get diverse articles
        seed_topics = self.seed_categories[language]
        
        with tqdm(total=num_articles, desc=f"{language.upper()} articles") as pbar:
            attempts = 0
            max_attempts = num_articles * 3  # Allow some failures
            
            while len(articles) < num_articles and attempts < max_attempts:
                attempts += 1
                
                try:
                    # Get random search term from seed topics or random
                    if random.random() < 0.7:  # 70% from seed topics
                        search_term = random.choice(seed_topics)
                    else:  # 30% completely random
                        search_term = wikipedia.random(1)
                    
                    # Search for articles
                    search_results = wikipedia.search(search_term, results=10)
                    
                    for title in search_results:
                        if len(articles) >= num_articles:
                            break
                        
                        # Skip if already collected
                        if title in seen_titles:
                            continue
                        
                        try:
                            # Get full article
                            page = wikipedia.page(title, auto_suggest=False)
                            
                            # Filter: Only include substantial articles
                            if len(page.content) < 500:  # Skip very short articles
                                continue
                            
                            article = {
                                'title': page.title,
                                'content': page.content,
                                'url': page.url,
                                'language': language,
                                'categories': page.categories[:10],  # First 10 categories
                                'summary': page.summary
                            }
                            
                            articles.append(article)
                            seen_titles.add(title)
                            pbar.update(1)
                            
                        except wikipedia.exceptions.DisambiguationError:
                            # Skip disambiguation pages
                            continue
                        except wikipedia.exceptions.PageError:
                            # Skip if page doesn't exist
                            continue
                        except Exception as e:
                            # Skip any other errors
                            continue
                    
                    # Small delay to avoid overwhelming Wikipedia servers
                    time.sleep(0.1)
                    
                except Exception as e:
                    continue
        
        print(f"\n✓ Collected {len(articles)} {self.lang_codes[language]} articles\n")
        return articles
    
    def save_articles(self, articles, language):
        """
        Save articles to JSON file.
        
        Args:
            articles: List of article dictionaries
            language: Language code
        """
        output_file = self.output_dir / f'wikipedia_{language}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved to {output_file}")
        
        # Also save a sample for quick inspection
        sample_file = self.output_dir / f'wikipedia_{language}_sample.json'
        sample = articles[:10]  # First 10 articles
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Sample saved to {sample_file}")
    
    def collect_all_languages(self, articles_per_lang=5000):
        """
        Collect articles for all languages.
        
        Args:
            articles_per_lang: Number of articles to collect per language
        """
        total_start = time.time()
        stats = {}
        
        for lang_code in ['en', 'es', 'fr']:
            start_time = time.time()
            
            # Collect articles
            articles = self.collect_articles(lang_code, articles_per_lang)
            
            # Save articles
            self.save_articles(articles, lang_code)
            
            # Calculate stats
            elapsed = time.time() - start_time
            stats[lang_code] = {
                'count': len(articles),
                'time': elapsed,
                'avg_length': sum(len(a['content']) for a in articles) / len(articles)
            }
        
        # Print summary
        total_time = time.time() - total_start
        self._print_summary(stats, total_time)
    
    def _print_summary(self, stats, total_time):
        """Print collection summary."""
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        
        total_articles = sum(s['count'] for s in stats.values())
        
        for lang, stat in stats.items():
            print(f"\n{self.lang_codes[lang]}:")
            print(f"  Articles: {stat['count']:,}")
            print(f"  Time: {stat['time']/60:.1f} minutes")
            print(f"  Avg length: {stat['avg_length']:,.0f} characters")
        
        print(f"\n{'='*60}")
        print(f"Total articles: {total_articles:,}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"{'='*60}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Collect Wikipedia articles for cross-lingual retrieval'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en', 'es', 'fr'],
        choices=['en', 'es', 'fr'],
        help='Languages to collect (default: en es fr)'
    )
    parser.add_argument(
        '--articles-per-lang',
        type=int,
        default=5000,
        help='Number of articles per language (default: 5000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = WikipediaDataCollector(output_dir=args.output_dir)
    
    # Collect articles
    print("\n" + "="*60)
    print("WIKIPEDIA DATA COLLECTION")
    print("="*60)
    print(f"Target: {args.articles_per_lang:,} articles per language")
    print(f"Languages: {', '.join(args.languages)}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    collector.collect_all_languages(articles_per_lang=args.articles_per_lang)
    
    print("\n✅ Data collection complete!")
    print(f"📁 Files saved in: {args.output_dir}\n")


if __name__ == '__main__':
    main()
