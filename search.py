import json
import numpy as np
import os
import hashlib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

class PaperSearcher:
    def __init__(self, papers_file, model_type="openai", api_key=None, base_url=None):
        with open(papers_file, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)
        
        self.model_type = model_type
        self.cache_file = self._get_cache_file(papers_file, model_type)
        self.embeddings = None
        
        if model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv('OPENAI_API_KEY'),
                base_url=base_url
            )
            self.model_name = "text-embedding-3-large"
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_name = "all-MiniLM-L6-v2"
        
        self._load_cache()
    
    def _get_cache_file(self, papers_file, model_type):
        base_name = Path(papers_file).stem
        file_hash = hashlib.md5(papers_file.encode()).hexdigest()[:8]
        cache_name = f"cache_{base_name}_{file_hash}_{model_type}.npy"
        return str(Path(papers_file).parent / cache_name)
    
    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                self.embeddings = np.load(self.cache_file)
                if len(self.embeddings) == len(self.papers):
                    print(f"Loaded cache: {self.embeddings.shape}")
                    return True
                self.embeddings = None
            except:
                self.embeddings = None
        return False
    
    def _save_cache(self):
        np.save(self.cache_file, self.embeddings)
        print(f"Saved cache: {self.cache_file}")
    
    def _create_text(self, paper):
        parts = []
        if paper.get('title'):
            parts.append(f"Title: {paper['title']}")
        if paper.get('abstract'):
            parts.append(f"Abstract: {paper['abstract']}")
        if paper.get('keywords'):
            kw = ', '.join(paper['keywords']) if isinstance(paper['keywords'], list) else paper['keywords']
            parts.append(f"Keywords: {kw}")
        return ' '.join(parts)
    
    def _embed_openai(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            embeddings.extend([item.embedding for item in response.data])
        
        return np.array(embeddings)
    
    def _embed_local(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=len(texts) > 100)
    
    def compute_embeddings(self, force=False):
        if self.embeddings is not None and not force:
            print("Using cached embeddings")
            return self.embeddings
        
        print(f"Computing embeddings ({self.model_name})...")
        texts = [self._create_text(p) for p in self.papers]
        
        if self.model_type == "openai":
            self.embeddings = self._embed_openai(texts)
        else:
            self.embeddings = self._embed_local(texts)
        
        print(f"Computed: {self.embeddings.shape}")
        self._save_cache()
        return self.embeddings
    
    def search(self, examples=None, query=None, top_k=100):
        if self.embeddings is None:
            self.compute_embeddings()
        
        if examples:
            texts = []
            for ex in examples:
                text = f"Title: {ex['title']}"
                if ex.get('abstract'):
                    text += f" Abstract: {ex['abstract']}"
                texts.append(text)
            
            if self.model_type == "openai":
                embs = self._embed_openai(texts)
            else:
                embs = self._embed_local(texts)
            
            query_emb = np.mean(embs, axis=0).reshape(1, -1)
        
        elif query:
            if self.model_type == "openai":
                query_emb = self._embed_openai(query).reshape(1, -1)
            else:
                query_emb = self._embed_local(query).reshape(1, -1)
        else:
            raise ValueError("Provide either examples or query")
        
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [{
            'paper': self.papers[idx],
            'similarity': float(similarities[idx])
        } for idx in top_indices]
    
    def display(self, results, n=10):
        print(f"\n{'='*80}")
        print(f"Top {len(results)} Results (showing {min(n, len(results))})")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results[:n], 1):
            paper = result['paper']
            sim = result['similarity']
            
            print(f"{i}. [{sim:.4f}] {paper['title']}")
            print(f"   #{paper.get('number', 'N/A')} | {paper.get('primary_area', 'N/A')}")
            print(f"   {paper['forum_url']}\n")
    
    def save(self, results, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model_name,
                'total': len(results),
                'results': results
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")

