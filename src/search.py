import json
import numpy as np
import os
import hashlib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel:
    def __init__(self, model_type="openai", api_key=None, base_url=None):
        self.model_type = model_type

        if self.model_type == "openai":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url
            )
            self.model_name = "text-embedding-3-large"
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it with: pip install sentence-transformers"
                )

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "openai":
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self.client.embeddings.create(
                    input=batch, model=self.model_name
                )
                embeddings.extend([item.embedding for item in response.data])
            return np.array(embeddings)
        else:
            return self.model.encode(texts, show_progress_bar=len(texts) > 100)


class PaperSearcher:
    def __init__(self, papers_file, model_type="openai", api_key=None, base_url=None):
        with open(papers_file, "r", encoding="utf-8") as f:
            self.papers = json.load(f)

        self.model = EmbeddingModel(model_type, api_key, base_url)
        self.cache_file = self._get_cache_file(papers_file, self.model.model_name)
        self.embeddings = None

        self._load_cache()

    def _get_cache_file(self, papers_file, model_name):
        base_name = Path(papers_file).stem
        file_hash = hashlib.md5(papers_file.encode()).hexdigest()[:8]

        # Sanitize model_name for use in filename
        safe_model_name = "".join(
            c for c in model_name if c.isalnum() or c in ("-", "_")
        ).rstrip()

        cache_name = f"cache_{base_name}_{file_hash}_{safe_model_name}.npy"
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        return str(output_dir / cache_name)

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                self.embeddings = np.load(self.cache_file)
                if len(self.embeddings) == len(self.papers):
                    print(f"Loaded cache: {self.embeddings.shape}")
                    return True
                self.embeddings = None
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.embeddings = None
        return False

    def _save_cache(self):
        np.save(self.cache_file, self.embeddings)
        print(f"Saved cache: {self.cache_file}")

    def _create_text(self, paper):
        parts = []
        if paper.get("title"):
            parts.append(f"Title: {paper['title']}")
        if paper.get("abstract"):
            parts.append(f"Abstract: {paper['abstract']}")
        if paper.get("keywords"):
            kw = (
                ", ".join(paper["keywords"])
                if isinstance(paper["keywords"], list)
                else paper["keywords"]
            )
            parts.append(f"Keywords: {kw}")
        return " ".join(parts)

    def compute_embeddings(self, force=False):
        if self.embeddings is not None and not force:
            print("Using cached embeddings")
            return self.embeddings

        print(f"Computing embeddings ({self.model.model_name})...")
        texts = [self._create_text(p) for p in self.papers]

        self.embeddings = self.model.embed(texts)

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
                if ex.get("abstract"):
                    text += f" Abstract: {ex['abstract']}"
                texts.append(text)

            embs = self.model.embed(texts)
            query_emb = np.mean(embs, axis=0).reshape(1, -1)

        elif query:
            query_emb = self.model.embed(query).reshape(1, -1)
        else:
            raise ValueError("Provide either examples or query")

        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {"paper": self.papers[idx], "similarity": float(similarities[idx])}
            for idx in top_indices
        ]

    def display(self, results, n=10):
        print(f"\n{'='*80}")
        print(f"Top {len(results)} Results (showing {min(n, len(results))})")
        print(f"{'='*80}\n")

        for i, result in enumerate(results[:n], 1):
            paper = result["paper"]
            sim = result["similarity"]

            print(f"{i}. [{sim:.4f}] {paper['title']}")
            print(
                f"   #{paper.get('number', 'N/A')} | {paper.get('primary_area', 'N/A')}"
            )
            print(f"   {paper.get('forum_url', 'N/A')}\n")

    def save(self, results, output_file):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": self.model.model_name,
                    "total": len(results),
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved to {output_file}")
