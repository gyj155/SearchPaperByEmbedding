from .search import PaperSearcher
from dotenv import load_dotenv

load_dotenv(override=True)

# Use local model (free)
searcher = PaperSearcher("iclr2026_papers.json", model_type="local")

# Or use OpenAI (better quality)
# searcher = PaperSearcher("iclr2026_papers.json", model_type="openai")

searcher.compute_embeddings()

examples = [
    {
        "title": "Improving Developer Emotion Classification via LLM-Based Augmentation",
        "abstract": "Detecting developer emotion in the informative data stream of technical commit messages...",
    },
]

results = searcher.search(examples=examples, top_k=100)

searcher.display(results, n=10)
searcher.save(results, "output/results.json")
