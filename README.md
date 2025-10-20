# Paper Semantic Search

Find similar papers using semantic search. Supports both local models (free) and OpenAI API (better quality).

## Features

- Request for papers from OpenReview (e.g., ICLR2026 submissions)
- Semantic search with example papers or text queries
- Support embedding caching 
- Embed model support: Open-source (e.g., all-MiniLM-L6-v2) or OpenAI

## Quick Start

```bash
pip install -r requirements.txt
```

### 1. Prepare Papers

```python
from crawl import crawl_papers

crawl_papers(
    venue_id="ICLR.cc/2026/Conference/Submission",
    output_file="iclr2026_papers.json"
)
```

### 2. Search Papers

```python
from search import PaperSearcher

# Local model (free)
searcher = PaperSearcher('iclr2026_papers.json', model_type='local')

# OpenAI model (better, requires API key)
# export OPENAI_API_KEY='your-key'
# searcher = PaperSearcher('iclr2026_papers.json', model_type='openai')

searcher.compute_embeddings()

# Search with example papers that you are interested in
examples = [
    {
        "title": "Your paper title",
        "abstract": "Your paper abstract..."
    }
]

results = searcher.search(examples=examples, top_k=100)

# Or search with text query
results = searcher.search(query="interesting topics", top_k=100)

searcher.display(results, n=10)
searcher.save(results, 'results.json')
```



## How It Works

1. Paper titles and abstracts are converted to embeddings
2. Embeddings are cached automatically
3. Your query is embedded using the same model
4. Cosine similarity finds the most similar papers
5. Results are ranked by similarity score

## Cache

Embeddings are cached as `cache_<filename>_<hash>_<model>.npy`. Delete to recompute.

## Example Output

```
================================================================================
Top 100 Results (showing 10)
================================================================================

1. [0.8456] Paper a
   #12345 | foundation or frontier models, including LLMs
   https://openreview.net/forum?id=xxx

2. [0.8234] Paper b
   #12346 | applications to robotics, autonomy, planning
   https://openreview.net/forum?id=yyy
```

## Tips

- Use 1-5 example papers for best results, or a paragraph of description of your interested topic
- Local model is good enough for most cases
- OpenAI model for critical search (~$1 for 18k queries)

If it's useful, please consider giving a star~