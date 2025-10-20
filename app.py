import streamlit as st
import pandas as pd
import json
import os
import sys
from dotenv import load_dotenv
from src.search import PaperSearcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(initial_sidebar_state="expanded")

load_dotenv(override=True)

st.title("Paper Semantic Search")

uploaded_file = st.file_uploader("Upload a papers JSON file", type="json")

if uploaded_file is not None:
    papers_json = json.load(uploaded_file)

    if isinstance(papers_json, dict) and "results" in papers_json:
        papers = [r["paper"] for r in papers_json["results"]]
    elif isinstance(papers_json, dict):
        papers = [papers_json]
    else:
        papers = papers_json

    papers_file = uploaded_file.name
    with open(papers_file, "w", encoding="utf-8") as f:
        json.dump(papers, f)

    st.sidebar.header("Settings")
    model_type = st.sidebar.selectbox("Select model type", ("openai", "local"), index=0)

    api_key = None
    base_url = None
    if model_type == "openai":
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        base_url_input = st.sidebar.text_input(
            "Enter your OpenAI Base URL (optional)",
            value=os.getenv("OPENAI_BASE_URL", ""),
        )
        base_url = base_url_input if base_url_input else None

    top_k = st.sidebar.number_input("Number of results", min_value=1, value=100)

    primary_areas = sorted(list(set(p.get("primary_area", "N/A") for p in papers)))
    selected_areas = st.sidebar.multiselect(
        "Filter by primary area", options=primary_areas
    )

    @st.cache_resource
    def _get_searcher(papers_file, model_type, api_key=None, base_url=None):
        return PaperSearcher(
            papers_file, model_type=model_type, api_key=api_key, base_url=base_url
        )

    try:
        searcher = _get_searcher(
            papers_file, model_type, api_key=api_key, base_url=base_url
        )
        with st.spinner("Computing embeddings..."):
            searcher.compute_embeddings()
        st.success("Embeddings computed and cached.")

        st.header("Search")
        query = st.text_area("Enter your search query (e.g., describe a paper)")

        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    results = searcher.search(query=query, top_k=top_k)

                st.header("Results")

                filtered_results = results
                if selected_areas:
                    filtered_results = [
                        r
                        for r in results
                        if r["paper"].get("primary_area", "N/A") in selected_areas
                    ]

                results_data = []
                for result in filtered_results:
                    paper = result["paper"]
                    results_data.append(
                        {
                            "Similarity": f"{result['similarity']:.4f}",
                            "Title": paper["title"],
                            "Primary Area": paper.get("primary_area", "N/A"),
                            "URL": paper.get("forum_url", "No URL available."),
                            "Abstract": paper.get("abstract", "No abstract available."),
                        }
                    )

                if results_data:
                    df = pd.DataFrame(results_data)
                    st.dataframe(df)
                else:
                    st.warning("No results found matching your criteria.")
            else:
                st.warning("Please enter a search query.")
    except ImportError as e:
        st.error(f"An error occurred: {e}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
