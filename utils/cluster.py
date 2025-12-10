#!/usr/bin/env python3
"""
Clustering utility for word embeddings.
Supports Gemini and HuggingFace embedding models.
"""

import argparse
import time
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
import plotly.express as px
from typing import List, Callable, Optional
import os


def get_gemini_embedder(model_name: str, api_key: str) -> tuple[Callable, Callable]:
    """Return embedding function and LLM function for Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def embed_fn(texts: List[str]) -> np.ndarray:
        results = client.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        return np.array([np.array(emb.values) for emb in results.embeddings])

    def llm_fn(prompt: str) -> str:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text

    return embed_fn, llm_fn


def get_huggingface_embedder(model_name: str) -> tuple[Callable, None]:
    """Return embedding function for HuggingFace models."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    def embed_fn(texts: List[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    return embed_fn, None


def get_cluster_labels(viz_df: pd.DataFrame, llm_call: Callable) -> dict:
    """Auto-label clusters using LLM."""
    cluster_names = {}

    for cluster_id in viz_df['cluster'].unique():
        if cluster_id == '-1':
            cluster_names[cluster_id] = 'Outliers'
            continue

        cluster_words = viz_df[viz_df['cluster'] == cluster_id]['word'].tolist()[:10]
        cluster_defs = viz_df[viz_df['cluster'] == cluster_id]['definition'].tolist()[:10]

        prompt = f"""These words belong to the same semantic cluster:

Words: {', '.join(cluster_words)}

Definitions:
{chr(10).join(f'- {w}: {d}' for w, d in zip(cluster_words, cluster_defs))}

Provide a short 1-3 word category label for this cluster. Reply with just the label, nothing else."""

        label = llm_call(prompt).strip()
        cluster_names[cluster_id] = label

    return cluster_names


def cluster_embeddings(
    input_csv: str,
    provider: str,
    model_name: str,
    output_prefix: str,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
    save_embeddings: bool = True,
    auto_label: bool = True
):
    """Main clustering pipeline."""

    # Load data
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    words = df['word'].tolist()
    definitions = df['definition'].tolist()

    # Get embedding function
    if provider == 'gemini':
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY required for Gemini provider")
        embed_fn, llm_fn = get_gemini_embedder(model_name, api_key)
        texts = definitions

        # Batch process with rate limiting
        print(f"Generating embeddings with {provider}/{model_name}")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = embed_fn(batch)
            all_embeddings.append(emb)
            if i + batch_size < len(texts):
                time.sleep(1)
        embeddings = np.vstack(all_embeddings)

    elif provider == 'huggingface':
        embed_fn, llm_fn = get_huggingface_embedder(model_name)
        texts = (df['word'].astype(str) + ": " + df['definition'].astype(str)).tolist()
        print(f"Generating embeddings with {provider}/{model_name}")
        embeddings = embed_fn(texts)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Save embeddings
    if save_embeddings:
        emb_path = f'{output_prefix}_embeddings.npy'
        np.save(emb_path, embeddings)
        print(f"Embeddings saved to {emb_path}")

    # Mean-center
    embeddings_centered = embeddings - embeddings.mean(axis=0)

    # UMAP reduce
    print("Reducing dimensions with UMAP")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    coords = reducer.fit_transform(embeddings_centered)

    # Cluster
    print("Clustering with HDBSCAN")
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric='cosine').fit_predict(embeddings_centered)

    # Build visualization dataframe
    viz_df = pd.DataFrame({
        'word': words,
        'definition': definitions,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2] if n_components >= 3 else 0,
        'cluster': labels.astype(str)
    })

    # Auto-label clusters
    if auto_label and llm_fn:
        print("Auto-labeling clusters with LLM")
        cluster_names = get_cluster_labels(viz_df, llm_fn)
        viz_df['cluster_label'] = viz_df['cluster'].map(cluster_names)
    else:
        viz_df['cluster_label'] = viz_df['cluster']

    # Save clustered data
    csv_path = f'{output_prefix}_clustered.csv'
    viz_df.to_csv(csv_path, index=False)
    print(f"Clustered data saved to {csv_path}")

    # Create visualization
    print("Creating visualization")
    if n_components >= 3:
        fig = px.scatter_3d(
            viz_df, x='x', y='y', z='z',
            color='cluster_label',
            hover_data=['word', 'definition'],
            title=f'Word Clusters ({provider}/{model_name})'
        )
    else:
        fig = px.scatter(
            viz_df, x='x', y='y',
            color='cluster_label',
            hover_data=['word', 'definition'],
            title=f'Word Clusters ({provider}/{model_name})'
        )

    fig.update_traces(marker_size=4)
    html_path = f'{output_prefix}_clusters.html'
    fig.write_html(html_path)
    print(f"Visualization saved to {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Cluster word embeddings using Gemini or HuggingFace models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_csv', help='Input CSV file with word and definition columns')
    parser.add_argument('--provider', choices=['gemini', 'huggingface'], default='huggingface',
                        help='Embedding provider')
    parser.add_argument('--model', default=None,
                        help='Model name (default: gemini-embedding-001 for Gemini, '
                             'sentence-transformers/all-MiniLM-L6-v2 for HuggingFace)')
    parser.add_argument('--output', default='output',
                        help='Output file prefix')
    parser.add_argument('--api-key', help='API key for Gemini (or set GEMINI_API_KEY env var)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for embedding generation')
    parser.add_argument('--n-components', type=int, default=3,
                        help='Number of UMAP components')
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    parser.add_argument('--min-cluster-size', type=int, default=5,
                        help='HDBSCAN min_cluster_size parameter')
    parser.add_argument('--no-save-embeddings', action='store_true',
                        help='Skip saving embeddings to disk')
    parser.add_argument('--no-auto-label', action='store_true',
                        help='Skip auto-labeling clusters with LLM')

    args = parser.parse_args()

    # Set default models
    if args.model is None:
        if args.provider == 'gemini':
            args.model = 'gemini-embedding-001'
        else:
            args.model = 'sentence-transformers/all-MiniLM-L6-v2'

    cluster_embeddings(
        input_csv=args.input_csv,
        provider=args.provider,
        model_name=args.model,
        output_prefix=args.output,
        api_key=args.api_key,
        batch_size=args.batch_size,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster_size,
        save_embeddings=not args.no_save_embeddings,
        auto_label=not args.no_auto_label
    )


if __name__ == '__main__':
    main()
