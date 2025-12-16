import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import umap
from sklearn.cluster import HDBSCAN
from google import genai
from google.genai import types
from typing import List
import plotly.express as px

GEMINI_API_KEY = "AIzaSyAgTUD2uF5pifWrNh5Hq6Bhe8iBER6n4qk"
model_name = "gemini-embedding-001"
client = genai.Client(api_key=GEMINI_API_KEY)

def llm_call(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

def embed_text_batch(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a batch of texts."""
    results = client.models.embed_content(
        model=model_name,
        contents=texts,
        config=types.EmbedContentConfig(task_type="CLUSTERING")
    )
    return np.array([np.array(emb.values) for emb in results.embeddings])

def get_cluster_labels(viz_df, llm_call):
    """
    Auto-label clusters using LLM.
    llm_call: function that takes a prompt string and returns a label string
    """
    cluster_names = {}
    
    for cluster_id in viz_df['cluster'].unique():
        if cluster_id == '-1':
            cluster_names[cluster_id] = 'Outliers'
            continue
        
        # Get top words in cluster
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


def generate_embeddings(df, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=32):
    """
    Generate embeddings for a pandas DataFrame with 'word' and 'definition' columns.
    Embeddings are generated from the combined word + definition.

    Args:
        df: pandas DataFrame with 'word' and 'definition' columns
        model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        batch_size: batch size for processing (default: 32)

    Returns:
        numpy array of embeddings with shape (len(df), embedding_dim)
    """
    # Load the model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Combine word and definition for embedding
    texts = (df['word'].astype(str) + ": " + df['definition'].astype(str)).tolist()

    # Generate embeddings with batch processing
    print(f"Generating embeddings for {len(texts)} word-definition pairs...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


# Example usage
if __name__ == "__main__":
    # Load DataFrame
    df = pd.read_csv('data/indian_words.csv')
    words = df['word'].tolist()
    definitions = df['definition'].tolist()
    MODEL_NAME = 'mehularora/scrabble-embed-v2'
    BATCH_SIZE = 32
    EMBEDDINGS_PATH = 'embeddings_finetuned.npy'
    SAVE_EMBEDDINGS = True

    # Generate embeddings
    all_embeddings = generate_embeddings(
        df=df,
        model_name=MODEL_NAME, 
        batch_size=BATCH_SIZE
    )

    # Save embeddings if needed
    if SAVE_EMBEDDINGS:
        np.save(EMBEDDINGS_PATH, all_embeddings)
        print(f"Embeddings saved to {EMBEDDINGS_PATH}")

    # all_embeddings = np.load(EMBEDDINGS_PATH)

    baseline_emb = np.vstack(all_embeddings)

    # Mean-center
    baseline_centered = baseline_emb - baseline_emb.mean(axis=0)

    # UMAP reduce
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    baseline_coords = reducer.fit_transform(baseline_centered)

    # Cluster in high-dim
    baseline_labels = HDBSCAN(min_cluster_size=5, metric='cosine').fit_predict(baseline_centered)

    # Build df for viz
    viz_df = pd.DataFrame({
        'word': words,
        'definition': definitions,
        'x': baseline_coords[:, 0],
        'y': baseline_coords[:, 1],
        'z': baseline_coords[:, 2],
        'cluster': baseline_labels.astype(str)
    })

    cluster_names = get_cluster_labels(viz_df, llm_call)
    viz_df['cluster_label'] = viz_df['cluster'].map(cluster_names)


    # Save embeddings for later comparison
    np.save('finetuned_embeddings.npy', baseline_emb)
    viz_df.to_csv('finetuned_clustered.csv', index=False)

    # Viz
    fig = px.scatter_3d(
        viz_df, x='x', y='y', z='z',
        color='cluster_label',
        hover_data=['word', 'definition'],
        title='Indian-Origin Words (Finetuned Baseline, Mean-Centered)'
    )
    fig.update_traces(marker_size=4)
    fig.write_html('indian_words_finetuned.html')
