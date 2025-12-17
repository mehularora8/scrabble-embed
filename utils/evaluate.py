import os
import time
import torch
import numpy as np
from google import genai
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

class ModelCardData:
    """
    Dummy model card data for Gemini.
    This is used to match the interface of the SentenceTransformer model card.
    """
    def __init__(self):
        self.model_card_data = {}
    
    def set_evaluation_metrics(self, model, metrics, epoch, steps):
        print(f"Setting evaluation metrics for {model} at epoch {epoch} and steps {steps}")
        print(f"Metrics: {metrics}")
        self.model_card_data['evaluation_metrics'] = metrics
        self.model_card_data['epoch'] = epoch
        self.model_card_data['steps'] = steps
    

class GeminiEmbedder:
    """Wrapper for Gemini to match SentenceTransformer interface."""

    def __init__(self, api_key, model_name="gemini-embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.similarity_fn_name = 'cosine'
        self.model_card_data = ModelCardData()

    def encode(self, sentences, batch_size=100, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            try:    
                print(f"Encoding batch {i}")
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
                all_embeddings.extend([e.values for e in result.embeddings])
            except Exception as e:
                print(f"Error encoding batch {i}: {e}")
                time.sleep(60)
                print(f"Retrying batch {i}")
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
                all_embeddings.extend([e.values for e in result.embeddings])
                continue
        
        return np.array(all_embeddings)

    @property
    def similarity(self):
        def cosine_sim(e1, e2):
            if isinstance(e1, np.ndarray):
                e1 = torch.from_numpy(e1.astype(np.float32))
            if isinstance(e2, np.ndarray):
                e2 = torch.from_numpy(e2.astype(np.float32))
            e1 = e1 / e1.norm(dim=1, keepdim=True)
            e2 = e2 / e2.norm(dim=1, keepdim=True)
            return e1 @ e2.T
        return cosine_sim

    def encode_query(self, queries, **kwargs):
        """Encode queries for retrieval."""
        return self.encode(queries, **kwargs)

    def encode_document(self, documents, **kwargs):
        """Encode documents for retrieval."""
        return self.encode(documents, **kwargs)


def create_evaluator(dataset, name="test"):
    """Create evaluator from dataset with word/definition columns."""
    return InformationRetrievalEvaluator(
        queries={i: ex['word'] for i, ex in enumerate(dataset)},
        corpus={i: ex['definition'] for i, ex in enumerate(dataset)},
        relevant_docs={i: [i] for i in range(len(dataset))},
        name=name
    )


def run_evaluations(test_dataset, base_model_name, finetuned_path=None, gemini_key=None):
    """Run evaluations for base, finetuned, and gemini models."""
    evaluator = create_evaluator(test_dataset)
    results = {}

    # Base model
    print("\n=== Base Model ===")
    base = SentenceTransformer(base_model_name)
    results['base'] = evaluator(base)
    print(results['base'])

    # Finetuned model
    if finetuned_path:
        print("\n=== Finetuned Model ===")
        finetuned = SentenceTransformer(finetuned_path)
        results['finetuned'] = evaluator(finetuned)
        print(results['finetuned'])

    # Gemini
    if gemini_key:
        print("\n=== Gemini ===")
        gemini = GeminiEmbedder(gemini_key)
        results['gemini'] = evaluator(gemini)
        print(results['gemini'])

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    # Get all metrics
    all_metrics = set()
    for res in results.values():
        all_metrics.update(res.keys())

    # Print each metric
    for metric in sorted(all_metrics):
        print(f"\n{metric}:")
        for model, res in results.items():
            if metric in res:
                print(f"  {model:12s}: {res[metric]:.4f}")

    return results

if __name__ == "__main__":
    test_dataset = load_dataset("csv", data_files="data/indian_words.csv")['train']
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    FINETUNED_PATH = "mehularora/scrabble-embed-v2"
    run_evaluations(
        test_dataset, 
        base_model_name=BASE_MODEL_NAME, 
        finetuned_path=FINETUNED_PATH, 
        gemini_key=GEMINI_API_KEY
    )