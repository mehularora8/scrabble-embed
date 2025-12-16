# dict-embed

Embedding model fine-tuned on the Collins Scrabble Words (CSW24) dictionary. Maps words to their definitions in embedding space.

## Why

Wanted to explore semantic trends within the Scrabble lexicon. So fine-tuned a model.

## Approach

Contrastive learning fine-tune on `sentence-transformers/all-MiniLM-L6-v2`.

**Loss:** Matryoshka Representation Learning with Multiple Negatives Ranking Loss. The contrastive objective pulls word-definition pairs together while pushing apart non-matching pairs. Matryoshka trains embeddings at multiple dimensions (384, 256) simultaneously, so you can trade off quality vs. speed at inference time without retraining.

**Data:** CSW24 word-definition pairs, split 80/10/10 for train/val/test.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mehularora8/dict-embed")

# Embed a word
word_embedding = model.encode("QUIXOTIC")

# Embed a definition
def_embedding = model.encode("resembling Don Quixote; impractical")

# Use cosine similarity for retrieval
```

## Training

```bash
pip install -r requirements.txt
```

Then run the notebook. You'll need to:
1. Set `DATA_LOCATION` to your CSW24 csv path
2. Set `FINAL_MODEL_REPO` for local save
3. Set `REPO_ID` for HuggingFace Hub push

## Results

Evaluated using Information Retrieval metrics (accuracy, precision, recall, NDCG) on the validation set. The model learns to place words near their correct definitions in embedding space.

## License

MIT