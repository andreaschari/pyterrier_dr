import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from .biencoder import BiEncoder
from tqdm import tqdm

class JinaEmbedder(BiEncoder):
    def __init__(self, model_name='jinaai/jina-embeddings-v4', batch_size=32, text_field='text', verbose=False, device=None):
        super().__init__(batch_size=batch_size, text_field=text_field, verbose=verbose)
        self.model_name = model_name
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = SentenceTransformer(model_name, trust_remote_code=True).to(self.device).eval()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    def encode_queries(self, texts, batch_size=None, prompt="query"):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)

        if len(texts) == 0:
            return np.empty(shape=(0, 0))

        return self.model.encode(sentences=texts, 
                                batch_size=batch_size or self.batch_size, 
                                show_progress_bar=show_progress,
                                task="retrieval",
                                prompt_name=prompt
                                )

    def encode_docs(self, texts, batch_size=None, prompt="passage"):
        show_progress = False
        if isinstance(texts, tqdm):
            texts.disable = True
            show_progress = True
        texts = list(texts)

        if len(texts) == 0:
            return np.empty(shape=(0, 0))
            
        return self.model.encode(sentences=texts, 
                                batch_size=batch_size or self.batch_size, 
                                show_progress_bar=show_progress,
                                task="retrieval",
                                prompt_name=prompt
                                )
    def __repr__(self):
        return f'JinaEmbedder({repr(self.model_name)})'