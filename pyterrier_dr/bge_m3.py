from pyterrier.transformer import Transformer
from tqdm import tqdm
import pyterrier as pt
import pandas as pd
import numpy as np
import torch
from .biencoder import BiEncoder


class BGEM3(BiEncoder):
    def __init__(self, model_name='BAAI/bge-m3', batch_size=32, max_length=8192, text_field='text', verbose=False, device=None, use_fp16=False):
        super().__init__(batch_size, text_field, verbose)
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise ImportError("BGE-M3 requires the FlagEmbedding package. You can install it using 'pip install -U FlagEmbedding'")
        
        self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16, device=self.device)


    def __repr__(self):
        return f'BGEM3({repr(self.model_name)})'
    
    # Only does single_vec encoding
    def query_encoder(self, verbose=None, batch_size=None):
        return BGEM3QueryEncoder(self, verbose=verbose, batch_size=batch_size)
    def doc_encoder(self, verbose=None, batch_size=None):
        return BGEM3DocEncoder(self, verbose=verbose, batch_size=batch_size)
    
    # Can do dense, sparse and colbert encodings
    def query_multi_encoder(self, verbose=None, batch_size=None, return_dense=True, return_sparse=True, return_colbert_vecs=True):
        return BGEM3QueryEncoder(self, verbose=verbose, batch_size=batch_size, return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs)
    def doc_multi_encoder(self, verbose=None, batch_size=None, return_dense=True, return_sparse=True, return_colbert_vecs=True):
        return BGEM3DocEncoder(self, verbose=verbose, batch_size=batch_size, return_dense=return_dense, return_sparse=return_sparse, return_colbert_vecs=return_colbert_vecs)

class BGEM3QueryEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3, verbose=None, batch_size=None, max_length=None, return_dense=True, return_sparse=False, return_colbert_vecs=False):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

        self.dense = return_dense
        self.sparse = return_sparse
        self.multivecs = return_colbert_vecs
    
    def encode(self, texts):
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=self.dense, return_sparse=self.sparse, return_colbert_vecs=self.multivecs)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in ['query'])
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        bgem3_results = self.encode(it)

        if self.dense:
            query_vec = [bgem3_results['dense_vecs'][i] for i in inv]
            inp = inp.assign(query_vec=query_vec)
        if self.sparse:
            # for sparse convert ids to the actual tokens
            query_toks = [self.bge_factory.model.convert_id_to_token(bgem3_results['lexical_weights'][i]) for i in inv]
            inp = inp.assign(query_toks=query_toks)
        if self.multivecs:
            query_multivecs = [bgem3_results['colbert_vecs'][i] for i in inv]
            inp = inp.assign(query_multivecs=query_multivecs)

        return inp
    
    def __repr__(self):
        return f'{repr(self.bge_factory)}.query_encoder()'

class BGEM3DocEncoder(pt.Transformer):
    def __init__(self, bge_factory: BGEM3, verbose=None, batch_size=None, max_length=None, return_dense=True, return_sparse=False, return_colbert_vecs=False):
        self.bge_factory = bge_factory
        self.verbose = verbose if verbose is not None else bge_factory.verbose
        self.batch_size = batch_size if batch_size is not None else bge_factory.batch_size
        self.max_length = max_length if max_length is not None else bge_factory.max_length

        self.dense = return_dense
        self.sparse = return_sparse
        self.multivecs = return_colbert_vecs
        
    def encode(self, texts):
        return self.bge_factory.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length,
                             return_dense=self.dense, return_sparse=self.sparse, return_colbert_vecs=self.multivecs)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        # check if the input dataframe contains the field(s) specified in the text_field
        assert all(c in inp.columns for c in [self.bge_factory.text_field])
        it = inp[self.bge_factory.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Documents', unit='doc')
        bgem3_results = self.encode(it)

        if self.dense:
            doc_vec = bgem3_results['dense_vecs']
            inp = inp.assign(doc_vec=list(doc_vec))
        if self.sparse:
            toks = bgem3_results['lexical_weights']
            # for sparse convert ids to the actual tokens
            toks = [self.bge_factory.model.convert_id_to_token(doc) for doc in list(toks)]
            inp = inp.assign(toks=toks)
        if self.multivecs:
            doc_multivecs = bgem3_results['colbert_vecs']
            inp = inp.assign(doc_multivecs=list(doc_multivecs))

        return inp

    def __repr__(self):
        return f'{repr(self.bge_factory)}.doc_encoder()'
