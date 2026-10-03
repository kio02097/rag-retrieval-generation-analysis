"""Embedding and pooling implementations extracted from the research notebook."""
import torch
from torch import nn
from sentence_transformers import SentenceTransformer, models
from langchain_core.embeddings import Embeddings

class UnifiedConcatPooling(nn.Module):
    def __init__(self, word_embedding_dimension: int,
                 use_first=False, use_last=False, use_mean=False, use_max=False):
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.use_first = use_first
        self.use_last  = use_last
        self.use_mean  = use_mean
        self.use_max   = use_max

    def forward(self, features):
        token_embeddings = features["token_embeddings"]  # [B,T,H]
        attention_mask  = features["attention_mask"]     # [B,T]

        outs = []

        if self.use_first:
            outs.append(token_embeddings[:, 0])

        if self.use_last:
            last_idx = attention_mask.long().sum(dim=1) - 1
            last_idx = torch.clamp(last_idx, min=0)
            bidx = torch.arange(token_embeddings.size(0), device=token_embeddings.device)
            outs.append(token_embeddings[bidx, last_idx])

        if self.use_mean:
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            summed = (token_embeddings * mask).sum(dim=1)
            denom  = mask.sum(dim=1).clamp(min=1e-6)
            outs.append(summed / denom)

        if self.use_max:
            minus_inf = torch.finfo(token_embeddings.dtype).min
            masked = token_embeddings.masked_fill(attention_mask.eq(0).unsqueeze(-1), minus_inf)
            outs.append(masked.max(dim=1).values)

        if not outs:
            raise ValueError("At least one pooling must be enabled.")

        sentence_embedding = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        features["sentence_embedding"] = sentence_embedding
        return features

    def get_sentence_embedding_dimension(self):
        mult = int(self.use_first) + int(self.use_last) + int(self.use_mean) + int(self.use_max)
        return self.word_embedding_dimension * mult

CFG_MAP = {
    "C":  dict(first=True,  last=False, mean=False, max=False),
    "X":  dict(first=False, last=False, mean=False, max=True),
    "M":  dict(first=False, last=False, mean=True,  max=False),
    "CM": dict(first=True,  last=False, mean=True,  max=False),
    "CX": dict(first=True,  last=False, mean=False, max=True),
    "XM": dict(first=False, last=False, mean=True,  max=True),
}

MODEL_REGISTRY = {
    "SBERT": dict(model_name="sentence-transformers/all-MiniLM-L6-v2", is_decoder=False, force_append_eos=False),
    "bge":   dict(model_name="BAAI/bge-m3",                           is_decoder=False, force_append_eos=False),
    "e5":  dict(model_name="intfloat/multilingual-e5-large",                       is_decoder=False,  force_append_eos=False),
    "llama": dict(model_name="meta-llama/Llama-3.2-1B",               is_decoder=True,  force_append_eos=True),
   "bling": dict(model_name="Lajavaness/bilingual-embedding-large",               is_decoder=False,  force_append_eos = False),
  "solon": dict(model_name="OrdalieTech/Solon-embeddings-large-0.1",               is_decoder=False,  force_append_eos=False),
  "mxbai": dict(model_name="mixedbread-ai/mxbai-embed-large-v1",               is_decoder=False,  force_append_eos=False),
  "gemma": dict(model_name="google/embeddinggemma-300m",               is_decoder=False,  force_append_eos=False)
   
}

_model_cache = {}

_wrapper_cache = {}

def parse_collection(collection_name: str):
    if "_" not in collection_name:
        raise ValueError(f"collection_name must contain '_' like 'bge_C': {collection_name}")
    prefix, cond = collection_name.rsplit("_", 1)
    if cond not in CFG_MAP:
        raise ValueError(f"Unknown pooling condition suffix={cond} in {collection_name}")
    if prefix not in MODEL_REGISTRY:
        raise ValueError(f"Unknown collection prefix={prefix} in {collection_name} (add to MODEL_REGISTRY)")
    return prefix, cond

def build_sentence_transformer(prefix: str, cond: str, device: str):
    key = (prefix, cond, device)
    if key in _model_cache:
        return _model_cache[key]

    info = MODEL_REGISTRY[prefix]
    model_name = info["model_name"]
    is_decoder = info["is_decoder"]

    model_args = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
   
    word_embedding_model = models.Transformer(
        model_name_or_path=model_name,
        model_args=model_args,
        
    )

    tokenizer = word_embedding_model.tokenizer
    auto_model = word_embedding_model.auto_model

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            auto_model.resize_token_embeddings(len(tokenizer))

    auto_model.config.pad_token_id = tokenizer.pad_token_id

    cfg = CFG_MAP[cond].copy()

    if is_decoder and cfg["first"]:
        cfg["first"] = False
        cfg["last"] = True

    pooling_model = UnifiedConcatPooling(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        use_first=cfg["first"],
        use_last=cfg["last"],
        use_mean=cfg["mean"],
        use_max=cfg["max"],
    )

    st = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(device)
    _model_cache[key] = (st, tokenizer, info["force_append_eos"])
    return _model_cache[key]

def make_wrapper_for_collection(collection_name: str, device: str) -> Embeddings:
    cache_key = (collection_name, device)
    if cache_key in _wrapper_cache:
        return _wrapper_cache[cache_key]

    prefix, cond = parse_collection(collection_name)
    st, tokenizer, force_append_eos = build_sentence_transformer(prefix, cond, device)

    EOS = tokenizer.eos_token or ""

    def maybe_append_eos(text: str) -> str:
        if not force_append_eos or not EOS:
            return text
        text = "" if text is None else str(text)
        return text if text.endswith(EOS) else (text + EOS)

    class MyCustomModelWrapper(Embeddings):
        def __init__(self, model: SentenceTransformer, batch_size: int = 32, normalize: bool = False):
            self.model = model
            self.batch_size = batch_size
            self.normalize = normalize
            self.model.eval()

        def embed_documents(self, texts):
            texts = [maybe_append_eos(t) for t in texts]
            vecs = self.model.encode(
                texts, batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return vecs.tolist()

        def embed_query(self, text):
            text = maybe_append_eos(text)
            vec = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )[0]
            return vec.tolist()

    wrapper = MyCustomModelWrapper(st, batch_size=32, normalize=False)
    _wrapper_cache[cache_key] = wrapper
    return wrapper

