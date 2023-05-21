import torch.cuda
import torch.backends
import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    'multilingual-mpnet':"paraphrase-multilingual-mpnet-base-v2",
    'mpnet':"all-mpnet-base-v2",
    'minilm-l6':'all-MiniLM-L6-v2',
    'minilm-l12':'all-MiniLM-L12-v2',
    'multi-qa':"multi-qa-mpnet-base-dot-v1",
    'alephbert':'imvladikon/sentence-transformers-alephbert',
    'sbert-cn':'uer/sbert-base-chinese-nli'
} 

# Embedding model name
EMBEDDING_MODEL_CN = "text2vec"
EMBEDDING_MODEL_EN = "minilm-l6"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "")



# 匹配后单段上下文长度
CHUNK_SIZE = 200