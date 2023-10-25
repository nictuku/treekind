import openai

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#openai.log = "debug"


try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    logging.info("Loading index from storage")
    index = load_index_from_storage(storage_context)
    logging.info("Index loaded")
except FileNotFoundError:
    logging.info("Loading documents")
    documents = SimpleDirectoryReader('data', recursive=True,
                                      num_files_limit=1500,
                                      required_exts=[".md"]).load_data()
    logging.info("Loaded %d documents", len(documents))
    logging.info("Building index")
    index = VectorStoreIndex.from_documents(documents)
    logging.info("Index built")
    logging.info("Storing index")
    index.storage_context.persist()
    logging.info("Index stored")

from openai.embeddings_utils import get_embedding, cosine_similarity
import pdb;pdb.set_trace()

doc = "162b987d-b61c-47d0-a71b-b631708cc164"
x = index.vector_store.get(doc)

sim = cosine_similarity(x, x)
print(sim)

retriever = index.as_retriever()
nodes = retriever.retrieve("Qual atividade estou planejando fazer com o Samuel?")
for node in nodes:
    print(node)
    print()
