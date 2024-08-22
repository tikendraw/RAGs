import dotenv 
dotenv.load_dotenv()
from llama_index.core.workflow import Context
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.workflow import (Context,Workflow,StartEvent,StopEvent,step)
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import CompactAndRefine
# response_synthesizer = get_response_synthesizer(
#     response_mode=ResponseMode.COMPACT
# )
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from .abc.agents.rag_agent import RagAgent
from .abc.vector_db.base_vector_db import BaseVectorDB
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.tools.google import GmailToolSpec





class MyRagAgent(RagAgent):
    
    def __init__(self, db_retriever:BaseRetriever, **kwargs):
        self.retri = db_retriever
        
    