

from .base_agent import BaseAgent
from ..vector_db.base_vector_db import BaseVectorDB
from abc import ABC , abstractmethod
from llama_index.core.workflow import Workflow



class RagAgent(Workflow):
    
    def ingest(self,*args, **kwargs):
        return self._ingest(*args, **kwargs)
            
    async def aingest(self, *args, **kwargs):
        return await self._aingest(*args, **kwargs)
    
    @abstractmethod
    def _ingest(self, *args, **kwargs):
        pass

    @abstractmethod
    async def _aingest(self, *args, **kwargs):
        pass