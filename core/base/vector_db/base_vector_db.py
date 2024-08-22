from abc import ABC, abstractmethod


class BaseVectorDB(ABC):
    
    def get_n_similar(self, n:int, **kwargs):
        return self._get_n_similar(n, **kwargs)
    
    @abstractmethod
    def _get_n_similar(self, n:int, **kwargs):
        pass

    async def aget_n_similar(self, n:int, **kwargs):
        return await self._aget_n_similar(n, **kwargs)
    
    @abstractmethod
    async def _aget_n_similar(self, n:int, **kwargs):
        pass
    
    
    
