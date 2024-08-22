from dotenv import load_dotenv
load_dotenv('../.env')
import os
os.environ['HF_HOME'] = '/home/t/.cache/huggingface'

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
import asyncio

from llama_index.llms.gemini import Gemini 
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.postprocessor import SentenceTransformerRerank


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, Node
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
import os
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore, Node


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]
    

class SearchEvent(Event):
    query: str

embedding_model = GeminiEmbedding()
embedding_model2 = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder=os.environ['HF_HOME']+'/hub')

llm = Gemini(model='models/gemini-1.5-pro')
Settings.llm = llm
Settings.embed_model  = embedding_model2


class RAGWorkflow(Workflow):
        
    @step(pass_context=True)
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        print('Vectorizing the documents...')
        documents = SimpleDirectoryReader(dirname).load_data()
        ctx.data["index"] = VectorStoreIndex.from_documents(
            documents=documents,
            # embed_model=embedding_model,
        )
        return StopEvent(result=f"Indexed {len(documents)} documents.")

        
    @step(pass_context=True)
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        ctx.data["query"] = query

        # get the index from the global context
        index = ctx.data.get("index")
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        print('Retrieving relevent nodes...')
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step(pass_context=True)
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        # Rerank the nodes
        # ranker = LLMRerank(
        #     choice_batch_size=5, top_n=3, llm=
        # )
        ranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
            )
\
        print(f"Reranking {len(ev.nodes)} nodes...")
        print(ctx.data.get("query"), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=ctx.data.get("query")
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step(pass_context=True)
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        query = ctx.data.get("query")
        context = '\n\n'.join(node.text for node in ev.nodes)
        prompt = f'this is query and you have to answer the query from the context as source of knowledge \
                read the context care fully this is the context you have to understand and answer accordingly. \
                if there is no relevent information in the context, only say "not enough information", \
                do not mention context, answer as you know from the context. :query: {query} context: {context} '
        response = await llm.acomplete(prompt)
        return StopEvent(result=response)



async def main():
    dirname = str(input('Set a directory to vectorize.'))
    w = RAGWorkflow()
    await w.run(dirname=dirname)

    user_input = str(input('ask a question: '))

    while not (user_input.strip().lower() == 'end'):
        
        
        result = await w.run(query=user_input)
        print(str(result))
        print()
        user_input = str(input('ask a question: '))
    return
        


if __name__=='__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
    
    # asyncio.run(asyncio.create_task(main))