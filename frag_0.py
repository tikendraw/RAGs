from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

from llama_index.core.node_parser import (TokenTextSplitter, HierarchicalNodeParser,
JSONNodeParser,
SentenceWindowNodeParser,
SemanticSplitterNodeParser,
SemanticDoubleMergingSplitterNodeParser,
LanguageConfig,
NodeParser,
HierarchicalNodeParser,
TextSplitter,
MarkdownElementNodeParser,
MetadataAwareTextSplitter,
LangchainNodeParser,
UnstructuredElementNodeParser)
docs = SimpleDirectoryReader('./data')