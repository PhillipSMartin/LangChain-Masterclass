# import embeddings model from OpenAI
from langchain.embeddings.base import Embeddings
# import base class to derive custom retriever from
from langchain.schema import BaseRetriever
# import vector database
from langchain.vectorstores.chroma import Chroma

# define custom class with two required functions
class RedundantFilterRetriever(BaseRetriever):
    # define attribute for injecting an Embeddings object
    embeddings: Embeddings
    # define attribute for injecting a pre-initialized Chroma database
    chroma: Chroma
    
    # accept a query and return a list of documents
    def get_relevant_documents( self, query ):
        # calculate embeddings for the query string
        emb = self.embeddings.embed_query( query )
        
        # feed embeddings into Chroma method that returns a list of similar documents, eliminating near-duplicates
        return self.chroma.max_marginal_relevance_search_by_vector(
            # specify embedding to search for
            embedding=emb,
            # specify tolerance (between 0 and 1) for similar documents (higher means more tolerance)
            lambda_mult=0.8
        )
        
    # define an asynchronous function that is required but that we will not use
    async def agent_relevant_documents( self ):
        return []