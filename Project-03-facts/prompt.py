# import function to read .env file
from dotenv import load_dotenv
# import class to build a retrieval chain
from langchain.chains import RetrievalQA
# import chat version of OpenAI
from langchain.chat_models import ChatOpenAI
# import embeddings model from OpenAI
from langchain.embeddings import OpenAIEmbeddings
# import vector database
from langchain.vectorstores.chroma import Chroma
# import our custom retriever
from redundant_filter_retriever import RedundantFilterRetriever

# read .env file and assign values to specified environment variables
load_dotenv()

# instantiate a ChatOpenAI LLM object - API key is obtained from OPENAI_API_KEY environment variable
chat = ChatOpenAI()

# instantiate embeddings model
embeddings = OpenAIEmbeddings()
# Can call embeddings.embed_query(text) to generate embeddings for a given text

# instantiate a Chroma (SQLite) database using constructor rather than from_documents method
#  specify the embeddings model (a different keyword for this syntaz)
#  specify the directory of the database (which already exists)
db = Chroma(
    embedding_function=embeddings,
    persist_directory="emb"
)

# get a retriever for the database
retriever = RedundantFilterRetriever(
		embeddings=embeddings,
		chroma=db
	)

# create a retrieaval chain of type RetreivalQA
#   specify the ChatOpenAI we instantiated above as the LLM
#   specify the retriever for our database
#   specify a chain_type of "stuff" 
chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

# for now, hard-code a question and run the chain
result = chain.run("What is an interesting fact about the English language?")

# print the result
print(result)