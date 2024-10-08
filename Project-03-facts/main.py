# import function to read .env file
from dotenv import load_dotenv
# import class to load .txt files
from langchain.document_loaders import TextLoader
# import embeddings model from OpenAI
from langchain.embeddings import OpenAIEmbeddings
# import class to split text into chunks
from langchain.text_splitter import CharacterTextSplitter
# import vector database
from langchain.vectorstores.chroma import Chroma

# read .env file and assign values to specified environment variables
load_dotenv()

# instantiate embeddings model
embeddings = OpenAIEmbeddings()
# Can call embeddings.embed_query(text) to generate embeddings for a given text


# instantiate text splitter
#   specify newline character for separator
#   specify chunk size (can include multiple separators)
#   specify chunk overlap
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# instantiate a TextLoader to read in facts.txt
loader = TextLoader("facts.txt")
# create a list of documents from files.txt by splitting it into chunks
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# instantiate a Chroma (SQLite) database
#  specify list of documents you want to create embeddings for
#  specify the embeddings model
#  specify the directory of the database
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# return a vector of k (default = 4) results most similar to the parameter
#   the results are tuples containing the document and the similarity score
# can instead call similarity_search to get documents without the score
results = db.similarity_search_with_score( 
    "What is an interesting fact about the English language?",
    k=4
)

# print the results
#  [ 0 ] is the document, which contains the content in the page_content member
#  [ 1 ] is the similarity score
for result in results:
    print ( "\n " )
    print( result[ 1 ] )
    print( result[ 0 ].page_content )