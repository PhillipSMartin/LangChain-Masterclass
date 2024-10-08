# import function to read .env file
from dotenv import load_dotenv
# import LLMChain for our chain
from langchain.chains import LLMChain
# import chat version of OpenAI
from langchain.chat_models import ChatOpenAI
# import summary memory
from langchain.memory import ConversationSummaryMemory
# import prompt templates
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# read .env file and assign values to specified environment variables
load_dotenv()

# instantiate a ChatOpenAI LLM object - API key is obtained from OPENAI_API_KEY environment variable
chat = ChatOpenAI()

# instantiate memory
#   specify in "llm" the language model that the summary model should use for its internal chain
#   specify in "memory_key" the key to use in the input variables to be passed to the ChatPromptTemplate
#   return_message=True specifies that the value for that key should be message objects, not simply strings
memory = ConversationSummaryMemory(
    llm=chat,
    memory_key="messages",
    return_messages=True)

# instantiate a chat prompt template
#   specify the input variable keys
#   specify in 'messages' a list of nested prompt templates, constructed from the from_template method of the class
#   add a MessagesPlaceHolder to contain messages from the memory object and add the associated variable nam
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

# instantiate an LLM chain
#   specify the prompt template and the LLM
#   specify "verbose" so we can see what is being passed in the system message
chain = LLMChain(
    prompt=prompt,
    llm=chat,
    memory=memory,
    verbose=True
)

# get user input
while True:
    content = input( ">>" )

    # call the LLM chain, specifying values for the input variables obtained from user input
    #   save the result
    result = chain( { "content": content } )
    
    # print the generated output from the "text" key in the dictionary returned in "result"
    print(result["text"])
