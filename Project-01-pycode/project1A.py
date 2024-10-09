# Call OpenAI directly

# import OpenAI for our LLM
from langchain.llms import OpenAI

# instantiate an OpenAI LLM object, specifying our API key
api_key = "..."
llm = OpenAI(
    openai_api_key=api_key
)
	
# pass instructions to the LLM as a text literal and save the result
result = llm("Write a haiku about the moon")

# print the result
print(result)