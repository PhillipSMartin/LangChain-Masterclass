# Call OpenAI through an LLMChain

# import OpenAI for our LLM
from langchain.llms import OpenAI

# import classes to build a chain
# a chain consists of
#   a prompt template
#   an LLM
# the input to the chain is a dictionary of input variables
# the output of the chain is a dictionary of the input variables with new output variables
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# instantiate an OpenAI LLM object, specifying our API key
api_key = "..."
llm = OpenAI(
    openai_api_key=api_key
)

# instantiate a prompt template
#   the input_variables parameter specifies the names of the input variables
#   the template parameter specifies a string with placeholders for the input variables
#       (this string will be passed to the LLM as a prompt)
    
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a short {language} function that will {task}"
)

# instantiate an LLM chain
#   the llm parameter specifies the LLM to use
#   the prompt parameter specifies the prompt template to use
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

# call the chain with a dictionary of input variables
result = code_chain({
    "language":"Python", 
    "task":"return a list of numbers"
})

# print the 'text' variable of result (which is the output of the LLM)
# 'language' and 'task' are also included in result
print(result["text"])

# output:
# def divisible_by_3():
#     numbers = []
#     for i in range(1, 101):
#         if i % 3 == 0:
#             numbers.append(i)
#     return numbers