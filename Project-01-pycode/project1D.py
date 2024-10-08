# Secure API key

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

# import argparse to parse command-line arguments
import argparse

# import load_dotenv to access environment variables
from dotenv import load_dotenv

# load environment variables from .env file
#   this will load the OPENAI_API_KEY environment variable from the .env file
load_dotenv()

# instantiate an argument parser
parser = argparse.ArgumentParser()
# add arguments
parser.add_argument("--language", type=str, default="Python", help="The programming language to use")
parser.add_argument("--task", type=str, default="return a list of numbers", help="The task to perform")
# parse the arguments
args = parser.parse_args()

# instantiate an OpenAI LLM object
# no need to specify an API key, as we'll use the environment variable OPENAI_API_KEY
llm = OpenAI()


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

# call the chain with a dictionary of input variables with values from the command-line arguments
result = code_chain({
    "language": args.language, 
    "task": args.task
})

# print the 'text' variable of result (which is the output of the LLM)
# 'language' and 'task' are also included in result
print(result["text"])

# call:
# python project1D.py --language C++ 

# output:
# std::vector<int> returnList(int n) {
#   std::vector<int> list;
#   for (int i = 1; i <= n; i++) {
#     list.push_back(i);
#   }
#   return list; 
# }