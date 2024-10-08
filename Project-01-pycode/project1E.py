# Secure API key

# import OpenAI for our LLM
from langchain.llms import OpenAI

# import classes to build a chain
# we use two types of chains:
#   LLMChain: a chain that takes a prompt template and an LLM
#   SequentialChain: a chain that takes a list of chains and applies them in sequence
from langchain.chains import LLMChain, SequentialChain
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

# instantiate a second prompt template
test_prompt = PromptTemplate(
    template="Write a unit test for the following {language} code:\n{code}",
    input_variables=["language", "code"]
)

# instantiate an LLM chain
#   the llm parameter specifies the LLM to use
#   the prompt parameter specifies the prompt template to use
#   the output_key parameter specifies the key in the output dictionary (instead of the default 'text')
#      this key will be used as input to the next chain
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# instantiate a second LLM chain
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
	output_key="test"
)

# instantiate a sequential chain to chain the two chains together
#   the chains parameter specifies the list of chains to chain together
#   the input_variables parameter specifies the input variables to pass to the first chain
#   the output_variables parameter specifies the outputs
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

# call the sequential chain with a dictionary of input variables with values from the command-line arguments
result = chain({
    "language": args.language, 
    "task": args.task
})

 # print the generated code and test separately
print(">>>>>> GENERATED CODE:")
print(result["code"])

print(">>>>>> GENERATED TEST:")
print(result["test"])

# output:
# >>>>>> GENERATED CODE:
#  such as

# def get_numbers(n):
#     """
#     Returns a list of numbers from 1 to n
#     """
#     numbers = []
#     for i in range(1, n+1):
#         numbers.append(i)
#     return numbers
    
# Example output: get_numbers(5) would return [1, 2, 3, 4, 5]
# >>>>>> GENERATED TEST:


# import unittest

# class TestGetNumbers(unittest.TestCase):
    
#     def test_get_numbers(self):
#         # Test input of 5
#         self.assertEqual(get_numbers(5), [1, 2, 3, 4, 5])
        
#         # Test input of 10
#         self.assertEqual(get_numbers(10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
#         # Test input of 0
#         self.assertEqual(get_numbers(0), [])
        
#         # Test input of negative number
#         self.assertEqual(get_numbers(-5), [])
        
#         # Test input of decimal number
#         self.assertEqual(get_numbers(3.5), [1, 2, 3])
        
#         # Test input of string
#         self.assertEqual(get_numbers("5"), [1, 2, 3, 4, 5])
        
# if __name__ == '__main__':
#     unittest.main()