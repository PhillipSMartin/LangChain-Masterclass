# import Tool class from LangChain
from langchain.tools import Tool
# import sql library
import sqlite3

# connect to database
conn = sqlite3.connect("db.sqlite")

# define function to run query
def run_sqlite_query(query):
    c = conn.cursor()
    c.execute( query )
    return c.fetchall()

# define Tool for ChatGPT
run_query_tool = Tool.from_function(
    # name is the name ChatGPT will use to ask for this tool
    name="run_sqlite_query",
    # description tells ChatGPT what the tool does - ChatGPT must understand this description so it knows when to use the tool
    description="Run a sqlite query.",
    # function is the function we must run to implement this tool
    func=run_sqlite_query
)
