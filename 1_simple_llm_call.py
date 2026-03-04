from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "Give me recipe of chocolate cake."})
print(result)

# in langsmith one single exicuation is called a trace, and each step in the chain is called a node. So in this example, we have one trace with three nodes: the prompt node, the model node, and the parser node.
# langsmith = project + trace + run