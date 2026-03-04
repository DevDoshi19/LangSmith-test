from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 

os.environ["LANGCHAIN_PROJECT"] = "seuential_chain_example"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)
model2 = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)
