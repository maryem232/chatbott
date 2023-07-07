from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from io import StringIO
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain
from typing import TextIO
import pandas as pd
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI
openai.api_key = "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"
model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ["OPENAI_API_KEY"] = "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"

pinecone.init(api_key='6833a51f-abd0-4540-9b77-c5d013f5b89c', environment='us-west4-gcp-free')
index = pinecone.Index('helow')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
