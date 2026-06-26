from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file
openai_client = OpenAI()  # Initialize the OpenAI client

from git_helper import load_lessons
from ingest import build_index
from rag_helper import RAGBase

lessons = load_lessons()
index = build_index(lessons)

rag_base = RAGBase(llm_client=openai_client, index = index)

response = rag_base.rag("How does the agentic loop work, and how is it different from plain RAG?")
print(response.answer)
print(response.usage) # can calculate cost from usage

response = rag_base.agentic_rag("How does the agentic loop work, and how is it different from plain RAG?")
print(response.last_message)
print(response.cost)



