from pydantic import BaseModel
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import OpenAIResponsesRunner,  DisplayingRunnerCallback  
from openai.types.responses.easy_input_message import EasyInputMessage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

AGENTIC_INSTRUCTIONS = """
You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering.
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

class LLMUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class RagResponse(BaseModel):
    answer: str
    usage: LLMUsage

class RAGBase:

    def __init__(self,
        index,
        llm_client,
        prompt_template=PROMPT_TEMPLATE,
        filename="",
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = {"rag": INSTRUCTIONS, "agentic_rag": AGENTIC_INSTRUCTIONS}
        self.prompt_template = prompt_template
        self.filename = filename
        self.model = model

    def search(self, query: str, num_results: int = 5):
        """
        Search the lessons database for entries matching the given query.
        """
        boost_dict = {"content": 3.0,}
        filter_dict = {"filename": self.filename} if self.filename else {} 
            
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
    
    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append("content: " + doc["content"])
            lines.append("filename: " + doc["filename"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )
    
    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions["rag"]},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return RagResponse(
            answer=response.output_text,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
        )
    
    def agentic_rag(self, query, previous_messages: list[EasyInputMessage | ResponseFunctionToolCall | ResponseOutputMessage | dict] = []):
        agent_tools = Tools()
        agent_tools.add_tool(self.search)
        chat_interface = StdOutputInterface()
        callback = DisplayingRunnerCallback(chat_interface)

        runner = OpenAIResponsesRunner(
            tools=agent_tools,
            developer_prompt=self.instructions["agentic_rag"],
            chat_interface=chat_interface,
            llm_client=OpenAIClient(model=self.model)
        )

        return runner.loop(
            prompt=query,
            previous_messages= previous_messages,
            callback=callback,
        )