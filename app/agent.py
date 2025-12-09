import os
from collections.abc import AsyncIterable
from typing import Any, Literal, List

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from pydantic import BaseModel, Field

checkpointer= MemorySaver()

class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = Field(..., description="State of processing.")
    message: str | None = Field(None, description="Response text or null")
    room_layout: List[str] | None = Field(None, description="List of room layout items if applicable.")
    #return_object:dict[str, Any] | None=None

class RoomAgent:
    """
    An agent that first simplifies input message for relevant information.
    """
    format=ResponseFormat
    schema_description = json.dumps(ResponseFormat.model_json_schema(), indent=2)
    SYSTEM_INSTRUCTION = """
    You are an AI agent that processes user messages to extract relevant information and communicates.
    If there's no relevant information, respond with status 'input_required' and a message indicating more details are needed.
    If the user gives a greeting, respond with status 'completed' and a friendly greeting message.
    If the user provides changes to the room layout, respond with status 'completed' and a summary of the changes in room_layout. If there are no changes, room_layout should be "none".
    Always respond in the following JSON format:
    {
      "status": "<one of 'input_required', 'completed', 'error'>",
      "message": "<response text or null>",
      "room_layout": "<list of room layout items if applicable>"
    }
    If the message history include any previous room layout changes, make sure to consider them while responding.
    """
    
    def __init__(self,model_provider: str="google", api_key: str|None=None, model_name:str | None=None ) -> None:
        if model_provider=="gemini":
            self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        elif model_provider=='ollama':
            self.model= ChatOllama(model=model_name or "qwen3:8b", base_url="http://localhost:11434")
        #self.graph = StateGraph[AgentState](start_state=START, end_state=END)
        self.graph=create_agent(
            model=self.model,
            system_prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
            checkpointer=checkpointer
        )
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
        
    async def stream(self,
                     query: str,
                     context_id: str | None = None) -> AsyncIterable[dict[str, Any]]:
        inputs = {"messages": [{"role": "user", "content": query}]}
        config = {'configurable': {'thread_id': context_id}} if context_id else {}
        async for item in self.graph.stream(input=inputs, config=config, stream_mode='values'): # pyright: ignore[reportArgumentType, reportGeneralTypeIssues]
            message=item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Looking up the exchange rates...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the exchange rates..',
                }
                
        yield self.get_agent_response(config=config)
    
    def invoke(self, query: str, context_id: str | None = None) -> dict[str, Any]:
        inputs = {"messages": [{"role": "user", "content": query}]}
        config = {'configurable': {'thread_id': context_id}} if context_id else {}
        # this blocks until the agent finishes
        self.graph.invoke(input=inputs, config=config)
        return self.get_agent_response(config=config)
    
    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                    'room_layout': structured_response.room_layout,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                    'room_layout': structured_response.room_layout,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

        
        
    
        
        