import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI(model="gpt-5.4-nano",temperature=0)

def make_default_graph():
    """Make a simple agent without tools"""
    graph_workflow = StateGraph(State)

    def call_model(state: State):
        return {"messages": [model.invoke(state["messages"])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("agent", END)

    agent = graph_workflow.compile()
    return agent


def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers together"""
        return a + b
    
    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model(state: State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END
        
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools",tool_node)

    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    
    return agent

agent = make_alternative_graph()

