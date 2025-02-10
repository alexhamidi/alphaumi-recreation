from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Dict, List, Any, Callable
from pydantic import BaseModel, Field
import json
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import omdb
from dotenv import load_dotenv
from typing_extensions import TypedDict, Literal
from prompt_lib import prompt_dict
import inspect
from functools import wraps

# Load environment variables
load_dotenv()

#==================================================#
# API SETUP
#==================================================#

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

omdb_api_key = os.getenv("OMDB_API_KEY")
if not omdb_api_key:
    raise ValueError("OMDB_API_KEY is not set in the environment variables.")
omdb_client = omdb.OMDBClient(apikey=omdb_api_key)

llm = ChatOpenAI(model="gpt-4o-mini")

#==================================================#
# Data Representation
#==================================================#




class Event(BaseModel):
    source: Literal["caller", "summarizer", "planner", "observation", "user"]
    value: str

class State(TypedDict):
    query: str
    execution_trajectory: List[Dict[str, Any]]
    action_rationale: str
    next_step: Literal["caller", "summarizer", "finish"]
    summary: str

class PlannerResponse(BaseModel):
    rationale: str = Field(description="Rationale for choosing the action")
    decision: Literal["caller", "summarizer", "finish"] = Field(description="Decide the next action to take")

class SummarizerResponse(BaseModel):
    summary: str = Field(description="Summary")

class CallerResponse(BaseModel):
    action: str = Field(description="Name for the next function")
    param: str = Field(description="Parameter passed to the function")
    # params: Dict[str, Any] = Field(description="Arguments to be passed to the function")
    # model_config = {
    #     "json_schema_extra": {
    #         "required": ["action", "params"]
    #     }
    # }

#==================================================#
# DEFINE TOOLS
#==================================================#

def tool(func: Callable) -> Callable:
    """Marks a function as a tool for dynamic extraction."""
    func._is_tool = True  # Custom attribute to flag tool functions
    return func

@tool
def web_search(query: str) -> Dict:
    """Conduct a web search using Tavily's search results."""
    from langchain_community.tools.tavily_search import TavilySearchResults
    search_tool = TavilySearchResults(max_results=2)
    return search_tool.invoke({"query": query})

@tool
def spotify_search(query: str) -> Dict:
    """Search for music on Spotify."""
    return sp.search(q=query)

@tool
def movie_search(query: str) -> Dict:
    """Search for a movie using OMDB API."""
    return omdb_client.get(title=query)


def extract_tool_info(func: Callable) -> Dict:
    """Extracts tool information from a function definition."""
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No description available."
    inputs = {name: param.annotation.__name__ for name, param in signature.parameters.items()}

    return {
        "name": func.__name__,
        "function": docstring,
        "input": inputs
    }

tool_functions = {name: func for name, func in globals().items() if callable(func) and getattr(func, "_is_tool", False)}
tools = [extract_tool_info(func) for func in tool_functions.values()]

#==================================================#
# UTIL FUNCTIONS
#==================================================#

def print_current_state(state: State):
    statestr = json.dumps(state, indent=2)
    print("====================CURRENT_STATE=====================")
    print(statestr)
    print("======================================================\n")

def get_tool_info_str() -> str:
    return json.dumps(tools, indent=2)


def get_execution_trajectory_str(execution_trajectory: List[Dict[str, Any]]) -> str:
    return json.dumps(execution_trajectory, indent=2)

def get_action_str(action_info: CallerResponse) -> str:
    return f'Action: {action_info.action}, Action Input: {json.dumps(action_info.param)}'

#==================================================#
# NODE IMPLEMENTATIONS
#==================================================#

def planner(state: State) -> State:
    """Decides the next step: calling an API, summarizing, or finishing."""
    print_current_state(state)

    tool_info_str = get_tool_info_str()
    execution_trajectory_str = get_execution_trajectory_str(state["execution_trajectory"])

    query = prompt_dict["planner"] \
                 .replace("{doc}", tool_info_str) \
                 .replace("{history}", execution_trajectory_str) \


    try:
        response: PlannerResponse = llm.with_structured_output(PlannerResponse).invoke([HumanMessage(content=query)])
    except Exception as e:
        print(f"Error invoking LLM for PlannerResponse: {e}")
        return state

    state["next_step"] = response.decision
    state["action_rationale"] = response.rationale
    state["execution_trajectory"].append(Event(source="planner", value=response.rationale).model_dump())

    return state



def caller(state: State) -> State:
    """Executes an API call based on the planner's decision."""
    tool_info_str = get_tool_info_str()
    execution_trajectory_str = get_execution_trajectory_str(state["execution_trajectory"])

    query = prompt_dict["caller"]\
                 .replace("{doc}", tool_info_str) \
                 .replace("{history}", execution_trajectory_str) \

    try:
        response: CallerResponse = llm.with_structured_output(CallerResponse).invoke([HumanMessage(content=query)])
    except Exception as e:
        print(f"Error invoking LLM for CallerResponse: {e}")
        return state

    action_str = get_action_str(response)
    state["execution_trajectory"].append(Event(source="caller", value=action_str).model_dump())

    action = response.action.strip().lower()

    print(f"ACTION: {action}")

    try:
        res = tool_functions[action](response.param)
        observation_str = json.dumps(res) if isinstance(res, dict) else str(res)
    except Exception as e:
        observation_str = f"Error executing {action}: {e}"


    state["execution_trajectory"].append(Event(source="observation", value=observation_str).model_dump())

    return state




def summarizer(state: State) -> State:
    """Summarizes the execution trajectory."""

    execution_trajectory_str = get_execution_trajectory_str(state["execution_trajectory"])


    query = prompt_dict["summarizer"] \
            .replace("{history}", execution_trajectory_str)

    messages = [HumanMessage(content=query)]

    try:
        response: SummarizerResponse = llm.with_structured_output(SummarizerResponse).invoke(messages)
    except Exception as e:
        print(f"Error invoking LLM for SummarizerResponse: {e}")
        return state

    state["execution_trajectory"].append(Event(source="summarizer", value=response.summary).model_dump())
    state["summary"] = response.summary

    return state

#==================================================#
# CONDITIONAL CHECK
#==================================================#

def next_node(state: State) -> str:
    """Determines the next node based on planner's decision."""
    return END if state["next_step"] == "finish" else state["next_step"]

#==================================================#
# GRAPH CONSTRUCTION
#==================================================#

def main():

    agent_builder = StateGraph(State)

    agent_builder.add_node("planner", planner)
    agent_builder.add_node("caller", caller)
    agent_builder.add_node("summarizer", summarizer)

    agent_builder.add_edge(START, "planner")
    agent_builder.add_conditional_edges("planner", next_node)
    agent_builder.add_edge("caller", "planner")
    agent_builder.add_edge("summarizer", END)

    agent = agent_builder.compile()

    query = input("Your query: ")
    # query = "tell me the running length of the movie blade runner"

    initial_state: State = {
        "query": query,
        "execution_trajectory": [Event(source="user", value=query).model_dump()],
        "action_rationale": "",
        "next_step": "",
        "summary": ""
    }

    final_state: State = agent.invoke(initial_state)
    print(final_state["summary"])


if __name__=="__main__":
    main()
