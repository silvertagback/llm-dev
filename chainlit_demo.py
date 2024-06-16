
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt4o"

from langchain_openai import AzureChatOpenAI
from langchain.tools import tool

# Non-sensitive tools (GET requests)

@tool
def mock_query_hotel_bookings(user: str) -> str:
    """This tool can be used to fetch information about hotel bookings."""
    return "Current bookings: 2 rooms booked for 3 nights each in Budapest."

@tool
def mock_query_flight_bookins(user: str) -> str:
    """This tool can be used to fetch information about flight bookings."""
    return "Current bookings: 2 flights booked to Budapest from Berlin."

@tool
def mock_query_car_rentals(user: str) -> str:
    """This tool can be used to fetch information about car rentals."""
    return "Current bookings: 1 car rented for 3 days in Budapest."

# Sensitive tools (e.g. POST requests)

@tool
def mock_book_flight(user: str, origin: str, destination: str, date: str) -> str:
    """This sensitive tool can be used to book flights."""
    return f"Flight booked from {origin} to {destination} on {date}. Flight number is LH123 with Lufthansa."

@tool
def mock_book_hotel(user: str, city: str, checkin: str, checkout: str) -> str:
    """This sensitive tool can be used to book hotels."""
    return f"Hotel booked in {city} from {checkin} to {checkout}. Room number is 123."

@tool
def mock_book_car(user: str, city: str, date: str) -> str:
    """This sensitive tool can be used to book cars."""
    return f"Car rented in {city} on {date}. Car model is BMW 3 Series."

plain_tools = [mock_query_hotel_bookings, mock_query_flight_bookins, mock_query_car_rentals]
sensitive_tools = [mock_book_flight, mock_book_hotel, mock_book_car]
all_tools = plain_tools + sensitive_tools

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

import chainlit as cl

AGENT_SYSTEM_PROMPT = SystemMessage(content="You are a travel assistant. You are asked to help the user with their travel plans.")

memory = SqliteSaver.from_conn_string("tool_agent_memory.sqlite")

@cl.on_chat_start
async def on_chat_start():
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        streaming=True
    )
    agent = create_react_agent(llm, all_tools, messages_modifier=AGENT_SYSTEM_PROMPT, checkpointer=memory)
    cl.user_session.set("agent", agent)

    config = RunnableConfig(
        configurable={
            "thread_id": "1",
        }
    )

    state = agent.get_state(config)
    chat_history = state.values["messages"]
    if chat_history:
        for message in chat_history:
            author = "user_message" if message.type == "human" else "assistant_message"
            if message.type != "tool":
                await cl.Message(content=message.content, type=author).send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    inputs = {"messages": [HumanMessage(content=message.content)]}

    config = RunnableConfig(
        configurable={
            "thread_id": "1",
        }, 
        callbacks=[cl.LangchainCallbackHandler(
            stream_final_answer=True,
        )]
    )
    
    response = agent.invoke(inputs, config)
    await cl.Message(content=response["messages"][-1].content).send()

