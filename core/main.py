import os
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

# Setup LLM
llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Tool: Web Search
search = DuckDuckGoSearchRun()
tools = [Tool(name="Web Search", func=search.run, description="Search the internet")]

# Agent Setup
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Run
if __name__ == "__main__":
    print("Summit AI Agent Initialized.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print("Summit AI:", response)
