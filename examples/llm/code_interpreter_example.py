from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import (  
    create_csv_agent,
)
from langchain.tools import Tool

load_dotenv()


def main():
    instructions = """ You are an agent designed to write and execute python code to answer question.
    you have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the output.
    If it does not seem like you can write code to answer the question, say "I don't know how to answer that question."
    """

    base_promt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_promt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    csv_agent: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    tools = [
        Tool(
            name="csv_agent",
            description="""useful when you need to answer question over episode_info.csv,
            takes an input of a question and returns the answer after running pandas calculations""",
            func=lambda x: csv_agent.invoke({"input": x, "intermediate_steps": []})
        ),
        Tool(
            name="python_agent",
            description="""useful when you need to write and execute python code,
            takes an input of a task and returns the output of the python code
            DOES NOT ACCEPT CODE AS INPUT""",
            func=lambda x: python_agent.invoke({"input": x, "intermediate_steps": []})
        )
    ]
    prompt = base_promt.partial(instructions="")
    grand_agent = create_react_agent(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=tools,
        prompt=prompt,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    result = grand_agent_executor.invoke(
        {
            "input": "Generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/lanchain, you have qrcode package installed already."
        }
    )
    print(result)


if __name__ == "__main__":
    main() 