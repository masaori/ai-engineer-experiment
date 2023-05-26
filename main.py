import json
from langchain.callbacks import get_openai_callback
from langchain.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.tools import ShellTool, BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, AgentExecutor, ConversationalChatAgent, ZeroShotAgent
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser, FORMAT_INSTRUCTIONS
from langchain.memory import ConversationBufferMemory
import os

from gpt4_output_parser import Gpt4OutputParser


os.environ["OPENAI_API_KEY"] = open("./openapi_key.txt", "r").read().strip()

# output_parser = Gpt4OutputParser()
# output_parser = ConvoOutputParser()
# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True)


def main():
    llm = ChatOpenAI(model_name='gpt-4')

    tools: list[BaseTool] = [
        ShellTool(),
        ReadFileTool(),
        WriteFileTool(
            description="""Use this to write file out to specific path you want to store.
        
        Action Plan Format:
        {{
            "file_path": string \\ The absolute file path to write to
            "text": string \\ The content to write to the file
            "append": bool \\ Whether to append to an existing file
        }}
        """),
        ListDirectoryTool(),
        # TODO: Make custom tool using gh command
        # TODO: Make custom tool using git command
    ]
    tool_descriptions = [f"{tool.name}: {tool.description}" for tool in tools]
    tool_names = [tool.name for tool in tools]

    action_play_history = []
    while True:
        action_plan_prompt = f"""
            What I want you to do:
            Please summary "what the following Typescript project aim" and write it out to README.md file to explain the project properly in Markdown notation.
            Project path: /Users/masaori/git/masaori/auto-test-writer-example-typescript

            Tools you can use:
            {tool_descriptions}

            Output:
            When responding to me, please output a response in one of two formats:

            **Option 1:**
            Use this if you want the human to use a tool.
            Markdown code snippet formatted in the following schema:

            {{
                "thought": string \\ The thought that led to this action
                "action": string \\ The action to take. Must be one of {tool_names}
                "action_input": string \\ The input to the action
            }}

            **Option #2:**
            Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

            {{
                "action": "Final Answer",
                "action_input": string \\ You should put what you want to return to use here
            }}

            History of your past Action Plans:
            {action_play_history}
        """
        # print(action_plan_prompt)
        action_plan_output = llm.predict(action_plan_prompt)
        # print(action_plan_output)
        action_plan = json.loads(action_plan_output)
        print('==== Action Plan ====')
        print(json.dumps(action_plan, indent=2))

        # If the action plan is a final answer, then return it
        if action_plan['action'] == 'Final Answer':
            print('==== Final Answer ====')
            print(action_plan['action_input'])
            break

        # Find the tool that matches the action plan from tools
        target_tool = None
        for tool in tools:
            if tool.name == action_plan['action']:
                target_tool = tool
                break
        if target_tool is None:
            raise Exception(
                f"Could not find tool with name {action_plan['action']}")
        print(f"==== Using tool: {target_tool.name} ====")

        action_plan_input: object
        try:
            action_plan_input = json.loads(action_plan['action_input'])
        except Exception as e:
            action_plan_input = action_plan['action_input']

        print('==== Action Plan Input ====')
        print(json.dumps(action_plan_input, indent=2))

        # Run the tool
        try:
            tool_output = target_tool.run(action_plan_input)
        except Exception as e:
            tool_output = str(e)

        print('==== Tool Output ====')
        print(tool_output)

        action_play_history.append({
            "thought": action_plan['thought'],
            "action": action_plan['action'],
            "action_input": action_plan['action_input'],
            "tool_output": tool_output,
        })


try:
    main()
except KeyboardInterrupt as e:
    print("KeyboardInterrupt")
except Exception as e:
    raise e

# agent_executor = initialize_agent(
#     agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     llm=llm,
#     tools=tools,
#     memory=memory,
#     verbose=True,
# )
# agent_executor.agent.output_parser = output_parser

# with get_openai_callback() as callback:
#     try:
#         response = agent_executor.run(
#             """
#             Please summary the following Typescript project and write it out to README.md file to explain the project properly.
#             Project path: /Users/masaori/git/masaori/auto-test-writer-example-typescript
#             """
#         )
#         print(response)
#     except KeyboardInterrupt as e:
#         print("KeyboardInterrupt")
#     except Exception as e:
#         response = str(e)
#         if not response.startswith("Could not parse LLM output: `"):
#             raise e
#         else:
#             response = response.removeprefix(
#                 "Could not parse LLM output: `").removesuffix("`")
#             print('Parse error occurred')

#     print(f"Total Tokens: {callback.total_tokens}")
#     print(f"Prompt Tokens: {callback.prompt_tokens}")
#     print(f"Completion Tokens: {callback.completion_tokens}")
#     print(f"Total Cost (USD): ${callback.total_cost}")
