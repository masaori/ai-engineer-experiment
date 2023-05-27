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

os.environ["OPENAI_API_KEY"] = open("./openapi_key.txt", "r").read().strip()

# output_parser = Gpt4OutputParser()
# output_parser = ConvoOutputParser()
# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True)


def main():
    llm = ChatOpenAI(model_name='gpt-4')

    tools: list[BaseTool] = [
        ShellTool(description="""Use this to run shell command.
            Please move to the project directory before running the command like this:
            cd <absolute/path/to/project> && <command>
        """),
        ReadFileTool(description="""Use this to read file from specific path you want to read.
            Please make sure that the file exists with using ListDirectoryTool.
        """),
        WriteFileTool(
            description="""Use this to write file out to specific path you want to store.
        
            Input Format:
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

    project_path = '/Users/masaori/git/masaori/auto-test-writer-example-typescript'
    output_instruction = f"""
                When responding to me, please output a response in one of two options as JSON string:
                    - No need to include any explanation.
                    - No need to make your JSON as a code block. Just a plain JSON string.
                    
                **Option #1:**
                {{
                    "thought": string \\ The thought that led to this action
                    "action": string \\ The action to take. Must be one of {tool_names}
                    "action_input": string \\ The input to the action
                    "save_to_history": bool \\ In order to reduce the token amount, please set this to True only when you need to refer to this action plan in the future.
                }}

                **Option #2:**
                {{
                    "action": "Final Answer",
                    "action_input": string \\ You should put what you want to return to use here
                }}
                """

    action_plan_history = []
    error_in_previous_time = None
    action_iteration_time = 0
    while True:
        print('')
        print('')
        print(
            f"==== Start Action ==== {action_iteration_time} {error_in_previous_time} {action_plan_history}")
        action_iteration_time += 1

        prompt = f"""
            Project Path:
            {project_path}

            What I want you to do:
                1. Write a Test
                    - Please make a new branch with appropriate name.
                    - Please write a test file for the follwoing .ts file.
                        - Target ts File is /Users/masaori/git/masaori/auto-test-writer-example-typescript/src/domain/usecases/CreateUserUsecase.ts
                    - Please check the type definitions those are related to the target ts file.
                    - Please output the test file at the same directory as the actual ts file.
                2. Check your Test file
                    - Please check if the transpiling succeeds.
                    - Please check if your tests pass correctly.
                    - If it fails, please fix your test file.
                3. Commit your Test file and Make Pull Request
                    - After you confirm that your test file is correct, please commit your test file.
                    - Please make a pull request to the main branch.

            Tips:
                - If you want to check if the typescript code transpiles properly, please run the following command:
                    - `cd {project_path} && npx tsc --noEmit`
                - If you want to check if the test code succeeds, please run the following command:
                    - `cd {project_path} && npx jest <path/to/test/file>`
                - If you want to check current git status, you can use the following shell command:
                    - `cd {project_path} && git status`
                - If you want to make your own branch, please run the following command:
                    - `cd {project_path} && git checkout -b <your branch name>`
                - If you want to commit your changes, please run the following command:
                    - `cd {project_path} && git add . && git commit -m "<appropriate commit message>" && git push -u origin <your branch name>`
                - If you want to make Pull Request, you can use the following shell command:
                    - `cd {project_path} && gh pr create --base main --title "<Your PR title>" --body "<Your PR body>"`

            Tools you can use:
            {json.dumps(tool_descriptions, indent=4)}

            Output:
                {output_instruction}

            History of your past Action Plans:
                {json.dumps(action_plan_history, indent=4)}
        """ if error_in_previous_time is None else f"""
            Please address the error in previous time:
                {json.dumps(error_in_previous_time, indent=4)}

            Output:
                {output_instruction}
        """

        action_plan_output = llm.predict(prompt)

        try:
            action_plan = json.loads(action_plan_output)
            if action_plan['action'] == 'Final Answer':
                print('==== Final Answer ====')
                print(action_plan['action_input'])
                break

            print('==== Action Plan ====')
            print(f"Thought: {action_plan['thought']}")
            print(f"Action: {action_plan['action']}")
            print(f"Action Input: {action_plan['action_input']}")
            print(f"Save to History: {action_plan['save_to_history']}")
        except json.decoder.JSONDecodeError as e:
            print(f"==== JSON Decode Error ==== {action_plan_output}")
            error_in_previous_time = {
                "error_message": f"Failed to decode your response as JSON. Please try again. {str(e)}",
                "your_response": action_plan_output,
            }
            continue
        except KeyError as e:
            print(f"==== Key Error ==== {action_plan_output} {str(e)}")
            error_in_previous_time = {
                "error_message": f"Failed to find key ({str(e)}) in your response. Please try again.",
                "your_response": action_plan,
            }
            continue
        except Exception as e:
            raise e

        # Find the tool that matches the action plan from tools
        target_tool = None
        for tool in tools:
            if tool.name == action_plan['action']:
                target_tool = tool
                break
        if target_tool is None:
            print(f"==== Tool Not Found ==== {action_plan_output}")
            error_in_previous_time = {
                "error_message": f"Failed to find tool {action_plan['action']} in your response. Please try again.",
                "your_response": action_plan_output,
            }
            continue

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

        if action_plan['save_to_history']:
            action_plan_history.append({
                "action_iteration_time": action_iteration_time,
                "thought": action_plan['thought'],
                "action": action_plan['action'],
                "action_input": action_plan['action_input'],
                "tool_output": tool_output,
            })

        error_in_previous_time = None


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
