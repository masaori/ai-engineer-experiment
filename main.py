import json
import time
import datetime
from langchain.callbacks import get_openai_callback
from langchain.tools.file_management import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
import os
import argparse
from openai.error import InvalidRequestError

from tools import ReadAndMemorizeFileTool, ShellTool


parser = argparse.ArgumentParser(
    prog='Auto Test Writer',
    description='Write a test file automatically',
    epilog='Enjoy the program! :)')
parser.add_argument(
    '-p', '--project_path', help='The absolute path to the project you want to work on', required=True)
parser.add_argument(
    '-f', '--file_path', help='The absolute path to the file you want to work on', required=False)
parser.add_argument(
    '-d', '--dir_path', help='The absolute path to the directory you want to work on', required=False)
parser.add_argument(
    '-t', '--task-type', help='The task type you want to do.', required=True)


args = parser.parse_args()

os.environ["OPENAI_API_KEY"] = open("./openapi_key.txt", "r").read().strip()


def main():
    llm = ChatOpenAI(model_name='gpt-4')

    tools: list[BaseTool] = [
        ShellTool(
            project_path=args.project_path,
            description="""Use this to run shell commands."""),
        ReadFileTool(description="""Use this to read file from specific absolute path.
            Please make sure that the file exists with using ListDirectoryTool.
        """),
        ReadAndMemorizeFileTool(description="""Use this to read file from specific absolute path when you want to memoize this file.
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
    ]
    tool_descriptions = [f"{tool.name}: {tool.description}" for tool in tools]
    tool_names = [tool.name for tool in tools]

    project_path = args.project_path
    file_path = args.file_path
    dir_path = args.dir_path
    task_type = args.task_type
    task_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_instruction = f"""
                When responding to me, please output a response in the following JSON format:
                {{
                    "thought": string \\ The thought that led to this action
                    "action": string \\ The action to take. Must be one of {tool_names}. If you want to stop this conversation, please set this to "Final Answer".
                    "action_input": string \\ The input to the action
                }}

                - No need to include any explanation.
                - No need to make your JSON as a code block. Just a plain JSON string.
                """

    # what_i_want_you_to_do = f"""
    #             1. Create a Test Plan for {file_path}
    #                 - Sort out every test cases and write each down in a test plan file as <file_name>.testplan.<number>.txt at the same directory as the specified file
    # """ if args.role == 'planner' else f"""
    #             1. Find a Test Plan for {file_path}
    #                 - Find a test plan file as <file_name>.testplan.<number>.txt at the same directory as the specified file
    #             2. Choose one of them and write a Test file for {file_path}
    #                 - Write a test file as <file_name>.test.<number>.ts at the same directory as the specified file
    #                 - Aim to write a test file that covers as much of the test cases as possible.
    #             3. Check your Test file
    #                 - Check if the transpiling succeeds.
    #                 - Check if your tests pass correctly.
    #                 - If it fails, Fix your test file.
    #             4. Commit your Test file and Make Pull Request
    #                 - After you confirm that your test file is correct, Commit your test file.
    #                 - Make a pull request to the main branch.
    # """

    what_i_want_you_to_do = f"""
                - Make your own branch
                - Write a test file for {file_path} with jest
                    - Write a test file at the same directory as the specified file
                    - Aim to write a test file that covers as much of the test cases as possible.
                - Check your Test file
                    - Check if the transpiling succeeds.
                    - Check if your tests pass correctly.
                    - If it fails, Fix your test file.
                - Commit your Test file and Make Pull Request
                    - After you confirm that your test file is correct, Commit your test file.
                    - Make a pull request to the main branch.
    """ if task_type == "write_test" else f"""
                Please write only one test file in this task following the steps below:
                - Delete all local branch
                - Update the repository to latest
                - Create your own branch including Task ID
                - Find the .ts file in {dir_path} which doesn't have a *.test.ts file.
                - Write a test file for the file with jest
                    - Write a test file at the same directory as the specified file
                    - Aim to write a test file that covers as much of the test cases as possible.
                - Check your Test file
                    - Check if the transpiling succeeds.
                    - Check if your tests pass correctly.
                    - If it fails, Fix your test file.
                - Commit your Test file and Make Pull Request
                    - After you confirm that your test file is correct, Commit your test file.
                    - Make a pull request to the main branch.
    """ if task_type == "find_and_write_test" else f"""
                - Make your own branch
                - Split the file {file_path} into multiple files for each exported type definition.
                    - Each file should include only one type definition.
                    - Name each file as <file_name>.<type_name>.ts
                    - Put each file into the same directory as the specified file.
                - Check if the transpiling succeeds.
                - DO NOT forget to commit your changes and Make Pull Request
    """ if task_type == "split_file" else None

    if what_i_want_you_to_do is None:
        raise Exception(
            f"Invalid task type. Please specify one of the following: write_test, split_file")

    base_prompt = f"""
            Project Path:
                {project_path}
            
            Task ID:
                {task_id}

            What I want you to do:
                {what_i_want_you_to_do}

            Shell Command Tips:
                - If you want to check current directory:
                    - `pwd`
                - If you want to check if the typescript code transpiles properly:
                    - `npx tsc --noEmit`
                - If you want to check if the test code succeeds:
                    - `npx jest <path/to/test/file> --coverage --collectCoverageFrom=<path/to/test/file>
                - If you want to check current git status:
                    - `git status`
                - If you want to delete all local branches:
                    - `git branch | xargs git branch -D`
                - If you want to update the repository to latest:
                    - `git fetch`
                - If you want to make your own branch from main branch:
                    - `git checkout -b <your branch name>` origin/main
                - If you want to update your branch:
                    - `git pull --rebase origin main`
                - If you want to commit your changes:
                    - `git add . && git commit -m "<appropriate commit message>" && git push -u origin <your branch name>`
                - If you want to make Pull Request:
                    - `gh pr create --base main --title "<Your PR title>" --body "<Your PR body>"`

            Tools you can use:
            {json.dumps(tool_descriptions, indent=4)}

            Output:
                {output_instruction}
    """
    previous_tool_output = None
    memorized_tool_outputs = []
    error_in_previous_time = None
    action_iteration_time = 1
    while True:
        print('')
        print('')
        print(
            f"==== Start Action ==== {action_iteration_time}")
        # print('== Memory ==')
        # print(json.dumps(memorized_tool_outputs, indent=4))
        print('')
        action_iteration_time += 1

        prompt = base_prompt + f"""
            Previous Tool Output:
                {json.dumps(previous_tool_output, indent=4)}
            Memorized Tool Outputs:
                {json.dumps(memorized_tool_outputs, indent=4)}
        """ if error_in_previous_time is None else f"""
            Please address the error in previous time:
                {json.dumps(error_in_previous_time, indent=4)}
        """

        with get_openai_callback() as callback:
            try:
                start_time = time.time()
                action_plan_output = llm.predict(prompt)
                elapsed_time = time.time() - start_time
                # print(f"    Request Time: {elapsed_time}")
            except InvalidRequestError as e:
                # token limit exceeded
                print(f"==== Token Limit Exceeded ==== {e}")
                # remove the oldest action plan from history
                memorized_tool_outputs.pop(0)
                time.sleep(60)
                continue
            except Exception as e:
                raise e

            # print("    total_cost:", callback.total_cost)
            # print("    total_tokens:", callback.total_tokens)
            # print("    prompt_tokens:", callback.prompt_tokens)
            # print("    successful_requests:", callback.successful_requests)
            # print("    completion_tokens:", callback.completion_tokens)

        try:
            action_plan = json.loads(action_plan_output)
            if action_plan['action'] == 'Final Answer':
                print('==== Final Answer ====')
                print(json.dumps(action_plan, indent=4))
                print('==== Action Plan History ====')
                print(json.dumps(memorized_tool_outputs, indent=4))
                break

            print('==== Action Plan ====')
            print(f"Thought: {action_plan['thought']}")
            print(f"Action: {action_plan['action']}")
            print(f"Action Input: {action_plan['action_input']}")
            print('')
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

        action_plan_input: object
        try:
            action_plan_input = json.loads(action_plan['action_input'])
        except Exception as e:
            action_plan_input = action_plan['action_input']

        print(
            f"== Using Tool: {target_tool.name} {json.dumps(action_plan_input, indent=2)} ==")

        # Run the tool
        try:
            tool_output = target_tool.run(action_plan_input)
        except Exception as e:
            tool_output = str(e)

        print('== Tool Output ==')
        print(tool_output)
        print('')

        memorized_tool_outputs.append({
            "action_iteration_time": action_iteration_time,
            "thought": action_plan['thought'],
            "action": action_plan['action'],
            "action_input": action_plan['action_input'],
            "tool_output": tool_output if target_tool.name == 'read_and_memorize_file' or target_tool.name == 'list_directory' else '(omitted)',
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
