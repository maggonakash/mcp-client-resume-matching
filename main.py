"""
agent.py – CLI chatbot that connects to a running Playwright MCP server to help with resume screening.

Prerequisites
-------------
- Start your MCP server (e.g., at http://localhost:8000/mcp)
- pip install llama-index llama-index-tools-mcp llama-index-llms-openai python-dotenv
- export OPENAI_API_KEY=sk‑...
"""

import asyncio
import dotenv

from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ToolCall, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.tools.mcp.client import BasicMCPClient
from llama_index.tools.mcp.base import McpToolSpec

from llama_index.tools.mcp import (
    get_tools_from_mcp_url,
    aget_tools_from_mcp_url,
)

# Load environment variables from .env file
dotenv.load_dotenv()
# Initialize the language model
llm = OpenAI(model="gpt-4o")

# --- MODIFIED: Updated the system prompt for resume screening ---
SYSTEM_PROMPT = """\
You are an expert resume screening assistant. You help HR managers and recruiters by analyzing resumes,
extracting key information like skills, experience, and education, and evaluating them against job descriptions.
Use the provided browser tools to navigate to resumes (e.g., from a URL), extract relevant data,
and answer questions about candidates to determine their suitability for a role.
"""


# ---------- 1. Utility helpers ---------- #
async def handle_user_message(
    message_content: str,
    agent: ReActAgent,
    agent_context: Context,
    verbose: bool = False,
) -> str:
    """
    Feed one user turn through the agent, streaming intermediate tool calls
    so you can watch what it's doing.
    """
    handler = agent.run(message_content, ctx=agent_context)

    async for event in handler.stream_events():
        if verbose and isinstance(event, ToolCall):
            # --- FIX: Removed the hardcoded tool arguments that would break functionality ---
            print(f"\n🔧 Calling tool '{event.tool_name}' with {event.tool_kwargs}")
        if isinstance(event, AgentStream):
            print(f"{event.delta}", end="", flush=True)
        elif verbose and isinstance(event, ToolCallResult):
            # Shorten the output for better readability in the console
            tool_output_str = str(event.tool_output)
            if len(tool_output_str) > 250:
                tool_output_str = tool_output_str[:250] + "..."
            print(f"✅ Tool '{event.tool_name}' returned: {tool_output_str}")

    answer = await handler  # wait for the final assistant message
    return str(answer)


async def build_agent(mcp_url: str, args) -> ReActAgent:
    """
    Connect to the Playwright MCP server via its URL, pull its schema, turn everything
    into LlamaIndex FunctionTools, and wrap them in a ReActAgent.
    """
    mcp_client = BasicMCPClient(
        command_or_url=mcp_url,
        args=args,
        timeout=30
    )
    mcp_toolspec = McpToolSpec(client=mcp_client)

    print(f"Connecting to MCP server at {mcp_url} and fetching tools...")
    tools = await mcp_toolspec.to_tool_list_async()  # → list[FunctionTool]
    print(f"✅ Successfully fetched {len(tools)} tools.")

    return ReActAgent(
        name="Resume-Screening-Assistant",
        description="An agent that can control a browser to screen resumes and extract information.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        verbose=True,
    )


# ---------- 2. Main event‑loop ---------- #
async def main() -> None:
    # 2‑A. Bootstrap
    # --- MODIFIED: Connect to the running HTTP server instead of launching a subprocess ---
    # mcp_server_url = "http://0.0.0.0:8080/mcp"
    try:
        agent = await build_agent("uv", ["--directory", "/Users/akashmaggon/Desktop/Projects/Accelarators/Resume Screening Project/Client/mcp-client-llamaindex/mcp_resume_screening", "run", "server.py"])
    except Exception as e:
        print(f"\n❌ Failed to connect to MCP server at '{mcp_server_url}'.")
        print("   Please ensure the server is running and accessible.")
        print(f"   Error details: {e}")
        return

    agent_context = Context(agent)

    # 2‑B. Interactive REPL
    print("\n▶️  Type 'exit' or Ctrl‑C to quit.")
    print("📄 Resume screening assistant initialized. Ask me to analyze a resume from a URL.")
    while True:
        try:
            user_msg = input("You  : ")
            if user_msg.lower() in {"exit", "quit"}:
                break
            # Add an empty line for better visual separation of agent's response
            print("Agent: ", end="")
            reply = await handle_user_message(
                user_msg, agent, agent_context, verbose=True
            )
            # The reply is printed streamingly inside handle_user_message
            print("\n") # Newline after agent is finished
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break


if __name__ == "__main__":
    asyncio.run(main())