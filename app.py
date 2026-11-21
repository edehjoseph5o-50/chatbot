
import os
import uuid
import json
import asyncio
import logging

from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from fastmcp import Client as MCPClient
from openai import OpenAI


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# NVIDIA CONFIGURATION
# --------------------------------------------------

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-super-vl-120b-a12b"
).strip()

nvidia_client = (
    OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        timeout=60.0
    )
    if NVIDIA_API_KEY else None
)


# --------------------------------------------------
# BUSINESS INFORMATION
# --------------------------------------------------

BUSINESS_NAME = os.getenv(
    "BUSINESS_NAME",
    "Ugobest Africa Paint"
)

BUSINESS_DESCRIPTION = os.getenv(
    "BUSINESS_DESCRIPTION",
    "A paint brand focused on quality paint products."
)

BUSINESS_KNOWLEDGE = os.getenv(
    "BUSINESS_KNOWLEDGE",
    "Ugobest Africa Paint offers high-quality coverage and anti-fade performance."
)

SUPPORT_CONTACT = os.getenv(
    "SUPPORT_CONTACT",
    ""
)


# --------------------------------------------------
# MCP CONFIGURATION
# --------------------------------------------------

MCP_SERVER_PATH = BASE_DIR / "time_server.py"


# --------------------------------------------------
# CONVERSATION SETTINGS
# --------------------------------------------------

MAX_MESSAGE_LENGTH = 3000

MAX_HISTORY_MESSAGES = 20

conversations = {}

conversation_lock = Lock()


# --------------------------------------------------
# CUSTOMER SERVICE SYSTEM INSTRUCTIONS
# --------------------------------------------------

SYSTEM_INSTRUCTION = f"""
You are the official AI customer service assistant
for {BUSINESS_NAME}.

BUSINESS DESCRIPTION:
{BUSINESS_DESCRIPTION}

BUSINESS KNOWLEDGE:
{BUSINESS_KNOWLEDGE}


YOUR RESPONSIBILITIES:

1. Welcome customers politely.
2. Answer questions about the business and its products.
3. Explain product benefits in simple English.
4. Help customers with product enquiries and orders.
5. Maintain a friendly and professional tone.
6. Understand Nigerian English.


IMPORTANT BUSINESS RULES:

1. Only provide information supported by the available
business knowledge.

2. Never invent product prices, sizes, colours, stock
availability, delivery charges or technical specifications.

3. If you do not know an answer, politely recommend
contacting the sales team.

4. Never claim that an order has been placed or payment
received.

5. Never request passwords, ATM card details or sensitive
customer information.

6. Do not make promises on behalf of the business.


SUPPORT CONTACT:

{SUPPORT_CONTACT if SUPPORT_CONTACT else "No verified customer support contact is currently available."}


MCP TOOL INSTRUCTIONS:

You have access to an MCP tool called get_current_time.

1. Always use get_current_time when a customer asks
about the current time, date or day.

2. Never guess or invent the current time or date.

3. The default timezone is Africa/Lagos.

4. When a customer requests another country's time,
pass the appropriate timezone to the tool.

5. Use the actual result returned by the MCP tool
when answering the customer.


RESPONSE FORMATTING:

- Use clean, professional and natural language.
- Always respond in plain text.
- Never use Markdown formatting.
- Do not use asterisks or unnecessary symbols.
- Avoid Markdown headings and code blocks.
- Use short paragraphs and numbered lists when necessary.
- Keep responses clear and customer-friendly.

Always represent {BUSINESS_NAME} professionally.
"""


# --------------------------------------------------
# MCP SCHEMA HELPERS
# --------------------------------------------------

_SCHEMA_STRIP_KEYS = {
    "additionalProperties",
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "title",
    "examples",
    "default"
}


def clean_schema(schema):

    if isinstance(schema, dict):

        return {
            key: clean_schema(value)
            for key, value in schema.items()
            if key not in _SCHEMA_STRIP_KEYS
        }

    if isinstance(schema, list):

        return [
            clean_schema(item)
            for item in schema
        ]

    return schema


def get_tool_schema(tool):

    schema = getattr(
        tool,
        "input_schema",
        None
    )

    if schema is None:

        schema = getattr(
            tool,
            "inputSchema",
            None
        )

    return schema or {
        "type": "object",
        "properties": {}
    }


# --------------------------------------------------
# DISCOVER MCP TOOLS
# --------------------------------------------------

async def list_mcp_tools():

    if not MCP_SERVER_PATH.exists():

        raise FileNotFoundError(
            f"MCP server not found: {MCP_SERVER_PATH}"
        )

    async with MCPClient(MCP_SERVER_PATH) as mcp_client:

        return await mcp_client.list_tools()


# --------------------------------------------------
# CONVERT MCP TOOLS TO NVIDIA FORMAT
# --------------------------------------------------

def mcp_tools_to_nvidia(mcp_tools):

    declarations = []

    for tool in mcp_tools:

        declarations.append({

            "type": "function",

            "function": {

                "name": tool.name,

                "description": tool.description or "",

                "parameters": clean_schema(
                    get_tool_schema(tool)
                )
            }
        })

    return declarations


# --------------------------------------------------
# EXECUTE MCP TOOLS
# --------------------------------------------------

async def call_mcp_tool(tool_name, arguments):

    async with MCPClient(MCP_SERVER_PATH) as mcp_client:

        result = await mcp_client.call_tool(
            tool_name,
            arguments
        )

        if result.is_error:

            raise RuntimeError(
                f"MCP tool execution failed: {tool_name}"
            )

        # Return structured result when available
        if result.data is not None:

            if isinstance(result.data, str):
                return result.data

            return json.dumps(
                result.data,
                ensure_ascii=False,
                default=str
            )

        # Otherwise return text content
        text_parts = []

        for item in result.content:

            if hasattr(item, "text"):

                text_parts.append(item.text)

        return "\n".join(text_parts) or "No result returned."


# --------------------------------------------------
# NVIDIA + MCP TOOL CALLING LOOP
# --------------------------------------------------

async def generate_with_mcp(messages_history):

    if nvidia_client is None:

        raise RuntimeError("NVIDIA API key is missing.")

    # Discover available MCP tools
    mcp_tools = await list_mcp_tools()

    # Convert discovered tools into NVIDIA format
    nvidia_tools = mcp_tools_to_nvidia(mcp_tools)

    logger.info(
        "Discovered MCP tools: %s",
        [tool.name for tool in mcp_tools]
    )

    # NVIDIA REQUEST
    def create_completion():

        return nvidia_client.chat.completions.create(

            model=NVIDIA_MODEL,

            messages=messages_history,

            temperature=0.3,

            top_p=0.95,

            max_tokens=1024,

            tools=nvidia_tools if nvidia_tools else None,

            parallel_tool_calls=True,

            tool_choice="auto",

            stream=False,

            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )

    # Make initial NVIDIA request
    response = await asyncio.to_thread(
        create_completion
    )

    message = response.choices[0].message

    # MCP TOOL EXECUTION LOOP
    for _ in range(3):

        if not message.tool_calls:

            return message.content or ""

        # Add NVIDIA assistant tool request
        messages_history.append(
            message.model_dump(exclude_none=True)
        )

        # Execute each requested tool
        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:

                arguments = json.loads(
                    tool_call.function.arguments or "{}"
                )

                if not isinstance(arguments, dict):

                    raise ValueError(
                        "Tool arguments must be a JSON object."
                    )

                logger.info(
                    "Executing MCP tool: %s with arguments: %s",
                    tool_name,
                    arguments
                )

                result_text = await call_mcp_tool(
                    tool_name,
                    arguments
                )

                logger.info(
                    "MCP tool result: %s",
                    result_text
                )

            except Exception as exc:

                logger.exception(
                    "MCP tool execution failed: %s",
                    tool_name
                )

                result_text = f"Tool error: {exc}"

            # Send MCP result back to NVIDIA
            messages_history.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "content": str(result_text)

            })

        # Ask NVIDIA to process the tool result
        response = await asyncio.to_thread(
            create_completion
        )

        message = response.choices[0].message

    # Return final response
    if message.tool_calls:

        return (
            "Sorry, I couldn't complete your request. "
            "Please try again."
        )

    return message.content or ""


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():

    return jsonify({

        "status": "running",

        "model": NVIDIA_MODEL,

        "nvidia_configured": nvidia_client is not None,

        "mcp_server_available": MCP_SERVER_PATH.exists()

    })


# --------------------------------------------------
# CUSTOMER CHAT API
# --------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        # CHECK NVIDIA API KEY
        if nvidia_client is None:

            return jsonify({

                "error": "NVIDIA API key is missing. Check your .env file."

            }), 503

        # GET CUSTOMER MESSAGE
        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({

                "error": "Invalid request."

            }), 400

        message = data.get("message", "")

        conversation_id = data.get("conversation_id")

        # VALIDATE MESSAGE
        if not isinstance(message, str):

            return jsonify({

                "error": "Message must be text."

            }), 400

        message = message.strip()

        if not message:

            return jsonify({

                "error": "Please enter a message."

            }), 400

        if len(message) > MAX_MESSAGE_LENGTH:

            return jsonify({

                "error": "Message is too long."

            }), 400

        # CREATE OR VALIDATE CONVERSATION
        with conversation_lock:

            if (
                not isinstance(conversation_id, str)
                or conversation_id not in conversations
            ):

                conversation_id = str(uuid.uuid4())

            history = list(
                conversations.get(conversation_id, [])
            )

        # PREPARE MESSAGE HISTORY
        messages_history = [

            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            }

        ]

        messages_history.extend(history)

        messages_history.append({

            "role": "user",

            "content": message

        })

        # GENERATE NVIDIA RESPONSE WITH MCP
        answer = asyncio.run(
            generate_with_mcp(messages_history)
        )

        answer = answer.strip()

        if not answer:

            answer = (
                "Sorry, I couldn't generate a response. "
                "Please try again."
            )

        # SAVE CONVERSATION HISTORY
        updated_history = history + [

            {
                "role": "user",
                "content": message
            },

            {
                "role": "assistant",
                "content": answer
            }

        ]

        with conversation_lock:

            conversations[conversation_id] = (
                updated_history[-MAX_HISTORY_MESSAGES:]
            )

        # RETURN RESPONSE TO FRONTEND
        return jsonify({

            "reply": answer,

            "conversation_id": conversation_id,

            "model": NVIDIA_MODEL

        })

    except Exception as exc:

        logger.exception("Customer service chat error")

        return jsonify({

            "error": (
                "Sorry, we are experiencing a temporary issue. "
                "Please try again."
            )

        }), 500


# --------------------------------------------------
# HANDLE LARGE REQUESTS
# --------------------------------------------------

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({

        "error": "Your request is too large."

    }), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
