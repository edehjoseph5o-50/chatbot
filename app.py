import os
import uuid

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI


# LOAD ENVIRONMENT VARIABLES
load_dotenv()

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024


# NVIDIA CONFIGURATION
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "nvidia/nemotron-3.5-super-vl-120b-a12b"
)

client = None

if NVIDIA_API_KEY:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )


# BUSINESS INFORMATION
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

SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "")


# CUSTOMER SERVICE INSTRUCTIONS
SYSTEM_INSTRUCTION = f"""
You are the official AI customer service assistant for
{BUSINESS_NAME}.

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

IMPORTANT RULES:

1. Only provide information supported by the available business knowledge.
2. Never invent prices, product sizes, colours, stock availability,
delivery charges or technical specifications.
3. If you do not know an answer, recommend contacting the sales team.
4. Never claim an order has been placed or payment received.
5. Never request passwords, ATM card details or sensitive information.
6. Do not make promises on behalf of the business.

SUPPORT CONTACT:
{SUPPORT_CONTACT if SUPPORT_CONTACT else "No verified support contact is currently available."}

RESPONSE FORMATTING:

- Use clean, professional and natural language.
- Always respond in plain text.
- Do not use Markdown formatting.
- Do not use asterisks or unnecessary symbols.
- Avoid Markdown headings and code blocks.
- Use short paragraphs and numbered lists when necessary.
- Keep responses clear and customer-friendly.

Always represent {BUSINESS_NAME} professionally.
"""


# TEMPORARY CONVERSATION STORAGE
# Conversations remain available while Flask is running.
conversations = {}

MAX_HISTORY_MESSAGES = 20


# HOME PAGE
@app.route("/")
def home():
    return render_template("index.html")


# HEALTH CHECK
@app.route("/api/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "running",
        "model": NVIDIA_MODEL,
        "nvidia_configured": client is not None
    })


# CUSTOMER CHAT API
@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        # CHECK NVIDIA API KEY
        if client is None:
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

        if len(message) > 3000:
            return jsonify({
                "error": "Message is too long. Maximum is 3000 characters."
            }), 400

        # CREATE OR VALIDATE CONVERSATION
        if (
            not isinstance(conversation_id, str)
            or conversation_id not in conversations
        ):
            conversation_id = str(uuid.uuid4())

        # GET PREVIOUS CONVERSATION HISTORY
        history = conversations.get(conversation_id, [])

        # PREPARE MESSAGES FOR NVIDIA
        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            }
        ]

        messages.extend(history)

        messages.append({
            "role": "user",
            "content": message
        })

        # SEND REQUEST TO NVIDIA
        completion = client.chat.completions.create(

            model=NVIDIA_MODEL,

            messages=messages,

            temperature=1,

            top_p=0.95,

            max_tokens=4096,

            stream=False,

            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )

        # GET AI RESPONSE
        bot_reply = completion.choices[0].message.content

        if not bot_reply:

            bot_reply = (
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
                "content": bot_reply
            }
        ]

        conversations[conversation_id] = (
            updated_history[-MAX_HISTORY_MESSAGES:]
        )

        # RETURN RESPONSE TO FRONTEND
        return jsonify({

            "reply": bot_reply,

            "conversation_id": conversation_id,

            "model": NVIDIA_MODEL

        })

    except Exception:

        app.logger.exception("NVIDIA customer service error")

        return jsonify({
            "error": (
                "Sorry, we are experiencing a temporary issue. "
                "Please try again."
            )
        }), 500


# HANDLE LARGE REQUESTS
@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "error": "Your request is too large."
    }), 413


# RUN APPLICATION
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )