import os
from flask import Flask, request, jsonify, render_template
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from google import genai


# LOAD ENVIRONMENT VARIABLES
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024


# GEMINI CONFIGURATION
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
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
    "Our paint offers high-quality coverage and anti-fade performance."
)

SUPPORT_CONTACT = os.getenv(
    "SUPPORT_CONTACT",
    ""
)


# INITIALIZE GEMINI CLIENT
client = None

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)


# CUSTOMER SERVICE SYSTEM INSTRUCTIONS
SYSTEM_INSTRUCTION = f"""
You are the official AI customer service assistant for
{BUSINESS_NAME}.

BUSINESS DESCRIPTION:
{BUSINESS_DESCRIPTION}

BUSINESS INFORMATION:
{BUSINESS_KNOWLEDGE}

YOUR ROLE:

1. Welcome customers politely and professionally.

2. Answer questions about the business and its products.

3. Explain product benefits in simple, understandable English.

4. Help customers understand how to make enquiries and place orders.

5. Assist customers with general product questions.

6. Maintain a friendly, helpful and professional tone.

7. Keep answers clear and reasonably short.

8. Understand and respond to natural English and Nigerian English.

IMPORTANT PRODUCT INFORMATION:

Ugobest Africa Paint is known for high-quality paint coverage
and anti-fade performance.

Explain these benefits when relevant.

IMPORTANT RULES:

1. Only provide business information that is available in your
business knowledge.

2. NEVER invent product prices, product sizes, colours, stock
availability, delivery charges, warranties or technical specifications.

3. If you do not know an answer, politely explain that the information
needs confirmation from the sales team.

4. Never claim that an order has been placed or payment received.

5. Never request customer passwords, ATM card numbers or other
sensitive payment information.

6. Do not make promises on behalf of the business.

7. If a customer wants to purchase a product, guide them towards
contacting the sales team.

SUPPORT CONTACT:

{SUPPORT_CONTACT if SUPPORT_CONTACT else "No verified customer support contact has been configured yet. Do not invent a phone number, email address or contact link."}

CONVERSATION STYLE:

Be welcoming.
Be professional.
Be helpful.
Use simple English.
Avoid unnecessarily long responses.



Always represent {BUSINESS_NAME} positively and professionally.



"""


@app.route("/")
def home():
    return render_template("index.html")


# SERVE CSS
@app.route("/style.css")
def stylesheet():

    return send_from_directory(
        BASE_DIR,
        "style.css"
    )


# SERVE JAVASCRIPT
@app.route("/script.js")
def javascript():

    return send_from_directory(
        BASE_DIR,
        "script.js"
    )


# HEALTH CHECK
@app.route("/api/health", methods=["GET"])
def health_check():

    return jsonify({

        "status": "running",

        "model": GEMINI_MODEL,

        "gemini_configured": client is not None

    })


# CUSTOMER CHAT API
@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        # CHECK API KEY
        if client is None:

            return jsonify({

                "error": "Gemini API key is missing. Please configure your .env file."

            }), 503


        # GET CUSTOMER MESSAGE
        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({

                "error": "Invalid request. Please send a valid JSON message."

            }), 400


        message = data.get("message", "")

        conversation_id = data.get("conversation_id")


        # VALIDATE MESSAGE
        if not isinstance(message, str):

            return jsonify({

                "error": "Your message must be text."

            }), 400


        message = message.strip()

        if not message:

            return jsonify({

                "error": "Please enter a message."

            }), 400


        if len(message) > 3000:

            return jsonify({

                "error": "Your message is too long. Please use fewer than 3000 characters."

            }), 400


        # VALIDATE CONVERSATION ID
        if conversation_id is not None:

            if (
                not isinstance(conversation_id, str)
                or len(conversation_id) > 300
            ):

                return jsonify({

                    "error": "Invalid conversation. Please start a new chat."

                }), 400


        # PREPARE GEMINI REQUEST
        request_arguments = {

            "model": GEMINI_MODEL,

            "input": message,

            "system_instruction": SYSTEM_INSTRUCTION,

        }


        # CONTINUE PREVIOUS CONVERSATION
        if conversation_id:

            request_arguments["previous_interaction_id"] = conversation_id


        # SEND CUSTOMER MESSAGE TO GEMINI
        response = client.interactions.create(
            **request_arguments
        )


        # GET AI RESPONSE
        bot_reply = response.output_text


        if not bot_reply:

            bot_reply = (
                "I'm sorry, I couldn't generate a response. "
                "Please try again."
            )


        # RETURN RESPONSE TO WEBSITE
        return jsonify({

            "reply": bot_reply,

            "conversation_id": response.id,

            "model": GEMINI_MODEL

        })


    except Exception:

        app.logger.exception("Gemini customer service error")

        return jsonify({

            "error": (
                "Sorry, we are experiencing a temporary issue. "
                "Please try again shortly."
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