const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

const chatMessages = document.getElementById("chat-messages");
const typingIndicator = document.getElementById("typing-indicator");
const newChatBtn = document.getElementById("new-chat-btn");

const suggestionButtons = document.querySelectorAll(".suggestion-btn");

let conversationId = localStorage.getItem("ugobestConversationId");
let isLoading = false;


// ADD A MESSAGE TO THE CHAT
function addMessage(message, sender) {

    const messageDiv = document.createElement("div");

    messageDiv.classList.add("message");
    messageDiv.classList.add(
        sender === "user" ? "user-message" : "bot-message"
    );

    const avatar = document.createElement("div");
    avatar.classList.add("message-avatar");
    avatar.textContent = sender === "user" ? "Y" : "U";

    const content = document.createElement("div");
    content.classList.add("message-content");

    // textContent prevents HTML or script injection
    content.textContent = message;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);

    chatMessages.appendChild(messageDiv);

    scrollToBottom();
}


// SCROLL TO THE LATEST MESSAGE
function scrollToBottom() {

    const chatBody = document.querySelector(".chat-body");

    chatBody.scrollTo({
        top: chatBody.scrollHeight,
        behavior: "smooth"
    });

}


// SHOW TYPING INDICATOR
function showTyping() {

    typingIndicator.classList.remove("hidden");

    scrollToBottom();

}


// HIDE TYPING INDICATOR
function hideTyping() {

    typingIndicator.classList.add("hidden");

}


// SEND MESSAGE TO FLASK BACKEND
async function sendMessage(message) {

    message = message.trim();

    if (!message || isLoading) {
        return;
    }

    isLoading = true;

    // Display customer message
    addMessage(message, "user");

    // Clear input
    userInput.value = "";

    // Disable button while waiting
    sendBtn.disabled = true;
    userInput.disabled = true;

    showTyping();

    try {

        const response = await fetch("/api/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                message: message,

                conversation_id: conversationId

            })

        });


        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error || "Something went wrong. Please try again."
            );

        }


        // Save conversation ID for future messages
        if (data.conversation_id) {

            conversationId = data.conversation_id;

            localStorage.setItem(
                "ugobestConversationId",
                conversationId
            );

        }


        // Display Gemini response
        addMessage(data.reply, "bot");

    }

    catch (error) {

        console.error("Chat error:", error);

        addMessage(
            error.message || "Unable to connect to the chatbot. Please try again.",
            "bot"
        );

    }

    finally {

        hideTyping();

        isLoading = false;

        sendBtn.disabled = false;
        userInput.disabled = false;

        userInput.focus();

    }

}


// HANDLE FORM SUBMISSION
chatForm.addEventListener("submit", function(event) {

    event.preventDefault();

    sendMessage(userInput.value);

});


// ENTER TO SEND MESSAGE
// SHIFT + ENTER TO CREATE A NEW LINE
userInput.addEventListener("keydown", function(event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        chatForm.requestSubmit();

    }

});


// SUGGESTED QUESTIONS
suggestionButtons.forEach(function(button) {

    button.addEventListener("click", function() {

        sendMessage(button.textContent);

    });

});


// START A NEW CONVERSATION
newChatBtn.addEventListener("click", function() {

    if (isLoading) {
        return;
    }

    conversationId = null;

    localStorage.removeItem("ugobestConversationId");

    chatMessages.innerHTML = "";

    addMessage(
        "Welcome! Thank you for contacting Ugobest Africa Paint. We are happy to assist you. What would you like to know about our products?",
        "bot"
    );

    userInput.focus();

});