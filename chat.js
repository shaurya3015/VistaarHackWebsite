/**
 * chat.js
 *
 * Handles the chat widget logic and the main UI dark mode toggle.
 */

const Chat = {
  // --- Properties ---
  chatToggle: null,
  chatWindow: null,
  chatClose: null,
  chatBody: null,
  chatInput: null,
  chatSend: null,

  /**
   * Initializes the Chat widget.
   */
  init() {
    // --- Find elements ---
    this.chatToggle = document.getElementById("chat-toggle");
    this.chatWindow = document.getElementById("chat-window");
    this.chatClose = document.getElementById("chat-close");
    this.chatBody = document.getElementById("chat-body");
    this.chatInput = document.getElementById("chat-input");
    this.chatSend = document.getElementById("chat-send");

    // Safety check
    if (
      !this.chatToggle ||
      !this.chatWindow ||
      !this.chatClose ||
      !this.chatBody ||
      !this.chatInput ||
      !this.chatSend
    ) {
      console.warn("Chat elements not found. Chat widget not initialized.");
      return;
    }

    // --- Attach Event Listeners ---
    this.chatToggle.addEventListener("click", () => {
      this.chatWindow.classList.toggle("chat-open");
    });

    this.chatClose.addEventListener("click", () => {
      this.chatWindow.classList.remove("chat-open");
    });

    this.chatSend.addEventListener("click", () => this.sendMessage());
    this.chatInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") this.sendMessage();
    });
  },

  /**
   * Sends the message currently in the chat input.
   */
  sendMessage() {
    const messageText = this.chatInput.value.trim();
    if (messageText === "") return;

    // Add user message
    this.addMessage(messageText, "user");

    // Clear input
    this.chatInput.value = "";

    // Fetch bot response
    this.getBotResponse(messageText, null);
  },

  /**
   * Adds a message to the chat body.
   */
  addMessage(text, sender) {
    const messageEl = document.createElement("div");
    messageEl.classList.add("chat-message", sender);
    messageEl.textContent = text;
    this.chatBody.appendChild(messageEl);

    // Auto-scroll to bottom
    this.chatBody.scrollTop = this.chatBody.scrollHeight;
  },

  /**
   * Send a message with context (used for PDF integration).
   */
  sendMessageWithContext(context, prompt) {
    this.chatWindow.classList.add("chat-open");
    this.addMessage(`Context: "${context}"`, "context-message");
    this.addMessage(prompt, "user");
    this.getBotResponse(prompt, context);
  },

  /**
   * Get bot response from Flask backend.
   */
  async getBotResponse(prompt, context = null) {
    try {
      // Add temporary "Thinking..." message
      const tempMessage = document.createElement("div");
      tempMessage.classList.add("chat-message", "bot");
      tempMessage.textContent = "Thinking my friend...";
      this.chatBody.appendChild(tempMessage);
      this.chatBody.scrollTop = this.chatBody.scrollHeight;

      // Send POST request to backend
      const response = await fetch("https://backend-ktnv.onrender.com/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, context }),
      });

      console.log("Prompt that I sent was:", prompt);
      console.log("Context that I sent was:", context);
      const data = await response.json();
      console.log("Raw Flask response:", data);

      // Remove "Thinking..." message
      tempMessage.remove();

      // Add real bot response
      this.addMessage(data.response || "No response from backend.", "bot");
    } catch (error) {
      console.error("Error getting bot response:", error);

      // Remove "Thinking..." message if it exists
      const bots = this.chatBody.querySelectorAll(".bot");
      if (bots.length > 0) bots[bots.length - 1].remove();

      // Show fallback message
      this.addMessage("Error connecting to Python backend.", "bot");
    }
  },
};

// --- Initialize on DOM load ---
document.addEventListener("DOMContentLoaded", () => {
  // --- Main UI Dark Mode Toggle ---
  const darkModeToggle = document.getElementById("dark-mode-toggle");
  if (darkModeToggle) {
    darkModeToggle.addEventListener("click", () => {
      document.body.classList.toggle("dark-mode");
    });
  }

  // --- Initialize the Chat Widget ---
  Chat.init();
});

