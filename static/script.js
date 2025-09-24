document.getElementById('chat-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const userMessage = input.value.trim();
    if (!userMessage) return;

    // Display user message
    chatBox.innerHTML += `<div class="message user"><strong>You:</strong> ${userMessage}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
    input.value = '';

    // Send to backend (adjust endpoint as needed)
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: userMessage})
    });
    const data = await response.json();

    // Display LLM response
    chatBox.innerHTML += `<div class="message llm"><strong>LLM:</strong> ${data.reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
});