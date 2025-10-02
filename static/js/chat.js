/**
 * Llama2-7B NPU Chatbot Frontend JavaScript
 */

class ChatBot {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isGenerating = false;
        
        this.initializeElements();
        this.initializeEventListeners();
        this.initializeWebSocket();
        this.updateModelInfo();
        
        // Auto-resize textarea
        this.autoResizeTextarea();
    }
    
    initializeElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        
        // Status elements
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        
        // Model info elements
        this.modelName = document.getElementById('modelName');
        this.modelStatus = document.getElementById('modelStatus');
        this.deviceType = document.getElementById('deviceType');
        
        // Performance elements
        this.responseTime = document.getElementById('responseTime');
        this.tokensPerSecond = document.getElementById('tokensPerSecond');
        this.tokenCount = document.getElementById('tokenCount');
        
        // Settings elements
        this.maxTokens = document.getElementById('maxTokens');
        this.temperature = document.getElementById('temperature');
        this.topP = document.getElementById('topP');
        this.topK = document.getElementById('topK');
        
        // Setting value displays
        this.maxTokensValue = document.getElementById('maxTokensValue');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.topPValue = document.getElementById('topPValue');
        this.topKValue = document.getElementById('topKValue');
        
        // Actions
        this.clearChatButton = document.getElementById('clearChat');
    }
    
    initializeEventListeners() {
        // Send message
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Settings sliders
        this.maxTokens.addEventListener('input', (e) => {
            this.maxTokensValue.textContent = e.target.value;
        });
        
        this.temperature.addEventListener('input', (e) => {
            this.temperatureValue.textContent = e.target.value;
        });
        
        this.topP.addEventListener('input', (e) => {
            this.topPValue.textContent = e.target.value;
        });
        
        this.topK.addEventListener('input', (e) => {
            this.topKValue.textContent = e.target.value;
        });
        
        // Clear chat
        this.clearChatButton.addEventListener('click', () => this.clearChat());
    }
    
    initializeWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus('connected', '接続済み');
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected', '切断されました');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.initializeWebSocket(), 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('disconnected', 'エラーが発生しました');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.updateConnectionStatus('connecting', '接続中...');
    }
    
    updateConnectionStatus(status, text) {
        this.statusDot.className = `status-dot ${status}`;
        this.statusText.textContent = text;
    }
    
    async updateModelInfo() {
        try {
            const response = await fetch('/api/model/info');
            const modelInfo = await response.json();
            
            this.modelName.textContent = modelInfo.name || 'Llama2-7B';
            this.modelStatus.textContent = modelInfo.is_loaded ? '読み込み済み' : '読み込み中...';
            
        } catch (error) {
            console.error('Failed to fetch model info:', error);
            this.modelStatus.textContent = 'エラー';
        }
    }
    
    sendMessage() {
        if (!this.isConnected || this.isGenerating) {
            return;
        }
        
        const message = this.messageInput.value.trim();
        if (!message) {
            return;
        }
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.adjustTextareaHeight();
        
        // Send message via WebSocket
        const messageData = {
            type: 'message',
            data: {
                message: message,
                settings: {
                    max_tokens: parseInt(this.maxTokens.value),
                    temperature: parseFloat(this.temperature.value),
                    top_p: parseFloat(this.topP.value),
                    top_k: parseInt(this.topK.value)
                }
            }
        };
        
        this.websocket.send(JSON.stringify(messageData));
        this.isGenerating = true;
        this.updateSendButton();
        
        // Add typing indicator
        this.addTypingIndicator();
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'start':
                // Response generation started
                break;
                
            case 'token':
                this.handleTokenReceived(data.data);
                break;
                
            case 'complete':
                this.handleGenerationComplete(data.data);
                break;
                
            case 'error':
                this.handleError(data.data);
                break;
                
            case 'pong':
                // Pong response for keep-alive
                break;
                
            default:
                console.warn('Unknown message type:', data.type);
        }
    }
    
    handleTokenReceived(data) {
        const token = data.token;
        
        // Remove typing indicator if it exists
        this.removeTypingIndicator();
        
        // Get or create the current bot message
        let currentMessage = this.getCurrentBotMessage();
        if (!currentMessage) {
            currentMessage = this.addMessage('', 'bot');
        }
        
        // Append token to the current message
        const messageContent = currentMessage.querySelector('.message-content');
        messageContent.textContent += token;
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    handleGenerationComplete(data) {
        this.isGenerating = false;
        this.updateSendButton();
        this.removeTypingIndicator();
        
        // Update performance info
        this.responseTime.textContent = `${data.inference_time.toFixed(2)}s`;
        this.tokenCount.textContent = data.total_tokens;
        
        const tokensPerSec = data.total_tokens / data.inference_time;
        this.tokensPerSecond.textContent = `${tokensPerSec.toFixed(1)} tokens/s`;
        
        // Add timestamp to the message
        const currentMessage = this.getCurrentBotMessage();
        if (currentMessage) {
            const timeElement = currentMessage.querySelector('.message-time');
            timeElement.textContent = new Date().toLocaleTimeString('ja-JP');
        }
    }
    
    handleError(data) {
        this.isGenerating = false;
        this.updateSendButton();
        this.removeTypingIndicator();
        
        const errorMessage = `エラー: ${data.error}`;
        this.addMessage(errorMessage, 'bot');
    }
    
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = sender === 'user' ? new Date().toLocaleTimeString('ja-JP') : '';
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }
    
    addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'typing-indicator';
        contentDiv.innerHTML = `
            <span>AIが入力中</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        typingDiv.appendChild(contentDiv);
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    getCurrentBotMessage() {
        const messages = this.chatMessages.querySelectorAll('.bot-message:not(.typing-indicator)');
        return messages[messages.length - 1];
    }
    
    updateSendButton() {
        this.sendButton.disabled = this.isGenerating || !this.isConnected;
        
        if (this.isGenerating) {
            this.sendButton.innerHTML = '<span class="send-icon">⏳</span>';
        } else {
            this.sendButton.innerHTML = '<span class="send-icon">📤</span>';
        }
    }
    
    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    clearChat() {
        // Keep only the welcome message
        const welcomeMessage = this.chatMessages.querySelector('.message.bot-message');
        this.chatMessages.innerHTML = '';
        if (welcomeMessage) {
            this.chatMessages.appendChild(welcomeMessage);
        }
        
        // Reset performance info
        this.responseTime.textContent = '-';
        this.tokensPerSecond.textContent = '-';
        this.tokenCount.textContent = '-';
    }
    
    autoResizeTextarea() {
        this.messageInput.addEventListener('input', () => {
            this.adjustTextareaHeight();
        });
    }
    
    adjustTextareaHeight() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});