// DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot JavaScript

class ChatbotApp {
    constructor() {
        this.ws = null;
        this.clientId = this.generateClientId();
        this.isConnected = false;
        this.isGenerating = false;
        this.currentMessage = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.checkServerHealth();
        
        // Auto-resize textarea
        this.setupTextareaAutoResize();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.elements = {
            chatMessages: document.getElementById('chat-messages'),
            chatInput: document.getElementById('chat-input'),
            sendButton: document.getElementById('send-button'),
            sendIcon: document.getElementById('send-icon'),
            statusText: document.getElementById('status-text'),
            deviceText: document.getElementById('device-text'),
            memoryText: document.getElementById('memory-text'),
            performanceInfo: document.getElementById('performance-info'),
            settingsToggle: document.getElementById('settings-toggle'),
            settingsPanel: document.getElementById('settings-panel'),
            maxTokens: document.getElementById('max-tokens'),
            maxTokensValue: document.getElementById('max-tokens-value'),
            temperature: document.getElementById('temperature'),
            temperatureValue: document.getElementById('temperature-value'),
            topP: document.getElementById('top-p'),
            topPValue: document.getElementById('top-p-value')
        };
    }
    
    setupEventListeners() {
        // Send message
        this.elements.sendButton.addEventListener('click', () => this.sendMessage());
        this.elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Settings toggle
        this.elements.settingsToggle.addEventListener('click', () => {
            this.elements.settingsPanel.classList.toggle('show');
            this.elements.settingsToggle.classList.toggle('active');
        });
        
        // Settings sliders
        this.elements.maxTokens.addEventListener('input', (e) => {
            this.elements.maxTokensValue.textContent = e.target.value;
        });
        
        this.elements.temperature.addEventListener('input', (e) => {
            const value = (e.target.value / 100).toFixed(1);
            this.elements.temperatureValue.textContent = value;
        });
        
        this.elements.topP.addEventListener('input', (e) => {
            const value = (e.target.value / 100).toFixed(1);
            this.elements.topPValue.textContent = value;
        });
    }
    
    setupTextareaAutoResize() {
        this.elements.chatInput.addEventListener('input', () => {
            this.elements.chatInput.style.height = 'auto';
            this.elements.chatInput.style.height = this.elements.chatInput.scrollHeight + 'px';
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.clientId}`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateStatus('接続済み', 'status-connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateStatus('切断', 'status-error');
            
            // Retry connection after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('エラー', 'status-error');
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'start':
                this.isGenerating = true;
                this.updateSendButton(true);
                this.currentMessage = this.addBotMessage('');
                this.showTypingIndicator();
                break;
                
            case 'token':
                this.removeTypingIndicator();
                if (this.currentMessage) {
                    const messageText = this.currentMessage.querySelector('.message-text');
                    messageText.textContent += data.data.token;
                    this.scrollToBottom();
                }
                break;
                
            case 'complete':
                this.isGenerating = false;
                this.updateSendButton(false);
                this.removeTypingIndicator();
                this.showPerformanceInfo(data.data);
                break;
                
            case 'error':
                this.isGenerating = false;
                this.updateSendButton(false);
                this.removeTypingIndicator();
                if (this.currentMessage) {
                    const messageText = this.currentMessage.querySelector('.message-text');
                    messageText.textContent = `エラー: ${data.data.message}`;
                    messageText.style.color = '#e53e3e';
                }
                break;
                
            case 'pong':
                // Handle ping/pong for connection keep-alive
                break;
        }
    }
    
    async checkServerHealth() {
        try {
            const response = await fetch('/api/health');
            const health = await response.json();
            
            this.elements.deviceText.textContent = health.npu_available ? 'NPU' : 'CPU';
            this.elements.memoryText.textContent = health.memory_usage;
            
            if (health.status === 'healthy') {
                this.updateStatus('準備完了', 'status-connected');
            } else {
                this.updateStatus('初期化中', 'status-loading');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus('サーバーエラー', 'status-error');
        }
        
        // Check health every 30 seconds
        setTimeout(() => this.checkServerHealth(), 30000);
    }
    
    sendMessage() {
        if (!this.isConnected || this.isGenerating) {
            return;
        }
        
        const message = this.elements.chatInput.value.trim();
        if (!message) {
            return;
        }
        
        // Add user message to chat
        this.addUserMessage(message);
        this.elements.chatInput.value = '';
        this.elements.chatInput.style.height = 'auto';
        
        // Get settings
        const settings = {
            max_tokens: parseInt(this.elements.maxTokens.value),
            temperature: parseFloat(this.elements.temperature.value) / 100,
            top_p: parseFloat(this.elements.topP.value) / 100
        };
        
        // Send to WebSocket
        this.ws.send(JSON.stringify({
            type: 'message',
            data: {
                message: message,
                settings: settings
            }
        }));
    }
    
    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
            </div>
        `;
        
        this.elements.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }
    
    addBotMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
            </div>
        `;
        
        this.elements.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }
    
    showTypingIndicator() {
        if (this.currentMessage) {
            const messageText = this.currentMessage.querySelector('.message-text');
            messageText.innerHTML = `
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
        }
    }
    
    removeTypingIndicator() {
        if (this.currentMessage) {
            const typingIndicator = this.currentMessage.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
        }
    }
    
    updateSendButton(isLoading) {
        if (isLoading) {
            this.elements.sendButton.disabled = true;
            this.elements.sendIcon.textContent = '⏳';
        } else {
            this.elements.sendButton.disabled = false;
            this.elements.sendIcon.textContent = '📤';
        }
    }
    
    updateStatus(text, className = '') {
        this.elements.statusText.textContent = text;
        this.elements.statusText.className = `status-value ${className}`;
    }
    
    showPerformanceInfo(data) {
        const tokensPerSecond = data.total_tokens / data.inference_time;
        this.elements.performanceInfo.textContent = 
            `生成時間: ${data.inference_time.toFixed(2)}秒 | ` +
            `トークン数: ${data.total_tokens} | ` +
            `速度: ${tokensPerSecond.toFixed(1)} トークン/秒`;
        
        // Hide performance info after 10 seconds
        setTimeout(() => {
            this.elements.performanceInfo.textContent = '';
        }, 10000);
    }
    
    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatbotApp();
});