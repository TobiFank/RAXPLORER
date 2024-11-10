// src/lib/api.ts

import axios from 'axios';

// Types
export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
}

export interface Chat {
    id: string;
    title: string;
    messages: ChatMessage[];
    createdAt: string;
}

export interface FileMetadata {
    id: string;
    name: string;
    size: string;
    pages: number;
    uploadedAt: string;
}

export interface ModelConfig {
    provider: 'claude' | 'chatgpt' | 'ollama';
    apiKey?: string;
    model: string;
    temperature: number;
    systemMessage?: string;
}

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_V1_PREFIX = '/api/v1';

console.log('API Configuration:', {
    API_BASE_URL,
    API_V1_PREFIX,
    fullBaseUrl: `${API_BASE_URL}${API_V1_PREFIX}`
});

const api = axios.create({
    baseURL: `${API_BASE_URL}${API_V1_PREFIX}`,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add response interceptor for debugging
api.interceptors.response.use(
    response => {
        console.log('API Response:', {
            url: response.config.url,
            method: response.config.method,
            status: response.status,
            data: response.data
        });
        return response;
    },
    error => {
        console.error('API Error:', {
            url: error.config?.url,
            method: error.config?.method,
            status: error.response?.status,
            data: error.response?.data,
            message: error.message
        });
        return Promise.reject(error);
    }
);

// Chat API
export const chatApi = {
    // Chat Management endpoints remain the same
    createChat: async (): Promise<Chat> => {
        console.log("Creating new chat");
        const response = await api.post('/chat/chats', {
            title: "New Chat" // Show me what payload you're actually sending
        });
        console.log("Chat creation response:", response.data);
        return response.data;
    },

    getChats: async (): Promise<Chat[]> => {
        const response = await api.get('/chat/chats');
        return response.data;
    },

    getChat: async (chatId: string): Promise<Chat> => {
        const response = await api.get(`/chat/chats/${chatId}`);
        return response.data;
    },

    deleteChat: async (chatId: string): Promise<void> => {
        await api.delete(`/chat/chats/${chatId}`);
    },

    updateChatTitle: async (chatId: string, title: string): Promise<Chat> => {
        const response = await api.patch(`/chat/chats/${chatId}`, {title});
        return response.data;
    },

    sendMessage: async function* (chatId: string, content: string, modelConfig: ModelConfig) {
        console.log('Sending message:', {
            chatId,
            content,
            modelConfig,
            url: `${API_BASE_URL}${API_V1_PREFIX}/chat/chats/${chatId}/messages`
        });

        try {
            const response = await fetch(`${API_BASE_URL}${API_V1_PREFIX}/chat/chats/${chatId}/messages`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content,
                    modelConfig,
                }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Message send error:', {
                    status: response.status,
                    statusText: response.statusText,
                    error: errorText
                });
                throw new Error(`Failed to send message: ${response.status} ${response.statusText}`);
            }

            if (!response.body) {
                console.error('No response body received');
                throw new Error('No response body');
            }

            console.log('Starting to read response stream');
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const {done, value} = await reader.read();
                if (done) {
                    console.log('Response stream complete');
                    break;
                }
                const chunk = decoder.decode(value);
                console.log('Received chunk:', chunk);
                yield chunk;
            }
        } catch (error) {
            console.error('Error in sendMessage:', error);
            throw error;
        }
    },
};

// File API
export const fileApi = {
    uploadFile: async (file: File): Promise<FileMetadata> => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post('/files/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    getFiles: async (): Promise<FileMetadata[]> => {
        const response = await api.get('/files');
        return response.data;
    },

    deleteFile: async (fileId: string): Promise<void> => {
        await api.delete(`/files/${fileId}`);
    },
};

// Model Configuration API
export const modelApi = {
    validateConfig: async (config: ModelConfig): Promise<{ valid: boolean, issues?: string[] }> => {
        try {
            const response = await api.post('/model/validate', config);
            return response.data;  // Return the full validation response
        } catch (error) {
            return {valid: false, issues: [error.message]};
        }
    },

    saveConfig: async (config: ModelConfig): Promise<void> => {
        await api.post('/model/config', config);
    },

    getConfig: async (): Promise<ModelConfig[]> => {
        const response = await api.get('/model/config');
        return response.data;
    },
};