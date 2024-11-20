// src/lib/api.ts

import axios from 'axios';

import type {Chat, FileMetadata, ModelConfig} from './types';

// API Configuration
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
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
        // More robust error handling
        const errorDetails = {
            url: error.config?.url || 'unknown',
            method: error.config?.method || 'unknown',
            status: error.response?.status || 'connection failed',
            data: error.response?.data || null,
            message: error.message || 'Unknown error'
        };

        // Only log if there are actual error details
        if (errorDetails.status !== 'connection failed' && errorDetails.message !== 'Unknown error') {
            console.error('API Error:', errorDetails);
        }
        // Add specific connection error handling
        if (errorDetails.status === 'connection failed') {
            console.warn('Backend connection failed. Please ensure the backend server is running at:', API_BASE_URL);
        }

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
    uploadFile: async (file: File, modelConfig: ModelConfig): Promise<FileMetadata> => {
        const formData = new FormData();
        formData.append('file', file);
        // Send modelConfig as stringified JSON in the model_config field
        formData.append('model_config_json', JSON.stringify(modelConfig));

        const response = await api.post('/files/', formData, {  // Note the trailing slash
            headers: {
                'Content-Type': 'multipart/form-data',
            }
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