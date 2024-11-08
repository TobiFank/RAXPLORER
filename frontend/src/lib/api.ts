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
    ollamaModel?: string;
    temperature: number;
}

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_V1_PREFIX = '/api/v1';

const api = axios.create({
    baseURL: `${API_BASE_URL}${API_V1_PREFIX}`,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Chat API
export const chatApi = {
    // Chat Management endpoints remain the same
    createChat: async (): Promise<Chat> => {
        const response = await api.post('/chat/chats');
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
        const response = await api.patch(`/chat/chats/${chatId}`, { title });
        return response.data;
    },

    // Single message endpoint that handles streaming
    sendMessage: async function* (chatId: string, content: string, modelConfig: ModelConfig) {
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

        if (!response.body) throw new Error('No response body');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            yield decoder.decode(value);
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
    validateConfig: async (config: ModelConfig): Promise<boolean> => {
        try {
            const response = await api.post('/model/validate', config);
            return response.data.valid;
        } catch (error) {
            return false;
        }
    },

    saveConfig: async (config: ModelConfig): Promise<void> => {
        await api.post('/model/config', config);
    },

    getConfig: async (): Promise<ModelConfig> => {
        const response = await api.get('/model/config');
        return response.data;
    },
};