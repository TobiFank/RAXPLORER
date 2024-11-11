// src/lib/types.ts

export type Provider = 'claude' | 'chatgpt' | 'ollama';

// These types are for frontend validation only
export type ClaudeModel =
    | 'claude-3-opus-20240229'
    | 'claude-3-sonnet-20240229'
    | 'claude-3-haiku-20240307'
    | 'claude-3-5-sonnet-20241022'
    | 'claude-3-5-haiku-20241022';

export type ChatGPTModel =
    | 'gpt-4o'
    | 'gpt-4o-mini'
    | 'gpt-4-turbo'
    | 'gpt-4-0125-preview'
    | 'gpt-3.5-turbo-0125';

// Matches current backend expectation
export interface ModelConfig {
    provider: Provider;
    apiKey?: string;
    model: string;  // Keeping as string for backend compatibility
    temperature: number;
    systemMessage?: string;
}

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

export interface EditState {
    chatId: string;
    title: string;
}

// Frontend-only model information
export const MODEL_INFORMATION = {
    claude: {
        defaultModel: 'claude-3-opus-20240229',
        models: [
            ['claude-3-opus-20240229', 'Claude 3 Opus'],
            ['claude-3-sonnet-20240229', 'Claude 3 Sonnet'],
            ['claude-3-haiku-20240307', 'Claude 3 Haiku'],
            ['claude-3-5-sonnet-20241022', 'Claude 3.5 Sonnet'],
            ['claude-3-5-haiku-20241022', 'Claude 3.5 Haiku']
        ] as const
    },
    chatgpt: {
        defaultModel: 'gpt-4o',
        models: [
            ['gpt-4o', 'GPT-4 Omni'],
            ['gpt-4o-mini', 'GPT-4 Omni Mini'],
            ['gpt-4-turbo', 'GPT-4 Turbo'],
            ['gpt-4-0125-preview', 'GPT-4 Turbo Preview'],
            ['gpt-3.5-turbo-0125', 'GPT-3.5 Turbo']
        ] as const
    },
    ollama: {
        defaultModel: '',
        models: [] as const
    }
} as const;