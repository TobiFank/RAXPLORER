// src/lib/types.ts

export type Provider = 'claude' | 'chatgpt' | 'ollama';

// Matches current backend expectation
export interface ModelConfig {
    provider: Provider;
    apiKey?: string;
    model: string;
    embeddingModel: string;
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
    status?: 'processing' | 'complete';
}

export interface EditState {
    chatId: string;
    title: string;
}

export interface ModelConfigError {
    message: string;
    details?: string[];
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
        ] as const,
        embeddingModels: [
            ['SOME_CLAUDE_EMBEDDING', 'Claude Embedding'],
        ] as const
    },
    chatgpt: {
        defaultModel: 'gpt-3.5-turbo-0125',
        defaultEmbedding: 'text-embedding-3-small',
        models: [
            ['gpt-4o', 'GPT-4 Omni'],
            ['gpt-4o-mini', 'GPT-4 Omni Mini'],
            ['gpt-4-turbo', 'GPT-4 Turbo'],
            ['gpt-4-0125-preview', 'GPT-4 Turbo Preview'],
            ['gpt-3.5-turbo-0125', 'GPT-3.5 Turbo']
        ] as const,
        embeddingModels: [
            ['text-embedding-3-large', 'text-embedding-3-large (3,072d)'],
            ['text-embedding-3-small', 'text-embedding-3-small (1,536d)'],
            ['text-embedding-ada-002', 'text-embedding-ada-002 (1,536d)']
        ] as const
    },
    ollama: {
        defaultModel: 'llama3.1',
        defaultEmbedding: 'nomic-embed-text',
        models: [] as const,
        embeddingModels: [] as const
    }
} as const;