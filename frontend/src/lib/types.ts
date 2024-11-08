export type Provider = 'claude' | 'chatgpt' | 'ollama';

export type ClaudeModel = 'claude-3-opus' | 'claude-3-sonnet' | 'claude-3-haiku';
export type ChatGPTModel = 'gpt-4' | 'gpt-4-0125-preview' | 'gpt-3.5-turbo';

export interface ModelConfig {
    provider: Provider;
    apiKey?: string;
    model: ClaudeModel | ChatGPTModel | string;
    ollamaModel?: string;
    temperature: number;
}