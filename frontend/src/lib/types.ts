// Updated types.ts

export type Provider = 'claude' | 'chatgpt' | 'ollama';

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

export interface ModelConfig {
    provider: Provider;
    apiKey?: string;
    model: ClaudeModel | ChatGPTModel | string;
    temperature: number;
    systemMessage?: string;
}