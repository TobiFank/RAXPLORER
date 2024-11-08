// src/lib/hooks/useModelConfig.ts
import { useEffect, useState } from "react";
import { modelApi, ModelConfig } from "@/lib/api";
import { Provider, ClaudeModel, ChatGPTModel } from "@/lib/types";

const defaultConfigs: Record<Provider, ModelConfig> = {
    claude: {
        provider: 'claude',
        model: 'claude-3-opus',
        temperature: 0.7,
        apiKey: '',
    },
    chatgpt: {
        provider: 'chatgpt',
        model: 'gpt-4',
        temperature: 0.7,
        apiKey: '',
    },
    ollama: {
        provider: 'ollama',
        model: '',
        temperature: 0.7,
        ollamaModel: '',
    }
};

export function useModelConfig() {
    // Track configs for all providers
    const [configs, setConfigs] = useState<Record<Provider, ModelConfig>>(defaultConfigs);
    // Track active provider
    const [activeProvider, setActiveProvider] = useState<Provider>('claude');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Get current active config - this maintains the existing API surface
    const modelConfig = configs[activeProvider];

    // Load configs for all providers
    const loadConfig = async () => {
        try {
            setIsLoading(true);
            const allConfigs = await modelApi.getConfig();

            // Convert array of configs to record
            const newConfigs = { ...defaultConfigs };
            allConfigs.forEach((config: ModelConfig) => {
                newConfigs[config.provider as Provider] = config;
            });

            setConfigs(newConfigs);
        } catch (err) {
            setError('Failed to load model configuration');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Load initial configs
    useEffect(() => {
        loadConfig();
    }, []);

    // Update config for specific provider
    const updateConfig = async (config: ModelConfig) => {
        try {
            setIsLoading(true);
            await modelApi.saveConfig(config);
            setConfigs(prev => ({
                ...prev,
                [config.provider]: config
            }));
        } catch (err) {
            setError('Failed to update model configuration');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Validate config for specific provider
    const validateConfig = async (config: ModelConfig) => {
        try {
            return await modelApi.validateConfig(config);
        } catch (err) {
            setError('Failed to validate model configuration');
            console.error(err);
            return false;
        }
    };

    // Switch active provider
    const switchProvider = (provider: Provider) => {
        setActiveProvider(provider);
    };

    return {
        modelConfig, // Current active config - maintains existing API surface
        configs, // All provider configs
        activeProvider,
        isLoading,
        error,
        updateConfig,
        validateConfig,
        switchProvider,
    };
}