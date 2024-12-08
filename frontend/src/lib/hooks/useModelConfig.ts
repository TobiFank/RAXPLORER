'use client';
// src/lib/hooks/useModelConfig.ts

import {useEffect, useState, useCallback} from "react";
import {modelApi} from "@/lib/api";
import {Provider, ModelConfig, MODEL_INFORMATION, ModelConfigError} from '@/lib/types';

const defaultConfigs: Record<Provider, ModelConfig> = {
    claude: {
        provider: 'claude',
        model: MODEL_INFORMATION.claude.defaultModel,
        embeddingModel: MODEL_INFORMATION.claude.defaultEmbedding,
        temperature: 0.3,
        apiKey: '',
        systemMessage: '',
    },
    chatgpt: {
        provider: 'chatgpt',
        model: MODEL_INFORMATION.chatgpt.defaultModel,
        embeddingModel: MODEL_INFORMATION.chatgpt.defaultEmbedding,
        temperature: 0.3,
        apiKey: '',
        systemMessage: '',
    },
    ollama: {
        provider: 'ollama',
        model: MODEL_INFORMATION.ollama.defaultModel,
        embeddingModel: MODEL_INFORMATION.ollama.defaultEmbedding,
        temperature: 0.3,
        systemMessage: '',
    }
};

const ACTIVE_PROVIDER_KEY = 'activeProvider';
const CONFIG_STORAGE_KEY = 'modelConfig';
const INITIAL_PROVIDER = 'ollama';

export function useModelConfig() {
    const [configs, setConfigs] = useState<Record<Provider, ModelConfig>>(defaultConfigs);

    useEffect(() => {
        const saved = window.localStorage.getItem(CONFIG_STORAGE_KEY);
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                setConfigs(parsed.configs || defaultConfigs);
            } catch (e) {
                console.error('Failed to parse stored config:', e);
            }
        }
    }, []);

    const [activeProvider, setActiveProvider] = useState<Provider>(INITIAL_PROVIDER);

    useEffect(() => {
        const saved = window.localStorage.getItem(CONFIG_STORAGE_KEY);
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                setActiveProvider(parsed.activeProvider || INITIAL_PROVIDER);
            } catch (e) {
                console.error('Failed to parse stored config:', e);
            }
        }
    }, []);

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<ModelConfigError | null>(null);
    const [isSaving, setIsSaving] = useState(false);

    const modelConfig = configs[activeProvider];

    // Load configurations from backend
    const loadConfig = useCallback(async () => {
        try {
            setIsLoading(true);
            const allConfigs = await modelApi.getConfig();

            // Convert array of configs to Record<Provider, ModelConfig>
            const newConfigs = {...defaultConfigs};
            allConfigs.forEach(config => {
                if (config.provider) {
                    newConfigs[config.provider] = {
                        ...defaultConfigs[config.provider],
                        ...config
                    };
                }
            });

            setConfigs(newConfigs);
            setError(null);
        } catch (err) {
            console.error('Failed to load model configuration:', err);
            setError({message: 'Failed to load model configuration'});
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadConfig();
    }, [loadConfig]);

    const updateDraft = useCallback((updates: Partial<ModelConfig>) => {
        setConfigs(prev => {
            const targetProvider = updates.provider || activeProvider;
            return {
                ...prev,
                [targetProvider]: {
                    ...prev[targetProvider],
                    ...updates,
                    provider: targetProvider,
                }
            };
        });
    }, [activeProvider]);

    const switchProvider = useCallback((provider: Provider) => {
        setActiveProvider(provider);
        if (typeof window !== 'undefined') {
            window.localStorage.setItem(ACTIVE_PROVIDER_KEY, JSON.stringify(provider));
        }
        setError(null);
    }, []);

    const saveConfig = async (config: ModelConfig): Promise<boolean> => {
        try {
            setIsSaving(true);
            setError(null);

            const validationResponse = await modelApi.validateConfig(config);
            if (!validationResponse.valid) {
                setError({
                    message: 'Configuration validation failed',
                    details: validationResponse.issues || []
                });
                return false;
            }

            await modelApi.saveConfig(config);

            const newConfigs = {
                ...configs,
                [config.provider]: config
            };
            setConfigs(newConfigs);

            // Persist to localStorage
            window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify({
                configs: newConfigs,
                activeProvider
            }));

            return true;
        } catch (err) {
            console.error('Save config error:', err);
            setError({
                message: 'Failed to save configuration',
                details: err.response?.data?.issues || [err.message]
            });
            return false;
        } finally {
            setIsSaving(false);
        }
    };

    return {
        modelConfig,
        configs,
        activeProvider,
        isLoading,
        isSaving,
        error,
        saveConfig,
        updateDraft,
        switchProvider,
    };
}