// src/lib/hooks/useModelConfig.ts
'use client';

import {useEffect, useState, useCallback} from "react";
import {modelApi, ModelConfig} from "@/lib/api";
import {Provider} from "@/lib/types";

const defaultConfigs: Record<Provider, ModelConfig> = {
    claude: {
        provider: 'claude',
        model: 'claude-3-opus-20240229',
        temperature: 0.7,
        apiKey: '',
        systemMessage: '',
    },
    chatgpt: {
        provider: 'chatgpt',
        model: 'gpt-4o',
        temperature: 0.7,
        apiKey: '',
        systemMessage: '',
    },
    ollama: {
        provider: 'ollama',
        model: '',
        temperature: 0.7,
        systemMessage: '',
    }
};

const STORAGE_KEY = 'modelConfigs';  // Changed to be more explicit
const ACTIVE_PROVIDER_KEY = 'activeProvider';

// Helper function to safely access localStorage
const getStorageValue = <T>(key: string, defaultValue: T): T => {
    if (typeof window === 'undefined') {
        return defaultValue;
    }
    try {
        const item = window.localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.warn(`Error reading localStorage key "${key}":`, error);
        return defaultValue;
    }
};

// Helper function to safely set localStorage
const setStorageValue = <T>(key: string, value: T): void => {
    if (typeof window !== 'undefined') {
        try {
            window.localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.warn(`Error setting localStorage key "${key}":`, error);
        }
    }
};

export interface ModelConfigError {
    message: string;
    details?: string[];
}

export function useModelConfig() {
    // Store configurations for all providers
    const [configs, setConfigs] = useState<Record<Provider, ModelConfig>>(() => {
        const savedConfigs = getStorageValue<Record<Provider, ModelConfig>>(STORAGE_KEY, defaultConfigs);
        // Ensure all providers have configurations by merging with defaults
        return Object.entries(defaultConfigs).reduce((acc, [provider, defaultConfig]) => ({
            ...acc,
            [provider]: {
                ...defaultConfig,
                ...(savedConfigs[provider as Provider] || {}),
                provider: provider as Provider, // Ensure provider is always correct
            }
        }), {} as Record<Provider, ModelConfig>);
    });

    const [activeProvider, setActiveProvider] = useState<Provider>(() =>
        getStorageValue<Provider>(ACTIVE_PROVIDER_KEY, 'claude')
    );

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<ModelConfigError | null>(null);
    const [isSaving, setIsSaving] = useState(false);

    // Get the current model config for active provider
    const modelConfig = configs[activeProvider];

    // Load configurations from backend
    const loadConfig = useCallback(async () => {
        try {
            setIsLoading(true);
            const allConfigs = await modelApi.getConfig();

            // Merge saved configs with defaults while preserving provider-specific configs
            const newConfigs = Object.entries(defaultConfigs).reduce((acc, [provider, defaultConfig]) => {
                const savedConfig = allConfigs.find(c => c.provider === provider);
                return {
                    ...acc,
                    [provider]: {
                        ...defaultConfig,
                        ...(savedConfig || {}),
                        provider: provider as Provider,
                    }
                };
            }, {} as Record<Provider, ModelConfig>);

            setConfigs(newConfigs);
            setStorageValue(STORAGE_KEY, newConfigs);
            setError(null);
        } catch (err) {
            setError({message: 'Failed to load model configuration'});
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadConfig();
    }, [loadConfig]);

    // Update configuration for a specific provider
    const updateDraft = useCallback((updates: Partial<ModelConfig>) => {
        setConfigs(prev => {
            // Only update the specific provider's config
            const targetProvider = updates.provider || activeProvider;
            const newConfigs = {
                ...prev,
                [targetProvider]: {
                    ...prev[targetProvider],
                    ...updates,
                    provider: targetProvider, // Ensure provider is always set correctly
                }
            };
            setStorageValue(STORAGE_KEY, newConfigs);
            return newConfigs;
        });
    }, [activeProvider]);

    // Switch active provider without modifying configs
    const switchProvider = useCallback((provider: Provider) => {
        setActiveProvider(provider);
        setStorageValue(ACTIVE_PROVIDER_KEY, provider);
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

            // Update only the specific provider's config
            setConfigs(prev => ({
                ...prev,
                [config.provider]: {
                    ...prev[config.provider],
                    ...config,
                    provider: config.provider,
                }
            }));

            setStorageValue(STORAGE_KEY, configs);
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