// src/lib/hooks/useModelConfig.ts
'use client';

import {useEffect, useState} from "react";
import {modelApi, ModelConfig} from "@/lib/api";
import {Provider} from "@/lib/types";

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
    }
};

const STORAGE_KEY = 'modelConfig';
const ACTIVE_PROVIDER_KEY = 'activeProvider';

// Helper function to safely access localStorage
const getStorageValue = (key: string, defaultValue: any) => {
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
const setStorageValue = (key: string, value: any) => {
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
    const [configs, setConfigs] = useState<Record<Provider, ModelConfig>>(() =>
        getStorageValue(STORAGE_KEY, defaultConfigs)
    );

    const [activeProvider, setActiveProvider] = useState<Provider>(() =>
        getStorageValue(ACTIVE_PROVIDER_KEY, 'claude')
    );

    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<ModelConfigError | null>(null);
    const [isSaving, setIsSaving] = useState(false);

    const modelConfig = configs[activeProvider];

    const loadConfig = async () => {
        try {
            setIsLoading(true);
            const allConfigs = await modelApi.getConfig();
            const newConfigs = {...defaultConfigs};
            allConfigs.forEach((config: ModelConfig) => {
                newConfigs[config.provider as Provider] = config;
            });
            setConfigs(newConfigs);
            setStorageValue(STORAGE_KEY, newConfigs);
            setError(null);
        } catch (err) {
            setError({message: 'Failed to load model configuration'});
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        loadConfig();
    }, []);

    const updateDraft = (updates: Partial<ModelConfig>) => {
        setConfigs(prev => {
            const newConfigs = {
                ...prev,
                [activeProvider]: {...prev[activeProvider], ...updates}
            };
            setStorageValue(STORAGE_KEY, newConfigs);
            return newConfigs;
        });
    };

    const switchProvider = (provider: Provider) => {
        setActiveProvider(provider);
        setStorageValue(ACTIVE_PROVIDER_KEY, provider);
        setError(null);
    };

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
            setConfigs(prev => ({
                ...prev,
                [config.provider]: config
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