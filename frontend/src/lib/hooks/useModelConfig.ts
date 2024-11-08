// src/lib/hooks/useModelConfig.ts
import {useEffect, useState} from "react";
import {modelApi, ModelConfig} from "@/lib/api";

export function useModelConfig() {
    const [modelConfig, setModelConfig] = useState<ModelConfig>({
        provider: 'claude',
        model: 'claude-3-opus',
        temperature: 0.7,
        apiKey: '',
        ollamaModel: '',
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadConfig();
    }, []);

    const loadConfig = async () => {
        try {
            setIsLoading(true);
            const config = await modelApi.getConfig();
            setModelConfig(config);
        } catch (err) {
            setError('Failed to load model configuration');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const updateConfig = async (config: ModelConfig) => {
        try {
            setIsLoading(true);
            await modelApi.saveConfig(config);
            setModelConfig(config);
        } catch (err) {
            setError('Failed to update model configuration');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const validateConfig = async (config: ModelConfig) => {
        try {
            return await modelApi.validateConfig(config);
        } catch (err) {
            setError('Failed to validate model configuration');
            console.error(err);
            return false;
        }
    };

    return {
        modelConfig,
        isLoading,
        error,
        updateConfig,
        validateConfig,
    };
}