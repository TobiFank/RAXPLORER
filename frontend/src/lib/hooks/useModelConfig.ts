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
        ollamaModel: '',
    }
};

export interface ModelConfigError {
    message: string;
    details?: string[];
}

export function useModelConfig() {
    const [configs, setConfigs] = useState<Record<Provider, ModelConfig>>(defaultConfigs);
    const [activeProvider, setActiveProvider] = useState<Provider>('claude');
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

    const saveConfig = async (config: ModelConfig): Promise<boolean> => {
        try {
            setIsSaving(true);
            setError(null);

            // First validate
            const validationResponse = await modelApi.validateConfig(config);
            if (!validationResponse.valid) {
                setError({
                    message: 'Configuration validation failed',
                    details: validationResponse.issues
                });
                return false;
            }

            // Then save if valid
            await modelApi.saveConfig(config);
            setConfigs(prev => ({
                ...prev,
                [config.provider]: config
            }));
            return true;
        } catch (err) {
            setError({
                message: 'Failed to save configuration',
                details: err.response?.data?.issues || [err.message]
            });
            return false;
        } finally {
            setIsSaving(false);
        }
    };

    const updateDraft = (updates: Partial<ModelConfig>) => {
        setConfigs(prev => ({
            ...prev,
            [activeProvider]: {...prev[activeProvider], ...updates}
        }));
    };

    const switchProvider = (provider: Provider) => {
        setActiveProvider(provider);
        setError(null);
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