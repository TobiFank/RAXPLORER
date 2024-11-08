'use client';

import React, {useCallback, useState} from 'react';
import {ChevronDown, FileText, Plus, Send, Settings, Trash2} from 'lucide-react';
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {Textarea} from '@/components/ui/textarea';
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from '@/components/ui/select';
import {Input} from '@/components/ui/input';
import {Alert} from '@/components/ui/alert';
import {useChat} from "@/lib/hooks/useChat";
import {useFiles} from "@/lib/hooks/useFiles";
import {useModelConfig} from "@/lib/hooks/useModelConfig";

const ChatInterface = () => {
    // Custom hooks for real functionality
    const {
        chats,
        activeChat,
        messages,
        isLoading: chatLoading,
        error: chatError,
        createChat,
        deleteChat,
        switchChat,
        sendMessage
    } = useChat();

    const {
        files,
        isLoading: filesLoading,
        error: filesError,
        uploadFile,
        deleteFile
    } = useFiles();

    const {
        modelConfig,
        isLoading: configLoading,
        error: configError,
        updateConfig,
        validateConfig
    } = useModelConfig();

    // Local UI state
    const [inputText, setInputText] = useState('');
    const [isSettingsExpanded, setIsSettingsExpanded] = useState(true);
    const [isDragging, setIsDragging] = useState(false);
    const [isModelValid, setIsModelValid] = useState(true);

    // File handling functions
    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const droppedFiles = Array.from(e.dataTransfer.files);
        for (const file of droppedFiles) {
            await uploadFile(file);
        }
    };

    // Message handling
    const handleSendMessage = async () => {
        if (!inputText.trim() || !validateModelSettings()) return;

        await sendMessage(inputText, modelConfig);
        setInputText('');
    };

    // Model validation
    const validateModelSettings = useCallback(() => {
        if (modelConfig.provider === 'ollama') {
            return modelConfig.ollamaModel?.trim() !== '';
        } else {
            return modelConfig.apiKey?.trim() !== '';
        }
    }, [modelConfig]);

    // Update model config
    const handleModelConfigUpdate = async (updates: Partial<typeof modelConfig>) => {
        const newConfig = {...modelConfig, ...updates};
        const isValid = await validateConfig(newConfig);
        setIsModelValid(isValid);
        if (isValid) {
            await updateConfig(newConfig);
        }
    };

    return (
        <div className="flex h-screen bg-gray-50">
            {/* Sidebar */}
            <div className="w-80 p-4 border-r bg-white overflow-y-auto">
                {/* Chat History */}
                <Button
                    className="w-full mb-4 flex items-center gap-2"
                    onClick={createChat}
                    disabled={chatLoading}
                >
                    <Plus size={16}/>
                    New Chat
                </Button>

                <div className="space-y-2 mb-4">
                    {chats.map((chat) => (
                        <div
                            key={chat.id}
                            className={`group flex items-center justify-between p-2 rounded-lg hover:bg-gray-100 cursor-pointer ${
                                activeChat === chat.id ? 'bg-gray-100' : ''
                            }`}
                            onClick={() => switchChat(chat.id)}
                        >
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate">
                                    {chat.title}
                                </p>
                                <p className="text-xs text-gray-500">
                                    {new Date(chat.createdAt).toLocaleDateString()}
                                </p>
                            </div>
                            {chats.length > 1 && (
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        deleteChat(chat.id);
                                    }}
                                >
                                    <Trash2 size={14}/>
                                </Button>
                            )}
                        </div>
                    ))}
                </div>

                {/* Model Settings */}
                <Card className="mb-4">
                    <CardHeader
                        className="cursor-pointer hover:bg-gray-50"
                        onClick={() => setIsSettingsExpanded(!isSettingsExpanded)}
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Settings size={16}/>
                                <CardTitle className="text-sm">Model Settings</CardTitle>
                            </div>
                            <ChevronDown
                                size={16}
                                className={`transition-transform ${isSettingsExpanded ? 'rotate-180' : ''}`}
                            />
                        </div>
                    </CardHeader>

                    {isSettingsExpanded && (
                        <CardContent className="space-y-4">
                            {/* Provider Selection */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Provider</label>
                                <Select
                                    value={modelConfig.provider}
                                    onValueChange={(value: 'claude' | 'chatgpt' | 'ollama') => handleModelConfigUpdate({
                                        provider: value,
                                        model: value === 'claude' ? 'claude-3-opus' :
                                            value === 'chatgpt' ? 'gpt-4' : '',
                                        apiKey: '', // Reset API key when changing provider
                                        ollamaModel: value === 'ollama' ? modelConfig.ollamaModel : undefined,
                                        temperature: modelConfig.temperature // Preserve temperature
                                    })}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select Provider"/>
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="claude">Claude</SelectItem>
                                        <SelectItem value="chatgpt">ChatGPT</SelectItem>
                                        <SelectItem value="ollama">Ollama (Local)</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            {/* API Settings */}
                            {(modelConfig.provider === 'claude' || modelConfig.provider === 'chatgpt') && (
                                <>
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">API Key</label>
                                        <Input
                                            type="password"
                                            placeholder={`Enter your ${modelConfig.provider === 'claude' ? 'Anthropic' : 'OpenAI'} API key`}
                                            value={modelConfig.apiKey || ''}
                                            onChange={(e) => handleModelConfigUpdate({
                                                ...modelConfig,
                                                apiKey: e.target.value
                                            })}
                                        />
                                        <p className="text-xs text-gray-500">
                                            {modelConfig.provider === 'claude'
                                                ? 'Get your API key from console.anthropic.com'
                                                : 'Get your API key from platform.openai.com'}
                                        </p>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-sm font-medium">Model</label>
                                        <Select
                                            value={modelConfig.model}
                                            onValueChange={(value) => handleModelConfigUpdate({
                                                ...modelConfig,
                                                model: value
                                            })}
                                        >
                                            <SelectTrigger>
                                                <SelectValue placeholder="Select Model"/>
                                            </SelectTrigger>
                                            <SelectContent>
                                                {modelConfig.provider === 'claude' ? (
                                                    <>
                                                        <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                                                        <SelectItem value="claude-3-sonnet">Claude 3 Sonnet</SelectItem>
                                                        <SelectItem value="claude-3-haiku">Claude 3 Haiku</SelectItem>
                                                    </>
                                                ) : (
                                                    <>
                                                        <SelectItem value="gpt-4">GPT-4 Turbo</SelectItem>
                                                        <SelectItem value="gpt-4-0125-preview">GPT-4
                                                            Preview</SelectItem>
                                                        <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
                                                    </>
                                                )}
                                            </SelectContent>
                                        </Select>
                                    </div>
                                </>
                            )}

                            {/* Ollama Settings */}
                            {modelConfig.provider === 'ollama' && (
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Model Name</label>
                                    <Input
                                        placeholder="e.g., llama2, mistral, codellama"
                                        value={modelConfig.ollamaModel || ''}
                                        onChange={(e) => handleModelConfigUpdate({
                                            ollamaModel: e.target.value
                                        })}
                                    />
                                    <p className="text-xs text-gray-500">
                                        Enter the name of your locally installed Ollama model
                                    </p>
                                </div>
                            )}

                            {/* Temperature Setting */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Temperature</label>
                                <div className="flex items-center gap-4">
                                    <input
                                        type="range"
                                        min="0"
                                        max="2"
                                        step="0.1"
                                        value={modelConfig.temperature}
                                        onChange={(e) => handleModelConfigUpdate({
                                            temperature: parseFloat(e.target.value)
                                        })}
                                        className="flex-1"
                                    />
                                    <span className="text-sm w-12 text-right">
                                        {modelConfig.temperature.toFixed(1)}
                                    </span>
                                </div>
                                <p className="text-xs text-gray-500">
                                    Controls randomness: 0 is focused, 2 is more creative
                                </p>
                            </div>

                            {/* Validation Alert */}
                            {!isModelValid && (
                                <Alert variant="destructive" className="mt-4">
                                    {modelConfig.provider === 'ollama'
                                        ? 'Please enter a valid model name'
                                        : 'Please enter your API key'}
                                </Alert>
                            )}
                        </CardContent>
                    )}
                </Card>

                {/* File Upload Area */}
                <div className="mt-4 space-y-4">
                    <h3 className="text-sm font-medium">Documents</h3>
                    <div
                        className={`border-2 border-dashed rounded-lg p-4 transition-colors ${
                            isDragging
                                ? 'border-blue-500 bg-blue-50'
                                : 'border-gray-300 hover:border-blue-400'
                        }`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="text-center">
                            <FileText className="mx-auto h-8 w-8 text-gray-400"/>
                            <p className="mt-2 text-sm text-gray-500">
                                Drag and drop files here or
                                <label className="mx-1 text-blue-500 hover:text-blue-600 cursor-pointer">
                                    browse
                                    <input
                                        type="file"
                                        className="hidden"
                                        onChange={(e) => {
                                            const file = e.target.files?.[0];
                                            if (file) uploadFile(file);
                                        }}
                                        multiple
                                    />
                                </label>
                            </p>
                            <p className="text-xs text-gray-400">
                                PDF, DOC, DOCX, Images accepted
                            </p>
                        </div>
                    </div>

                    {/* File List */}
                    <div className="space-y-2">
                        {files.map((file) => (
                            <div
                                key={file.id}
                                className="p-3 bg-gray-50 rounded-lg flex items-start justify-between group"
                            >
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium truncate">{file.name}</p>
                                    <p className="text-xs text-gray-500">
                                        {file.size} • {file.pages} pages
                                    </p>
                                </div>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                                    onClick={() => deleteFile(file.id)}
                                >
                                    <Trash2 size={14}/>
                                </Button>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                {/* Chat Messages */}
                <div className="flex-1 p-6 overflow-y-auto space-y-6">
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div
                                className={`p-4 rounded-lg max-w-[80%] ${
                                    message.role === 'user'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-gray-100 text-gray-900'
                                }`}
                            >
                                <div className="whitespace-pre-wrap">{message.content}</div>
                                <div className={`text-xs mt-2 ${
                                    message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                                }`}>
                                    {new Date(message.timestamp).toLocaleTimeString()}
                                </div>
                            </div>
                        </div>
                    ))}
                    {chatLoading && (
                        <div className="flex justify-start">
                            <div className="p-4 rounded-lg bg-gray-100">
                                <div className="animate-pulse flex space-x-2">
                                    <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                                    <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                                    <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="p-4 border-t bg-white">
                    {chatError && (
                        <Alert variant="destructive" className="mb-4">
                            {chatError}
                        </Alert>
                    )}
                    <div className="flex gap-2">
                        <Textarea
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            placeholder="Type your message..."
                            className="flex-1"
                            rows={1}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSendMessage();
                                }
                            }}
                            disabled={chatLoading || !isModelValid}
                        />
                        <Button
                            onClick={handleSendMessage}
                            disabled={
                                chatLoading ||
                                !validateModelSettings() ||
                                !inputText.trim()
                            }
                            className="px-4"
                        >
                            <Send size={16}/>
                        </Button>
                    </div>
                    {!validateModelSettings() && (
                        <p className="text-xs text-red-500 mt-2">
                            Please configure model settings before sending messages
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;