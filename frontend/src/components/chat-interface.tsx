'use client';
// src/components/chat-interface.tsx

import React, {useCallback, useEffect, useRef, useState} from 'react';
import {ChevronDown, FileText, Link2, Pencil, Plus, Send, Settings, Trash2} from 'lucide-react';
import {Card, CardContent, CardHeader, CardTitle} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {Textarea} from '@/components/ui/textarea';
import {Select, SelectContent, SelectItem, SelectTrigger, SelectValue} from '@/components/ui/select';
import {Input} from '@/components/ui/input';
import {Alert, AlertDescription, AlertTitle} from '@/components/ui/alert';
import {useChat} from "@/lib/hooks/useChat";
import {useFiles} from "@/lib/hooks/useFiles";
import {useModelConfig} from "@/lib/hooks/useModelConfig";
import {Chat, MODEL_INFORMATION, Provider} from "@/lib/types";

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
        sendMessage,
        updateChatTitle
    } = useChat();

    const {
        files,
        uploadFile,
        deleteFile
    } = useFiles();

    const {
        modelConfig,
        isSaving,
        error: configError,
        switchProvider,
        updateDraft,
        saveConfig
    } = useModelConfig();

    // Local UI state
    const [inputText, setInputText] = useState('');
    const [isSettingsExpanded, setIsSettingsExpanded] = useState(true);
    const [isDragging, setIsDragging] = useState(false);
    const [editingChatId, setEditingChatId] = useState<string | null>(null);
    const [editingTitle, setEditingTitle] = useState<string>('');
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const messagesContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (messagesContainerRef.current) {
            messagesContainerRef.current.scrollTo({
                top: messagesContainerRef.current.scrollHeight,
                behavior: 'smooth'
            });
        }
    }, [messages]);

    useEffect(() => {
        // Focus input when component mounts
        inputRef.current?.focus();
    }, []);

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
        droppedFiles.forEach(file => uploadFile(file, modelConfig));
    };

    const handleTitleSubmit = async (chatId: string) => {
        if (editingTitle.trim() && editingTitle !== 'New Chat') {
            try {
                await updateChatTitle(chatId, editingTitle.trim());
                // Only clear editing state after successful update
                setEditingChatId(null);
                setEditingTitle('');
            } catch (error) {
                console.error('Failed to update chat title:', error);
                // On error, keep the editing state active
            }
        }
    };

    const startEditing = (chat: Chat, e: React.MouseEvent) => {
        e.stopPropagation();
        setEditingChatId(chat.id);
        setEditingTitle(chat.title);
    };

    // Message handling
    const handleSendMessage = async () => {
        console.log("Send button clicked");
        console.log("Input text:", inputText);
        console.log("Current model config:", modelConfig);
        console.log("Chat loading state:", chatLoading);
        console.log("Model settings valid:", validateModelSettings());

        if (!inputText.trim()) {
            console.log("Input text is empty, returning");
            return;
        }

        try {
            console.log("Attempting to send message...");
            const currentInput = inputText;
            setInputText(''); // Clear input before sending
            await sendMessage(currentInput, modelConfig);
            console.log("Message sent successfully");

            // Use setTimeout to ensure focus happens after state updates
            setTimeout(() => {
                inputRef.current?.focus();
            }, 0);
        } catch (error) {
            console.error("Error sending message:", error);
        }
    };


    // Model validation
    const validateModelSettings = useCallback(() => {
        const isValid = modelConfig.provider === 'ollama'
            ? modelConfig.model?.trim() !== ''
            : modelConfig.apiKey?.trim() !== '';

        return isValid;
    }, [modelConfig]);

    const handleNewChat = async () => {
        console.log("Creating new chat");
        try {
            await createChat();
            console.log("Chat created successfully");
        } catch (error) {
            console.error("Failed to create chat:", error);
        }
    };

    const MessageContent = ({content}: { content: string }) => {
        const [mainContent, references] = content.split('References:', 2);
        const imageRegex = /\[IMAGE:(storage\/images\/[^|]+)\|([^|]*)\|([^\]]+)\]/g;
        const images = Array.from(content.matchAll(imageRegex));
        const cleanContent = mainContent.replace(imageRegex, '').trim();

        let transformedReferences = references ? references.trim() : '';

        if (transformedReferences) {
            // Replace document links
            transformedReferences = transformedReferences.replace(
                /\[View Document\]\((.*?)\)/g,
                (match, p1) =>
                    `<a href="${p1}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:text-blue-700">[View Document]</a>`
            );

            // Replace image references
            transformedReferences = transformedReferences.replace(
                imageRegex,
                (match, imagePath, altText) => {
                    // Remove or replace newlines within altText with a space
                    const sanitizedAltText = altText.replace(/\n+/g, ' ').trim();
                    const imageUrl = `${process.env.NEXT_PUBLIC_API_URL}/${imagePath}`;
                    const linkText = sanitizedAltText && sanitizedAltText !== ''
                        ? sanitizedAltText
                        : '[View Image]';
                    return `<a href="${imageUrl}" target="_blank" rel="noopener noreferrer" class="text-blue-500 hover:text-blue-700">${linkText}</a>`;
                }
            );

            // Collapse multiple newlines into one
            transformedReferences = transformedReferences.replace(/\n{2,}/g, '\n');

            // Convert single newlines to <br/>
            transformedReferences = transformedReferences.replace(/\n/g, '<br/>');
        }

        return (
            <>
                <div className="whitespace-pre-wrap">{cleanContent}</div>

                {images.length > 0 && (
                    <div className="grid grid-cols-2 gap-4 my-4">
                        {images.map((img, i) => (
                            <div key={i} className="flex flex-col items-center">
                                <a
                                    href={`${process.env.NEXT_PUBLIC_API_URL}/${img[1]}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <img
                                        src={`${process.env.NEXT_PUBLIC_API_URL}/${img[1]}`}
                                        alt={img[2]}
                                        className="rounded-lg shadow-md max-w-full h-auto"
                                    />
                                </a>
                                {img[2] && (
                                    <p className="text-sm text-gray-600 mt-2 text-center">
                                        {img[2]}
                                    </p>
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {transformedReferences && (
                    <div className="mt-4">
                        <strong>References:</strong>
                        <div
                            className="whitespace-pre-wrap"
                            dangerouslySetInnerHTML={{__html: transformedReferences}}
                        />
                    </div>
                )}
            </>
        );
    };

    return (
        <div className="flex h-screen bg-gray-50">
            {/* Sidebar */}
            <div className="w-80 p-4 border-r bg-white overflow-y-auto">
                {/* Chat History */}
                <Button
                    className="w-full mb-4 flex items-center gap-2"
                    onClick={handleNewChat}
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
                            onClick={() => editingChatId !== chat.id && switchChat(chat.id)}
                        >
                            <div className="flex-1 min-w-0">
                                {editingChatId === chat.id ? (
                                    <Input
                                        value={editingTitle}
                                        onChange={(e) => setEditingTitle(e.target.value)}
                                        onKeyDown={async (e) => {
                                            if (e.key === 'Enter') {
                                                e.preventDefault();
                                                await handleTitleSubmit(chat.id);
                                            } else if (e.key === 'Escape') {
                                                setEditingChatId(null);
                                                setEditingTitle('');
                                            }
                                        }}
                                        autoFocus
                                        className="h-6 text-sm"
                                        onClick={(e) => e.stopPropagation()}
                                    />
                                ) : (
                                    <>
                                        <p className="text-sm font-medium truncate">
                                            {chat.title}
                                        </p>
                                        <p className="text-xs text-gray-500">
                                            {new Date(chat.createdAt).toLocaleDateString()}
                                        </p>
                                    </>
                                )}
                            </div>
                            <div className="flex gap-1">
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                                    onClick={(e) => startEditing(chat, e)}
                                >
                                    <Pencil size={14}/>
                                </Button>
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
                            <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">
                    {MODEL_INFORMATION[modelConfig.provider].models.length > 0
                        ? MODEL_INFORMATION[modelConfig.provider].models.find(([value]) => value === modelConfig.model)?.[1]
                        : modelConfig.model || 'Not configured'}
                </span>
                                <ChevronDown
                                    size={16}
                                    className={`transition-transform ${isSettingsExpanded ? 'rotate-180' : ''}`}
                                />
                            </div>
                        </div>
                    </CardHeader>

                    {isSettingsExpanded && (
                        <CardContent className="space-y-4">
                            {/* Provider Selection */}
                            <Select
                                value={modelConfig.provider}
                                onValueChange={(value: Provider) => {
                                    switchProvider(value);
                                }}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="Select Provider"/>
                                </SelectTrigger>
                                <SelectContent>
                                    {/*<SelectItem value="claude">Claude</SelectItem>*/}
                                    <SelectItem value="chatgpt">ChatGPT</SelectItem>
                                    <SelectItem value="ollama">Ollama (Local)</SelectItem>
                                </SelectContent>
                            </Select>

                            {/* API Settings */}
                            {(modelConfig.provider === 'claude' || modelConfig.provider === 'chatgpt') && (
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">API Key</label>
                                    <Input
                                        type="password"
                                        placeholder="Enter your API key"  // Make this static to avoid hydration issues
                                        value={modelConfig.apiKey || ''}
                                        onChange={(e) => updateDraft({
                                            apiKey: e.target.value
                                        })}
                                    />
                                    {/* Move the provider-specific text to client-side only */}
                                    <p className="text-xs text-gray-500">
                                        Get your API key from {modelConfig.provider === 'claude'
                                        ? 'console.anthropic.com'
                                        : 'platform.openai.com'}
                                    </p>
                                </div>
                            )}

                            {/* Model Selection */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Model</label>
                                {modelConfig.provider === 'ollama' ? (
                                    <div className="space-y-2">
                                        <Input
                                            placeholder="e.g., llama2, mistral, codellama"
                                            value={modelConfig.model || ''}
                                            onChange={(e) => updateDraft({
                                                model: e.target.value
                                            })}
                                        />
                                        <p className="text-xs text-gray-500 flex items-center gap-1">
                                            <Link2 className="h-3 w-3"/>
                                            <a href="https://ollama.com/search"
                                               target="_blank"
                                               rel="noopener noreferrer"
                                               className="text-blue-500 hover:underline">
                                                Browse available models
                                            </a>
                                        </p>
                                    </div>
                                ) : (
                                    <Select
                                        value={modelConfig.model}
                                        onValueChange={(value) => updateDraft({
                                            model: value
                                        })}
                                    >
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select Model"/>
                                        </SelectTrigger>
                                        <SelectContent>
                                            {MODEL_INFORMATION[modelConfig.provider].models.map(([value, label]) => (
                                                <SelectItem key={value} value={value}>
                                                    {label}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                )}
                            </div>

                            {/* Embedding Model Selection */}
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Embedding Model</label>
                                {modelConfig.provider === 'ollama' ? (
                                    <Input
                                        placeholder="e.g., nomic-embed-text"
                                        value={modelConfig.embeddingModel || ''}
                                        onChange={(e) => updateDraft({
                                            embeddingModel: e.target.value
                                        })}
                                    />
                                ) : (
                                    <Select
                                        value={modelConfig.embeddingModel}
                                        onValueChange={(value) => updateDraft({
                                            embeddingModel: value
                                        })}
                                    >
                                        <SelectTrigger>
                                            <SelectValue placeholder="Select Embedding Model"/>
                                        </SelectTrigger>
                                        <SelectContent>
                                            {MODEL_INFORMATION[modelConfig.provider].embeddingModels.map(([value, label]) => (
                                                <SelectItem key={value} value={value}>
                                                    {label}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                )}
                            </div>

                            {/* Temperature Setting */}
                            {/*<div className="space-y-2">
                                <label className="text-sm font-medium">Temperature</label>
                                <div className="flex items-center gap-4">
                                    <input
                                        type="range"
                                        min="0"
                                        max="2"
                                        step="0.1"
                                        value={modelConfig.temperature}
                                        onChange={(e) => updateDraft({
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
                            </div>*/}

                            {/* System Message */}
                            {/*<div className="space-y-2">
                                <label className="text-sm font-medium">System Message</label>
                                <Textarea
                                    placeholder="Optional: Enter a system message to set the behavior of the AI"
                                    value={modelConfig.systemMessage || ''}
                                    onChange={(e) => updateDraft({
                                        systemMessage: e.target.value
                                    })}
                                    className="resize-none"
                                    rows={3}
                                />
                                <p className="text-xs text-gray-500">
                                    Define the AI&apos;s behavior and context for this model
                                </p>
                            </div>*/}

                            {/* Save Button */}
                            <Button
                                className="w-full"
                                onClick={() => saveConfig(modelConfig)}
                                disabled={isSaving}
                            >
                                {isSaving ? 'Saving...' : 'Save Configuration'}
                            </Button>

                            {/* Error Display */}
                            {configError && (
                                <Alert variant="destructive">
                                    <AlertTitle>{configError.message}</AlertTitle>
                                    {configError.details && configError.details.length > 0 && (
                                        <AlertDescription>
                                            <ul className="list-disc pl-4 mt-2">
                                                {configError.details.map((detail, index) => (
                                                    <li key={index}>{detail}</li>
                                                ))}
                                            </ul>
                                        </AlertDescription>
                                    )}
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
                                            const files = Array.from(e.target.files || []);
                                            // Now handle all files, not just the first one
                                            files.forEach(file => uploadFile(file, modelConfig));
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
                                    <div className="flex items-center gap-2">
                                        <p className="text-sm font-medium truncate">{file.name}</p>
                                        {file.status === 'processing' && (
                                            <div
                                                className="animate-spin rounded-full h-4 w-4 border-2 border-gray-300 border-t-blue-600"/>
                                        )}
                                    </div>
                                    <p className="text-xs text-gray-500">
                                        {file.size} {file.status !== 'processing' && `• ${file.pages} pages`}
                                    </p>
                                </div>
                                {file.status !== 'processing' && (
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                                        onClick={() => deleteFile(file.id)}
                                    >
                                        <Trash2 size={14}/>
                                    </Button>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                {/* Chat Messages */}
                <div ref={messagesContainerRef}
                     className="flex-1 p-6 overflow-y-auto space-y-6"
                >
                    {Array.isArray(messages) && messages.map((message, index) => (
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
                                <MessageContent content={message.content}/>
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
                            ref={inputRef}
                            value={inputText}
                            onChange={(e) => {
                                console.log("Input changed:", e.target.value);
                                setInputText(e.target.value);
                            }}
                            placeholder="Type your message..."
                            className="flex-1"
                            rows={1}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    console.log("Enter key pressed");
                                    e.preventDefault();
                                    handleSendMessage();
                                }
                            }}
                            disabled={chatLoading || !validateModelSettings()}
                        />
                        <Button
                            onClick={(e) => {
                                console.log("Button clicked");
                                e.preventDefault();
                                handleSendMessage();
                            }}
                            disabled={chatLoading || !validateModelSettings() || !inputText.trim()}
                            className="px-4"
                            type="button"
                        >
                            <Send size={16}/>
                        </Button>
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                        {!validateModelSettings() ? (
                            <p className="text-red-500">Please configure model settings before sending messages</p>
                        ) : null}
                        {chatLoading ? (
                            <p>Sending message...</p>
                        ) : null}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;