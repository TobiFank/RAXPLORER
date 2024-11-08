// src/lib/hooks/useChat.ts

import { useState, useEffect, useCallback } from 'react';
import { chatApi, fileApi, modelApi, Chat, ChatMessage, ModelConfig, FileMetadata } from '../api';

export function useChat() {
    const [chats, setChats] = useState<Chat[]>([]);
    const [activeChat, setActiveChat] = useState<string>('');
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Load initial chats
    useEffect(() => {
        loadChats();
    }, []);

    // Load chats from the backend
    const loadChats = async () => {
        try {
            setIsLoading(true);
            const loadedChats = await chatApi.getChats();
            setChats(loadedChats);
            if (loadedChats.length > 0 && !activeChat) {
                setActiveChat(loadedChats[0].id);
                setMessages(loadedChats[0].messages);
            }
        } catch (err) {
            setError('Failed to load chats');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Create a new chat
    const createChat = async () => {
        try {
            setIsLoading(true);
            const newChat = await chatApi.createChat();
            setChats([newChat, ...chats]);
            setActiveChat(newChat.id);
            setMessages([]);
        } catch (err) {
            setError('Failed to create new chat');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Delete a chat
    const deleteChat = async (chatId: string) => {
        try {
            setIsLoading(true);
            await chatApi.deleteChat(chatId);
            const newChats = chats.filter(chat => chat.id !== chatId);
            setChats(newChats);
            if (chatId === activeChat) {
                setActiveChat(newChats[0]?.id || '');
                setMessages(newChats[0]?.messages || []);
            }
        } catch (err) {
            setError('Failed to delete chat');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Switch active chat
    const switchChat = async (chatId: string) => {
        try {
            setIsLoading(true);
            const chat = await chatApi.getChat(chatId);
            setActiveChat(chatId);
            setMessages(chat.messages);
        } catch (err) {
            setError('Failed to switch chat');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Send a message
    const sendMessage = async (content: string, modelConfig: ModelConfig) => {
        if (!activeChat || !content.trim()) return;

        try {
            setIsLoading(true);
            // Add user message immediately
            const userMessage: ChatMessage = {
                role: 'user',
                content,
                timestamp: new Date().toISOString(),
            };
            setMessages(prev => [...prev, userMessage]);

            // Get streaming response
            const stream = chatApi.sendMessage(activeChat, content, modelConfig);
            let assistantMessage = '';

            // Create placeholder for assistant message
            const assistantPlaceholder: ChatMessage = {
                role: 'assistant',
                content: '',
                timestamp: new Date().toISOString(),
            };
            setMessages(prev => [...prev, assistantPlaceholder]);

            // Process the stream
            for await (const chunk of stream) {
                assistantMessage += chunk;
                setMessages(prev => [
                    ...prev.slice(0, -1),
                    {
                        ...assistantPlaceholder,
                        content: assistantMessage,
                    },
                ]);
            }

            // Update chats list
            const updatedChat = await chatApi.getChat(activeChat);
            setChats(prev =>
                prev.map(chat =>
                    chat.id === activeChat ? updatedChat : chat
                )
            );
        } catch (err) {
            setError('Failed to send message');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return {
        chats,
        activeChat,
        messages,
        isLoading,
        error,
        createChat,
        deleteChat,
        switchChat,
        sendMessage,
    };
}