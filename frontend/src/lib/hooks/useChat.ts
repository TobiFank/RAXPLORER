// src/lib/hooks/useChat.ts

import {useCallback, useEffect, useState} from 'react';
import {chatApi} from '../api';
import type {Chat, ChatMessage, ModelConfig} from '@/lib/types';


export function useChat() {
    const [chats, setChats] = useState<Chat[]>([]);
    const [activeChat, setActiveChat] = useState<string>('');
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (activeChat) {
            loadChatMessages(activeChat);
        }
    }, [activeChat]);

    const loadChatMessages = async (chatId: string) => {
        try {
            setIsLoading(true);
            const chat = await chatApi.getChat(chatId);
            setMessages(chat.messages || []);
        } catch (err) {
            setError('Failed to load chat messages');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // Load chats from the backend
    const loadChats = useCallback(async () => {
        try {
            setIsLoading(true);
            const loadedChats = await chatApi.getChats();
            setChats(loadedChats);
            if (loadedChats.length > 0 && !activeChat) {
                setActiveChat(loadedChats[0].id);
            }
        } catch (err) {
            setError(err.message === 'Network Error'
                ? 'Unable to connect to server. Please ensure the backend is running.'
                : 'Failed to load chats');
            console.error(err);
            setChats([]);
        } finally {
            setIsLoading(false);
        }
    }, [activeChat]);

    // Load initial chats
    useEffect(() => {
        loadChats();
    }, [loadChats]);

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
        if (!content.trim()) return;

        try {
            setIsLoading(true);
            let chatId = activeChat;

            if (!chatId) {
                console.log("No active chat, creating one...");
                try {
                    const newChat = await chatApi.createChat();
                    chatId = newChat.id;
                    setChats(prev => [newChat, ...prev]);
                    setActiveChat(chatId);
                    console.log("Created new chat:", chatId);
                } catch (err) {
                    console.error("Failed to create chat:", err);
                    setError('Failed to create new chat');
                    return;
                }
            }

            // Add user message immediately
            const userMessage: ChatMessage = {
                role: 'user',
                content,
                timestamp: new Date().toISOString(),
            };
            // Modified this line to ensure prev is treated as an array
            setMessages(prev => Array.isArray(prev) ? [...prev, userMessage] : [userMessage]);
            console.log('Added user message to state:', userMessage);

            // Get streaming response using the guaranteed chat ID
            console.log('Getting stream from API');
            const stream = chatApi.sendMessage(chatId, content, modelConfig);
            let assistantMessage = '';

            // Create placeholder for assistant message
            const assistantPlaceholder: ChatMessage = {
                role: 'assistant',
                content: '',
                timestamp: new Date().toISOString(),
            };
            // Modified this line to ensure prev is treated as an array
            setMessages(prev => Array.isArray(prev) ? [...prev, assistantPlaceholder] : [userMessage, assistantPlaceholder]);
            console.log('Added assistant placeholder to state');

            // Process the stream
            console.log('Starting to process stream');
            for await (const chunk of stream) {
                console.log('Received chunk:', chunk);
                assistantMessage += chunk;
                setMessages(prev => {
                    if (!Array.isArray(prev)) return [userMessage, { ...assistantPlaceholder, content: assistantMessage }];
                    return [
                        ...prev.slice(0, -1),
                        {
                            ...assistantPlaceholder,
                            content: assistantMessage,
                        },
                    ];
                });
            }

            // Update chats list
            console.log('Stream complete, updating chat');
            const updatedChat = await chatApi.getChat(chatId);
            setChats(prev =>
                prev.map(chat =>
                    chat.id === chatId ? updatedChat : chat
                )
            );
        } catch (err) {
            console.error('Error in sendMessage:', err);
            setError('Failed to send message');
        } finally {
            setIsLoading(false);
        }
    };

    const updateChatTitle = async (chatId: string, title: string): Promise<void> => {
        try {
            setIsLoading(true);
            const updatedChat = await chatApi.updateChatTitle(chatId, title);
            setChats(prev => prev.map(chat =>
                chat.id === chatId ? updatedChat : chat
            ));
            return Promise.resolve();
        } catch (err) {
            console.error('Error updating chat title:', err);
            return Promise.reject(err);
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
        updateChatTitle,
    };
}