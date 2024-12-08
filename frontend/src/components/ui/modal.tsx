// src/components/ui/modal.tsx
import React, { useEffect } from 'react';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ModalProps {
    isOpen: boolean;
    onClose: () => void;
    children: React.ReactNode;
    title?: string;
}

export function Modal({ isOpen, onClose, children, title }: ModalProps) {
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            <div className="absolute inset-0 bg-black/50" onClick={onClose} />
            <div className="relative z-50 w-[90vw] max-w-4xl max-h-[90vh] bg-white rounded-lg shadow-lg flex flex-col">
                <div className="flex justify-between items-center p-4 border-b">
                    {title && <h2 className="text-lg font-semibold truncate">{title}</h2>}
                    <Button variant="ghost" size="icon" onClick={onClose} className="ml-auto">
                        <X className="h-4 w-4" />
                    </Button>
                </div>
                <div className="flex-1 overflow-auto min-h-0">
                    {children}
                </div>
            </div>
        </div>
    );
}