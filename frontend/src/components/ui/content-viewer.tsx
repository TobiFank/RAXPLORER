// src/components/ui/content-viewer.tsx
import React from 'react';
import { Modal } from './modal';

interface ContentViewerProps {
    isOpen: boolean;
    onClose: () => void;
    url: string;
    type: 'image' | 'document';
    title: string;
}

const ContentViewer = ({ isOpen, onClose, url, type, title }: ContentViewerProps) => {
    return (
        <Modal isOpen={isOpen} onClose={onClose} title={title}>
            {type === 'image' ? (
                <img
                    src={url}
                    alt={title}
                    className="w-full h-full object-contain"
                />
            ) : (
                <iframe
                    src={url}
                    className="w-full h-full min-h-[80vh]"
                    title={title}
                />
            )}
        </Modal>
    );
}
export default ContentViewer;