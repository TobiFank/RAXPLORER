// src/lib/hooks/useFiles.ts
import {useEffect, useState} from "react";
import {fileApi, FileMetadata} from "@/lib/api";
import {ModelConfig} from "@/lib/types";

export function useFiles() {
    const [files, setFiles] = useState<FileMetadata[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadFiles();
    }, []);

    const loadFiles = async () => {
        try {
            setIsLoading(true);
            const loadedFiles = await fileApi.getFiles();
            setFiles(loadedFiles);
        } catch (err) {
            setError('Failed to load files');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const uploadFile = async (file: File, modelConfig: ModelConfig) => {
        try {
            // Create temporary file metadata
            const tempId = crypto.randomUUID();
            const tempFile: FileMetadata = {
                id: tempId,
                name: file.name,
                size: `${(file.size / 1024).toFixed(1)}KB`,
                pages: 0,
                uploadedAt: new Date().toISOString(),
                status: 'processing'
            };

            // Add to files immediately
            setFiles(prev => [...prev, tempFile]);

            // Process file
            setIsLoading(true);
            const uploadedFile = await fileApi.uploadFile(file, modelConfig);

            // Update with processed file
            setFiles(prev => prev.map(f =>
                f.id === tempId ? {...uploadedFile, status: 'complete'} : f
            ));
        } catch (err) {
            // Remove temp file on error
            setFiles(prev => prev.filter(f => f.id !== tempId));
            setError('Failed to upload file');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const deleteFile = async (fileId: string) => {
        try {
            setIsLoading(true);
            await fileApi.deleteFile(fileId);
            setFiles(prev => prev.filter(file => file.id !== fileId));
        } catch (err) {
            setError('Failed to delete file');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return {
        files,
        isLoading,
        error,
        uploadFile,
        deleteFile,
    };
}