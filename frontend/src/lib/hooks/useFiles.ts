// src/lib/hooks/useFiles.ts
import {useEffect, useState} from "react";
import {fileApi, FileMetadata} from "@/lib/api";

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

    const uploadFile = async (file: File) => {
        try {
            setIsLoading(true);
            const uploadedFile = await fileApi.uploadFile(file);
            setFiles(prev => [...prev, uploadedFile]);
        } catch (err) {
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