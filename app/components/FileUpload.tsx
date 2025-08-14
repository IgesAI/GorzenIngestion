'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, X, AlertCircle } from 'lucide-react'

interface FileUploadProps {
  onFilesUploaded: (files: File[]) => void
}

export default function FileUpload({ onFilesUploaded }: FileUploadProps) {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(null)
    
    if (rejectedFiles.length > 0) {
      setError('Some files were rejected. Please upload only PDF, DOCX, or TXT files.')
      return
    }

    // Check file sizes (50MB limit per file)
    const oversizedFiles = acceptedFiles.filter(file => file.size > 50 * 1024 * 1024)
    if (oversizedFiles.length > 0) {
      setError('Some files are too large. Maximum file size is 50MB.')
      return
    }

    setUploadedFiles(prev => [...prev, ...acceptedFiles])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md']
    },
    multiple: true,
    maxSize: 50 * 1024 * 1024 // 50MB
  })

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleContinue = () => {
    if (uploadedFiles.length > 0) {
      onFilesUploaded(uploadedFiles)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Upload Your Documents</h2>
        <p className="text-gray-300">
          Select PDF, DOCX, TXT, or Markdown files to create your vector database
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          upload-area border-2 border-dashed rounded-xl p-12 text-center cursor-pointer glass-dark
          ${isDragActive 
            ? 'drag-over' 
            : ''
          }
        `}
      >
        <input {...getInputProps()} />
        <Upload className="h-12 w-12 text-purple-300 mx-auto mb-4" />
        
        {isDragActive ? (
          <div>
            <p className="text-lg font-medium text-blue-600 mb-2">Drop your files here</p>
            <p className="text-sm text-blue-500">Release to upload</p>
          </div>
        ) : (
          <div>
            <p className="text-lg font-medium text-gray-200 mb-2">
              Drag & drop files here, or click to browse
            </p>
            <p className="text-sm text-gray-400 mb-4">
              Supports PDF, DOCX, TXT, and Markdown files (max 50MB each)
            </p>
            <button
              type="button"
              className="bg-purple-600 text-white px-6 py-2 rounded-lg hover:bg-purple-700 transition-colors"
            >
              Choose Files
            </button>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
          <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h4 className="font-medium text-red-800">Upload Error</h4>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Uploaded Files ({uploadedFiles.length})
          </h3>
          
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-3">
                  <File className="h-5 w-5 text-gray-500" />
                  <div>
                    <p className="font-medium text-gray-900 truncate max-w-xs">
                      {file.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                </div>
                
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                  aria-label="Remove file"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>

          <div className="flex justify-between items-center pt-4 border-t">
            <div className="text-sm text-gray-600">
              Total: {uploadedFiles.length} files ({formatFileSize(uploadedFiles.reduce((acc, file) => acc + file.size, 0))})
            </div>
            
            <button
              onClick={handleContinue}
              className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Continue to Configuration
            </button>
          </div>
        </div>
      )}

      {/* Upload Tips */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-900 mb-2">💡 Tips for best results:</h4>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Upload high-quality PDFs with selectable text (not scanned images)</li>
          <li>• Include diverse document types for comprehensive coverage</li>
          <li>• Organize related documents together for better context</li>
          <li>• Consider splitting very large documents into smaller sections</li>
        </ul>
      </div>
    </div>
  )
}
