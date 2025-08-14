'use client'

import { Loader2, CheckCircle, AlertCircle, FileText, Database } from 'lucide-react'

interface ProcessingJob {
  id: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  message: string
  totalFiles: number
  processedFiles: number
  indexName?: string
  currentFile?: string
  totalChunks?: number
}

interface ProcessingStatusProps {
  job: ProcessingJob
}

export default function ProcessingStatus({ job }: ProcessingStatusProps) {
  const getStatusIcon = () => {
    switch (job.status) {
      case 'queued':
        return <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
      case 'processing':
        return <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-8 w-8 text-green-500" />
      case 'failed':
        return <AlertCircle className="h-8 w-8 text-red-500" />
      default:
        return <Loader2 className="h-8 w-8 text-gray-400" />
    }
  }

  const getStatusColor = () => {
    switch (job.status) {
      case 'queued':
        return 'text-blue-600'
      case 'processing':
        return 'text-blue-600'
      case 'completed':
        return 'text-green-600'
      case 'failed':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  const getProgressBarColor = () => {
    switch (job.status) {
      case 'completed':
        return 'bg-green-500'
      case 'failed':
        return 'bg-red-500'
      default:
        return 'bg-blue-500'
    }
  }

  const fileProgress = job.totalFiles > 0 ? (job.processedFiles / job.totalFiles) * 100 : 0

  return (
    <div className="text-center space-y-6">
      <div className="flex flex-col items-center">
        {getStatusIcon()}
        <h2 className={`text-2xl font-bold mt-4 ${getStatusColor()}`}>
          {job.status === 'queued' && 'Processing Queued'}
          {job.status === 'processing' && 'Processing Documents'}
          {job.status === 'completed' && 'Processing Complete'}
          {job.status === 'failed' && 'Processing Failed'}
        </h2>
        <p className="text-gray-600 mt-2">{job.message}</p>
      </div>

      {/* File Progress */}
      {job.totalFiles > 0 && (
        <div className="bg-gray-50 rounded-lg p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <FileText className="h-5 w-5 text-gray-500" />
              <span className="font-medium text-gray-900">File Progress</span>
            </div>
            <span className="text-sm text-gray-600">
              {job.processedFiles} / {job.totalFiles} files
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-300 ${getProgressBarColor()}`}
              style={{ width: `${fileProgress}%` }}
            />
          </div>
          
          {job.currentFile && job.status === 'processing' && (
            <p className="text-sm text-gray-600">
              Currently processing: <span className="font-medium">{job.currentFile}</span>
            </p>
          )}
        </div>
      )}

      {/* Chunk Progress */}
      {job.totalChunks && job.progress > 0 && (
        <div className="bg-gray-50 rounded-lg p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Database className="h-5 w-5 text-gray-500" />
              <span className="font-medium text-gray-900">Vector Storage</span>
            </div>
            <span className="text-sm text-gray-600">
              {job.progress} chunks indexed
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-300 ${getProgressBarColor()}`}
              style={{ width: `${Math.min((job.progress / (job.totalChunks || 1)) * 100, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Processing Steps */}
      {job.status === 'processing' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h4 className="font-medium text-blue-900 mb-4">Processing Steps</h4>
          <div className="space-y-3 text-left">
            <ProcessingStep
              title="Document Conversion"
              description="Converting PDFs with Docling"
              isActive={job.processedFiles < job.totalFiles}
              isCompleted={false}
            />
            <ProcessingStep
              title="Smart Chunking"
              description="Breaking documents into semantic chunks"
              isActive={job.processedFiles > 0}
              isCompleted={false}
            />
            <ProcessingStep
              title="Embedding Generation"
              description="Creating vector embeddings"
              isActive={job.progress > 0}
              isCompleted={false}
            />
            <ProcessingStep
              title="Vector Storage"
              description="Storing in Pinecone index"
              isActive={job.progress > 0}
              isCompleted={false}
            />
          </div>
        </div>
      )}

      {/* Index Information */}
      {job.indexName && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-2">Index Details</h4>
          <p className="text-sm text-gray-600">
            <strong>Index Name:</strong> {job.indexName}
          </p>
          {job.status === 'completed' && (
            <p className="text-sm text-gray-600 mt-1">
              Your vector database is ready for use!
            </p>
          )}
        </div>
      )}

      {/* Error Details */}
      {job.status === 'failed' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="font-medium text-red-800 mb-2">Error Details</h4>
          <p className="text-sm text-red-700">{job.message}</p>
          <div className="mt-4">
            <button
              onClick={() => window.location.reload()}
              className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors text-sm"
            >
              Try Again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

interface ProcessingStepProps {
  title: string
  description: string
  isActive: boolean
  isCompleted: boolean
}

function ProcessingStep({ title, description, isActive, isCompleted }: ProcessingStepProps) {
  return (
    <div className="flex items-center space-x-3">
      <div className={`
        w-3 h-3 rounded-full
        ${isCompleted 
          ? 'bg-green-500' 
          : isActive 
            ? 'bg-blue-500 processing-animation' 
            : 'bg-gray-300'
        }
      `} />
      <div>
        <p className={`text-sm font-medium ${isActive ? 'text-blue-800' : 'text-gray-600'}`}>
          {title}
        </p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
    </div>
  )
}
