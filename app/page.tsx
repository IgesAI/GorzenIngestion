'use client'

import { useState } from 'react'
import { Upload, Database, Zap, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import FileUpload from './components/FileUpload'
import ConfigForm from './components/ConfigForm'
import ProcessingStatus from './components/ProcessingStatus'

type Step = 'upload' | 'config' | 'processing' | 'complete'

interface ProcessingJob {
  id: string
  status: 'queued' | 'processing' | 'completed' | 'failed'
  progress: number
  message: string
  totalFiles: number
  processedFiles: number
  indexName?: string
}

export default function Home() {
  const [currentStep, setCurrentStep] = useState<Step>('upload')
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [processingJob, setProcessingJob] = useState<ProcessingJob | null>(null)

  const handleFilesUploaded = (files: File[]) => {
    setUploadedFiles(files)
    setCurrentStep('config')
  }

  const handleConfigSubmit = async (config: any) => {
    setCurrentStep('processing')
    
    // Create FormData with files and config
    const formData = new FormData()
    uploadedFiles.forEach((file, index) => {
      formData.append(`files`, file)
    })
    formData.append('config', JSON.stringify(config))

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      })

      const job = await response.json()
      setProcessingJob(job)

      // Poll for status updates
      pollJobStatus(job.id)
    } catch (error) {
      console.error('Failed to start processing:', error)
    }
  }

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/status/${jobId}`)
        const job = await response.json()
        setProcessingJob(job)

        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval)
          if (job.status === 'completed') {
            setCurrentStep('complete')
          }
        }
      } catch (error) {
        console.error('Failed to fetch job status:', error)
        clearInterval(interval)
      }
    }, 2000)
  }

  const resetFlow = () => {
    setCurrentStep('upload')
    setUploadedFiles([])
    setProcessingJob(null)
  }

  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Header */}
      <div className="text-center mb-12">
        <div className="flex items-center justify-center mb-4">
          <Database className="h-12 w-12 text-blue-600 mr-3" />
          <h1 className="text-4xl font-bold text-gray-900">Gorzen Ingestion</h1>
        </div>
        <p className="text-xl text-gray-600 mb-6">
          Transform any document collection into a searchable vector database in minutes
        </p>
        
        {/* Progress Steps */}
        <div className="flex items-center justify-center space-x-8 mb-8">
          <StepIndicator
            step={1}
            label="Upload Files"
            isActive={currentStep === 'upload'}
            isCompleted={['config', 'processing', 'complete'].includes(currentStep)}
            icon={<Upload className="h-5 w-5" />}
          />
          <StepIndicator
            step={2}
            label="Configure"
            isActive={currentStep === 'config'}
            isCompleted={['processing', 'complete'].includes(currentStep)}
            icon={<Zap className="h-5 w-5" />}
          />
          <StepIndicator
            step={3}
            label="Process"
            isActive={currentStep === 'processing'}
            isCompleted={currentStep === 'complete'}
            icon={<Loader2 className="h-5 w-5" />}
          />
          <StepIndicator
            step={4}
            label="Complete"
            isActive={currentStep === 'complete'}
            isCompleted={currentStep === 'complete'}
            icon={<CheckCircle className="h-5 w-5" />}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="bg-white rounded-lg shadow-lg p-8">
        {currentStep === 'upload' && (
          <FileUpload onFilesUploaded={handleFilesUploaded} />
        )}

        {currentStep === 'config' && (
          <ConfigForm 
            fileCount={uploadedFiles.length}
            onSubmit={handleConfigSubmit}
            onBack={() => setCurrentStep('upload')}
          />
        )}

        {currentStep === 'processing' && processingJob && (
          <ProcessingStatus job={processingJob} />
        )}

        {currentStep === 'complete' && processingJob && (
          <div className="text-center">
            <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Vector Database Created Successfully!
            </h2>
            <p className="text-gray-600 mb-6">
              Your documents have been processed and indexed in Pinecone.
            </p>
            
            <div className="bg-gray-50 rounded-lg p-6 mb-6">
              <h3 className="font-semibold text-gray-900 mb-2">Index Details:</h3>
              <p><strong>Index Name:</strong> {processingJob.indexName}</p>
              <p><strong>Documents Processed:</strong> {processingJob.processedFiles}</p>
              <p><strong>Total Chunks:</strong> {processingJob.progress}</p>
            </div>

            <div className="space-y-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2">Next Steps:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Your vector database is ready for semantic search</li>
                  <li>• Use the Pinecone console to explore your index</li>
                  <li>• Connect your applications using the Pinecone SDK</li>
                </ul>
              </div>
              
              <button
                onClick={resetFlow}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Process More Documents
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Features */}
      {currentStep === 'upload' && (
        <div className="mt-12 grid md:grid-cols-3 gap-6">
          <FeatureCard
            icon={<Upload className="h-8 w-8 text-blue-600" />}
            title="Easy Upload"
            description="Drag and drop any PDF, DOCX, or text files. Batch processing supported."
          />
          <FeatureCard
            icon={<Zap className="h-8 w-8 text-green-600" />}
            title="Smart Processing"
            description="Advanced document parsing with Docling and intelligent chunking strategies."
          />
          <FeatureCard
            icon={<Database className="h-8 w-8 text-purple-600" />}
            title="Vector Database"
            description="Automatically creates and populates your Pinecone index with searchable embeddings."
          />
        </div>
      )}
    </main>
  )
}

interface StepIndicatorProps {
  step: number
  label: string
  isActive: boolean
  isCompleted: boolean
  icon: React.ReactNode
}

function StepIndicator({ step, label, isActive, isCompleted, icon }: StepIndicatorProps) {
  return (
    <div className="flex flex-col items-center">
      <div className={`
        w-12 h-12 rounded-full flex items-center justify-center border-2 mb-2
        ${isCompleted 
          ? 'bg-green-500 border-green-500 text-white' 
          : isActive 
            ? 'bg-blue-500 border-blue-500 text-white' 
            : 'bg-gray-100 border-gray-300 text-gray-400'
        }
      `}>
        {isCompleted ? <CheckCircle className="h-5 w-5" /> : icon}
      </div>
      <span className={`text-sm font-medium ${isActive ? 'text-blue-600' : 'text-gray-500'}`}>
        {label}
      </span>
    </div>
  )
}

interface FeatureCardProps {
  icon: React.ReactNode
  title: string
  description: string
}

function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <div className="bg-white rounded-lg p-6 shadow-md border border-gray-100">
      <div className="mb-4">{icon}</div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600 text-sm">{description}</p>
    </div>
  )
}
