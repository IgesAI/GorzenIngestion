import { NextRequest, NextResponse } from 'next/server'
import { v4 as uuidv4 } from 'uuid'

// Simple in-memory job store for demo
const jobs = new Map<string, any>()

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const files = formData.getAll('files') as File[]
    const configStr = formData.get('config') as string
    
    if (!files.length || !configStr) {
      return NextResponse.json({ error: 'Missing files or config' }, { status: 400 })
    }

    const config = JSON.parse(configStr)
    const jobId = uuidv4()

    // Create job
    const job = {
      id: jobId,
      status: 'processing',
      progress: 0,
      message: 'Starting document processing...',
      totalFiles: files.length,
      processedFiles: 0,
      indexName: config.indexName,
      createdAt: new Date().toISOString()
    }

    jobs.set(jobId, job)

    // Simulate processing (replace with actual Pinecone integration)
    simulateProcessing(jobId, files, config)

    return NextResponse.json(job)
  } catch (error) {
    console.error('API error:', error)
    return NextResponse.json(
      { error: 'Failed to start processing' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const jobId = searchParams.get('jobId')
  
  if (!jobId) {
    return NextResponse.json({ error: 'Missing jobId' }, { status: 400 })
  }

  const job = jobs.get(jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  return NextResponse.json(job)
}

async function simulateProcessing(jobId: string, files: File[], config: any) {
  const job = jobs.get(jobId)
  if (!job) return

  // Simulate processing steps
  const steps = [
    'Converting documents with Docling...',
    'Extracting text and metadata...',
    'Creating intelligent chunks...',
    'Generating embeddings...',
    'Creating Pinecone index...',
    'Uploading vectors to Pinecone...',
    'Finalizing index...'
  ]

  for (let i = 0; i < steps.length; i++) {
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    job.message = steps[i]
    job.progress = Math.floor(((i + 1) / steps.length) * 100)
    job.processedFiles = Math.min(i + 1, files.length)
    
    jobs.set(jobId, job)
  }

  // Complete the job
  const totalChunks = files.length * 12 // Simulate chunks
  job.status = 'completed'
  job.progress = totalChunks
  job.totalChunks = totalChunks
  job.processedFiles = files.length
  job.message = `Successfully processed ${files.length} files and created ${totalChunks} vector embeddings in index "${config.indexName}"`
  
  jobs.set(jobId, job)
}