import { NextRequest, NextResponse } from 'next/server'
import { v4 as uuidv4 } from 'uuid'
import { jobStore, ProcessingJob } from '@/lib/jobStore'

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
    const job: ProcessingJob = {
      id: jobId,
      status: 'processing',
      progress: 0,
      message: 'Processing files...',
      totalFiles: files.length,
      processedFiles: 0,
      indexName: config.indexName,
      createdAt: new Date().toISOString()
    }

    jobStore.set(jobId, job)

    // Simulate processing for demo purposes
    // In production, this would call your Python ingestion pipeline
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

async function simulateProcessing(jobId: string, files: File[], config: any) {
  // Simulate processing steps
  const steps = [
    'Converting documents...',
    'Extracting text content...',
    'Creating chunks...',
    'Generating embeddings...',
    'Uploading to Pinecone...',
    'Finalizing index...'
  ]

  for (let i = 0; i < steps.length; i++) {
    await new Promise(resolve => setTimeout(resolve, 2000)) // 2 second delay
    
    jobStore.update(jobId, {
      message: steps[i],
      progress: Math.floor((i + 1) * 100 / steps.length),
      processedFiles: Math.min(i + 1, files.length)
    })
  }

  // Complete the job
  const totalChunks = files.length * 15 // Simulate 15 chunks per file
  jobStore.update(jobId, {
    status: 'completed',
    progress: totalChunks,
    totalChunks: totalChunks,
    processedFiles: files.length,
    message: `Successfully processed ${files.length} files and created ${totalChunks} vector embeddings`
  })
}
