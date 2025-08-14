import { NextRequest, NextResponse } from 'next/server'
import { writeFile, mkdir } from 'fs/promises'
import { join } from 'path'
import { v4 as uuidv4 } from 'uuid'
import { exec } from 'child_process'
import { promisify } from 'util'
import { jobStore, ProcessingJob } from '@/lib/jobStore'

const execAsync = promisify(exec)

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
      status: 'queued',
      progress: 0,
      message: 'Preparing files for processing...',
      totalFiles: files.length,
      processedFiles: 0,
      indexName: config.indexName,
      createdAt: new Date().toISOString()
    }

    jobStore.set(jobId, job)

    // Process files asynchronously
    processFiles(jobId, files, config).catch(error => {
      console.error('Processing error:', error)
      jobStore.update(jobId, {
        status: 'failed',
        message: error.message || 'Processing failed'
      })
    })

    return NextResponse.json(job)
  } catch (error) {
    console.error('API error:', error)
    return NextResponse.json(
      { error: 'Failed to start processing' },
      { status: 500 }
    )
  }
}

async function processFiles(jobId: string, files: File[], config: any) {
  const job = jobStore.get(jobId)
  if (!job) return

  try {
    // Create temporary directory
    const tempDir = join(process.cwd(), 'temp', jobId)
    await mkdir(tempDir, { recursive: true })

    // Update job status
    jobStore.update(jobId, {
      status: 'processing',
      message: 'Saving uploaded files...'
    })

    // Save files to temp directory
    const filePaths: string[] = []
    for (const file of files) {
      const bytes = await file.arrayBuffer()
      const buffer = Buffer.from(bytes)
      const filePath = join(tempDir, file.name)
      await writeFile(filePath, buffer)
      filePaths.push(filePath)
    }

    // Update job status
    jobStore.update(jobId, {
      message: 'Starting document processing...'
    })

    // Build command for Python script
    const pythonCommand = buildPythonCommand(tempDir, config)
    
    // Execute Python processing script
    jobStore.update(jobId, {
      message: 'Processing documents with Docling...'
    })

    const { stdout, stderr } = await execAsync(pythonCommand, {
      env: {
        ...process.env,
        PINECONE_API_KEY: config.pineconeApiKey,
        OPENAI_API_KEY: config.openaiApiKey || '',
        PINECONE_CLOUD: config.cloud,
        PINECONE_REGION: config.region
      },
      timeout: 30 * 60 * 1000 // 30 minutes timeout
    })

    // Parse output for progress updates
    const lines = stdout.split('\n')
    let totalChunks = 0
    let processedChunks = 0

    for (const line of lines) {
      if (line.includes('Generated') && line.includes('chunks')) {
        const match = line.match(/Generated (\d+) chunks/)
        if (match) {
          totalChunks += parseInt(match[1])
        }
      }
      if (line.includes('Processed') && line.includes('chunks')) {
        const match = line.match(/Processed (\d+) chunks/)
        if (match) {
          processedChunks = parseInt(match[1])
        }
      }
    }

    // Update final job status
    jobStore.update(jobId, {
      status: 'completed',
      progress: totalChunks,
      totalChunks: totalChunks,
      processedFiles: files.length,
      message: `Successfully processed ${files.length} files and created ${totalChunks} vector embeddings`
    })

    // Clean up temp files
    await execAsync(`rm -rf "${tempDir}"`)

  } catch (error) {
    console.error('Processing error:', error)
    jobStore.update(jobId, {
      status: 'failed',
      message: error instanceof Error ? error.message : 'Unknown processing error'
    })
  }
}

function buildPythonCommand(sourceDir: string, config: any): string {
  const parts = [
    'python',
    'ingest.py',
    `--source "${sourceDir}"`,
    `--index "${config.indexName}"`,
    `--chunk-size "${config.chunkSize}"`,
    `--cloud "${config.cloud}"`,
    `--region "${config.region}"`
  ]

  if (config.useOpenAI) {
    parts.push('--use-openai')
  }

  if (config.enrichments.code) {
    parts.push('--enrich-code')
  }

  if (config.enrichments.formula) {
    parts.push('--enrich-formula')
  }

  if (config.enrichments.pictureClasses) {
    parts.push('--enrich-picture-classes')
  }

  if (config.enrichments.pictureDescription) {
    parts.push('--enrich-picture-description')
  }

  return parts.join(' ')
}

// Get job status
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const jobId = searchParams.get('jobId')
  
  if (!jobId) {
    return NextResponse.json({ error: 'Missing jobId' }, { status: 400 })
  }

  const job = jobStore.get(jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  return NextResponse.json(job)
}
