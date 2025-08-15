import { NextRequest, NextResponse } from 'next/server'
import { v4 as uuidv4 } from 'uuid'

// Simple in-memory job store for demo
const jobs = new Map<string, any>()

// Share the jobs store globally for status endpoint access
if (typeof global !== 'undefined') {
  if (!global.jobsStore) {
    global.jobsStore = jobs
  }
}

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

    // Use shared jobs store
    const sharedJobs = global.jobsStore || jobs
    sharedJobs.set(jobId, job)

    // Start actual Pinecone processing
    processWithPinecone(jobId, files, config)

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

  // Use shared jobs store
  const sharedJobs = global.jobsStore || jobs
  const job = sharedJobs.get(jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  return NextResponse.json(job)
}

async function processWithPinecone(jobId: string, files: File[], config: any) {
  const sharedJobs = global.jobsStore || jobs
  const job = sharedJobs.get(jobId)
  if (!job) return

  try {
    // Create temporary directory for uploaded files
    const fs = await import('fs')
    const path = await import('path')
    const { spawn } = await import('child_process')
    const os = await import('os')
    
    const tempDir = path.join(os.tmpdir(), `gorzen-${jobId}`)
    await fs.promises.mkdir(tempDir, { recursive: true })

    // Update job status
    job.message = 'Preparing files for processing...'
    job.progress = 5
    sharedJobs.set(jobId, job)

    // Save uploaded files to temp directory
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const filePath = path.join(tempDir, file.name)
      const buffer = Buffer.from(await file.arrayBuffer())
      await fs.promises.writeFile(filePath, buffer)
      
      job.message = `Saved file ${i + 1}/${files.length}: ${file.name}`
      job.progress = 5 + Math.floor((i / files.length) * 10)
      sharedJobs.set(jobId, job)
    }

    // Update job status
    job.message = 'Starting Pinecone ingestion...'
    job.progress = 15
    sharedJobs.set(jobId, job)

    // Prepare Python command arguments
    const pythonArgs = [
      'ingest.py',
      '--source', tempDir,
      '--index', config.indexName,
      '--chunk-size', config.chunkSize || 'medium',
      '--batch-size', '32'
    ]

    // Add optional parameters
    if (config.useOpenAI) {
      pythonArgs.push('--use-openai')
    }
    if (config.enrichments?.code) {
      pythonArgs.push('--enrich-code')
    }
    if (config.enrichments?.formula) {
      pythonArgs.push('--enrich-formula')
    }
    if (config.enrichments?.pictureClasses) {
      pythonArgs.push('--enrich-picture-classes')
    }
    if (config.enrichments?.pictureDescription) {
      pythonArgs.push('--enrich-picture-description')
    }

    // Set up environment variables
    const env = { ...process.env }
    if (config.pineconeApiKey) {
      env.PINECONE_API_KEY = config.pineconeApiKey
    }
    if (config.openaiApiKey) {
      env.OPENAI_API_KEY = config.openaiApiKey
    }

    // Try different Python executables in order of preference
    const pythonCommands = process.platform === 'win32' 
      ? ['python', 'python.exe', 'py', 'python3']
      : ['python3', 'python']
    
    let pythonProcess: any = null
    let lastError: Error | null = null
    
    // Try each Python command until one works
    for (const pythonCmd of pythonCommands) {
      try {
        pythonProcess = spawn(pythonCmd, pythonArgs, {
          cwd: process.cwd(),
          env,
          stdio: ['pipe', 'pipe', 'pipe'],
          shell: true
        })
        
        // If spawn succeeds, break out of loop
        break
      } catch (error) {
        lastError = error as Error
        console.log(`Failed to spawn with ${pythonCmd}:`, error)
        continue
      }
    }
    
    // If no Python command worked, throw error
    if (!pythonProcess) {
      throw new Error(`Failed to find Python executable. Tried: ${pythonCommands.join(', ')}. Last error: ${lastError?.message}`)
    }

    let totalChunks = 0
    let processedFiles = 0

    // Handle stdout for progress updates
    pythonProcess.stdout.on('data', (data: Buffer) => {
      const output = data.toString()
      console.log('Python stdout:', output)
      
      // Parse progress from Python output
      if (output.includes('Processing:')) {
        processedFiles++
        job.message = `Processing document ${processedFiles}/${files.length}...`
        job.processedFiles = processedFiles
        job.progress = 15 + Math.floor((processedFiles / files.length) * 60)
        sharedJobs.set(jobId, job)
      } else if (output.includes('Generated') && output.includes('chunks')) {
        const match = output.match(/Generated (\d+) chunks/)
        if (match) {
          const chunks = parseInt(match[1])
          totalChunks += chunks
          job.message = `Generated ${totalChunks} chunks so far...`
          sharedJobs.set(jobId, job)
        }
      } else if (output.includes('Upserting')) {
        job.message = 'Uploading vectors to Pinecone...'
        job.progress = 75 + Math.floor(Math.random() * 20) // 75-95%
        sharedJobs.set(jobId, job)
      }
    })

    // Handle stderr
    pythonProcess.stderr.on('data', (data: Buffer) => {
      const error = data.toString()
      console.error('Python stderr:', error)
      
      // Don't fail on warnings, only on actual errors
      if (error.toLowerCase().includes('error') && !error.toLowerCase().includes('warning')) {
        job.status = 'failed'
        job.message = `Processing failed: ${error}`
        sharedJobs.set(jobId, job)
      }
    })

    // Handle process completion
    pythonProcess.on('close', async (code: number | null) => {
      try {
        // Clean up temp directory
        await fs.promises.rm(tempDir, { recursive: true, force: true })
        
        if (code === 0) {
          // Success
          job.status = 'completed'
          job.progress = totalChunks || files.length * 12 // Fallback estimate
          job.totalChunks = totalChunks || files.length * 12
          job.processedFiles = files.length
          job.message = `Successfully processed ${files.length} files and created ${job.progress} vector embeddings in index "${config.indexName}"`
        } else {
          // Failure
          job.status = 'failed'
          job.message = `Processing failed with exit code ${code}. Please check your API keys and try again.`
        }
        
        sharedJobs.set(jobId, job)
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError)
      }
    })

    // Handle process errors
    pythonProcess.on('error', (error: Error) => {
      console.error('Python process error:', error)
      job.status = 'failed'
      job.message = `Failed to start processing: ${error.message}`
      sharedJobs.set(jobId, job)
    })

  } catch (error) {
    console.error('Processing error:', error)
    job.status = 'failed'
    job.message = `Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    sharedJobs.set(jobId, job)
  }
}