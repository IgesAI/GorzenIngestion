import { NextRequest, NextResponse } from 'next/server'

// Import the jobs store from the process route
// Note: In production, this should use a proper database or Redis
const jobs = new Map<string, any>()

// Share the jobs store with the process route
if (typeof global !== 'undefined') {
  if (!global.jobsStore) {
    global.jobsStore = jobs
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  try {
    const jobId = params.jobId
    
    if (!jobId) {
      return NextResponse.json({ error: 'Missing jobId' }, { status: 400 })
    }

    // Get the shared jobs store
    const sharedJobs = global.jobsStore || jobs
    const job = sharedJobs.get(jobId)
    
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 })
    }

    return NextResponse.json(job)
  } catch (error) {
    console.error('Status API error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch job status' },
      { status: 500 }
    )
  }
}
