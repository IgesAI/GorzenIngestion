import { NextRequest, NextResponse } from 'next/server'
import { jobStore } from '@/lib/jobStore'

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const jobId = params.jobId
  
  if (!jobId) {
    return NextResponse.json({ error: 'Missing jobId' }, { status: 400 })
  }

  const job = jobStore.get(jobId)
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 })
  }

  return NextResponse.json(job)
}
