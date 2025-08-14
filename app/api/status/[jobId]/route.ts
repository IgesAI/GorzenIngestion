import { NextRequest, NextResponse } from 'next/server'

// Simple redirect to main process route
export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const jobId = params.jobId
  
  // Redirect to main process route with jobId
  const url = new URL('/api/process', request.url)
  url.searchParams.set('jobId', jobId)
  
  return fetch(url.toString())
}
