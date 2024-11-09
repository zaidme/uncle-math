import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  try {
    const body = await req.json()
    
    // Make sure FastAPI is actually running
    try {
      const response = await fetch('http://127.0.0.1:8000/api/generate-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem_text: body.problemText,
          isTestMode: body.isTestMode
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'FastAPI request failed')
      }

      const data = await response.json()
      return NextResponse.json(data)
      
    } catch (error: any) {
      // Check if it's a connection refused error
      if (error.cause?.code === 'ECONNREFUSED') {
        console.error('FastAPI server is not running')
        return NextResponse.json(
          { error: 'Backend server is not running. Please start the FastAPI server.' },
          { status: 503 }
        )
      }
      throw error
    }
    
  } catch (error) {
    console.error('Error in math-video route:', error)
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to generate video',
        details: error instanceof Error ? error.cause : undefined
      },
      { status: 500 }
    )
  }
}