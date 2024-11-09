'use client'

import { useState, useEffect } from 'react'

export default function MathVideoGenerator() {
  const [problemText, setProblemText] = useState('')
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isTestMode, setIsTestMode] = useState(true)
  const [backendStatus, setBackendStatus] = useState<'checking' | 'running' | 'not-running'>('checking')

  // Check if backend is running on component mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/health')
        if (response.ok) {
          setBackendStatus('running')
        } else {
          setBackendStatus('not-running')
        }
      } catch {
        setBackendStatus('not-running')
      }
    }

    checkBackend()
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (backendStatus !== 'running') {
      setError('Backend server is not running. Please start the FastAPI server.')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/math-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          problemText,
          isTestMode
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate video')
      }

      setVideoUrl(`http://127.0.0.1:8000${data.videoUrl}`)
      
    } catch (err) {
      console.error('Error in handleSubmit:', err)
      setError(
        err instanceof Error 
          ? err.message 
          : 'Failed to generate video. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-4">
      {backendStatus !== 'running' && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded">
          <p className="text-sm text-red-800">
            Backend server is not running. Please start the FastAPI server with:
            <code className="ml-2 p-1 bg-red-100 rounded">npm run fastapi-dev</code>
          </p>
        </div>
      )}

      <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
        <p className="text-sm text-yellow-800">
          Currently running in {isTestMode ? 'Test Mode' : 'Live Mode'}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="problemText" className="block text-sm font-medium mb-2">
            Enter Math Problem
          </label>
          <textarea
            id="problemText"
            value={problemText}
            onChange={(e) => setProblemText(e.target.value)}
            className="w-full p-2 border rounded"
            rows={4}
            placeholder={isTestMode ? "In test mode - will generate test animation" : "Enter your math problem here..."}
            required={!isTestMode}
          />
        </div>
        
        <div className="flex gap-4">
          <button
            type="submit"
            disabled={loading || (!isTestMode && !problemText)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Generating...' : isTestMode ? 'Generate Test Video' : 'Generate Video'}
          </button>

          <button
            type="button"
            onClick={() => setIsTestMode(!isTestMode)}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Switch to {isTestMode ? 'Live' : 'Test'} Mode
          </button>
        </div>
        
        {error && (
          <div className="p-3 text-red-500 bg-red-50 border border-red-200 rounded">
            {error}
          </div>
        )}
        
        {videoUrl && !error && (
          <div className="mt-4">
            <div className="relative w-full pt-[56.25%]">
              <video 
                src={videoUrl}
                controls
                autoPlay
                className="absolute top-0 left-0 w-full h-full rounded-lg shadow-lg"
                playsInline
              >
                <source src={videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        )}
      </form>
    </div>
  )
}