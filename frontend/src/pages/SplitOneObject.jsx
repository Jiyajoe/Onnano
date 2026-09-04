import React, { useState } from 'react'
import ObjectInput from '../components/ObjectInput.jsx'
import AnalysisProgress from '../components/AnalysisProgress.jsx'
import SingleResultView from '../components/SingleResultView.jsx'
import { api, ApiError } from '../services/api.js'

export default function SplitOneObject() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [errorMessage, setErrorMessage] = useState(null)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isUpdatingDivision, setIsUpdatingDivision] = useState(false)

  const handleImageSelected = async (fileOrBlob, dataUrl) => {
    setSelectedFile(fileOrBlob)
    setPreviewUrl(dataUrl)
    setErrorMessage(null)
    setStatus('loading')

    try {
      const data = await api.analyzeObject(fileOrBlob, 4)
      setAnalysisResult(data)
      setStatus('success')
    } catch (err) {
      setStatus('error')
      setErrorMessage(
        err instanceof ApiError
          ? err.message
          : 'We encountered an error processing the image. Please try again with good lighting.'
      )
    }
  }

  const handleImageRemoved = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setAnalysisResult(null)
    setStatus('idle')
    setErrorMessage(null)
  }

  const handlePartsChange = async (newPartsCount) => {
    if (!selectedFile) return
    setIsUpdatingDivision(true)
    try {
      const data = await api.divideObject(selectedFile, newPartsCount)
      setAnalysisResult(data)
    } catch (err) {
      console.error('Failed to update division:', err)
    } finally {
      setIsUpdatingDivision(false)
    }
  }

  return (
    <div className="page-wrapper">
      <div className="page-header-banner">
        <div className="mode-badge-top">MODE 1</div>
        <h1 className="main-title">Split One Object</h1>
        <p className="main-subtitle">
          AI/CV detects the physical object, corrects tilt/posture, identifies features, and divides it equally along its principal geometric axis.
        </p>
      </div>

      {status === 'idle' && (
        <div className="mode-input-section">
          <ObjectInput
            label="Upload Photo OR Open Camera"
            objectNumber={1}
            previewUrl={previewUrl}
            onImageSelected={handleImageSelected}
            onImageRemoved={handleImageRemoved}
          />
        </div>
      )}

      {status === 'loading' && (
        <div className="progress-container">
          <AnalysisProgress title="AI Understanding & Posture Normalization..." />
        </div>
      )}

      {status === 'error' && (
        <div className="error-display-card">
          <div className="error-icon">⚠️</div>
          <h3>Detection Alert</h3>
          <p>{errorMessage}</p>
          <button className="btn btn--primary" onClick={handleImageRemoved}>
            Try Another Photo
          </button>
        </div>
      )}

      {status === 'success' && analysisResult && (
        <div className="result-container-flow">
          <div className="reset-bar">
            <button className="btn btn--ghost btn--sm" onClick={handleImageRemoved}>
              ← Analyze Different Object
            </button>
          </div>

          <SingleResultView
            data={analysisResult}
            onPartsChange={handlePartsChange}
            isUpdating={isUpdatingDivision}
          />
        </div>
      )}
    </div>
  )
}
