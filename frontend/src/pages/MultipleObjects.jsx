import React, { useState } from 'react'
import ObjectInput from '../components/ObjectInput.jsx'
import AnalysisProgress from '../components/AnalysisProgress.jsx'
import MultiResultDashboard from '../components/MultiResultDashboard.jsx'
import { api, ApiError } from '../services/api.js'

export default function MultipleObjects() {
  const [objectCount, setObjectCount] = useState(3)
  const [objectsData, setObjectsData] = useState([
    { id: 1, file: null, previewUrl: null },
    { id: 2, file: null, previewUrl: null },
    { id: 3, file: null, previewUrl: null },
  ])

  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [errorMessage, setErrorMessage] = useState(null)
  const [comparisonResult, setComparisonResult] = useState(null)

  const handleCountChange = (newCount) => {
    const count = Math.max(2, Math.min(8, parseInt(newCount, 10) || 2))
    setObjectCount(count)

    setObjectsData((prev) => {
      const updated = []
      for (let i = 1; i <= count; i++) {
        const existing = prev.find((item) => item.id === i)
        if (existing) {
          updated.push(existing)
        } else {
          updated.push({ id: i, file: null, previewUrl: null })
        }
      }
      return updated
    })
  }

  const handleImageSelected = (index, fileOrBlob, dataUrl) => {
    setObjectsData((prev) => {
      const updated = [...prev]
      updated[index] = { ...updated[index], file: fileOrBlob, previewUrl: dataUrl }
      return updated
    })
  }

  const handleImageRemoved = (index) => {
    setObjectsData((prev) => {
      const updated = [...prev]
      updated[index] = { ...updated[index], file: null, previewUrl: null }
      return updated
    })
  }

  // Check if all N images have been provided
  const readyCount = objectsData.filter((item) => item.file !== null).length
  const allImagesReady = readyCount === objectCount

  const handleCompare = async () => {
    if (!allImagesReady) return

    setStatus('loading')
    setErrorMessage(null)

    try {
      const blobs = objectsData.map((item) => item.file)
      const data = await api.compareObjects(blobs)
      setComparisonResult(data)
      setStatus('success')
    } catch (err) {
      setStatus('error')
      setErrorMessage(
        err instanceof ApiError
          ? err.message
          : 'Failed to process and compare physical objects. Please ensure good lighting and contrasting backgrounds.'
      )
    }
  }

  const handleStartOver = () => {
    setStatus('idle')
    setComparisonResult(null)
    setErrorMessage(null)
    setObjectsData((prev) => prev.map((item) => ({ ...item, file: null, previewUrl: null })))
  }

  return (
    <div className="page-wrapper">
      <div className="page-header-banner">
        <div className="mode-badge-top mode-badge-top--multi">MODE 2</div>
        <h1 className="main-title">Divide & Compare Multiple Objects</h1>
        <p className="main-subtitle">
          Dynamic N-object input, independent camera/upload, pairwise computer vision feature matching, similarity matrix, and humorous Malayalam verdicts.
        </p>
      </div>

      {status === 'idle' && (
        <div className="multi-setup-container">
          {/* Object Count Stepper */}
          <div className="object-count-picker-card">
            <div className="picker-left">
              <span className="picker-icon">🔢</span>
              <div>
                <h3>How many objects do you want to compare?</h3>
                <p>Choose any number from 2 to 8 items</p>
              </div>
            </div>

            <div className="picker-controls">
              <button
                className="btn btn--stepper"
                onClick={() => handleCountChange(objectCount - 1)}
                disabled={objectCount <= 2}
                title="Decrease objects"
              >
                -
              </button>

              <div className="count-display">
                <span className="count-number">{objectCount}</span>
                <span className="count-text">Objects</span>
              </div>

              <button
                className="btn btn--stepper"
                onClick={() => handleCountChange(objectCount + 1)}
                disabled={objectCount >= 8}
                title="Increase objects"
              >
                +
              </button>
            </div>
          </div>

          {/* Dynamic N-object cards grid */}
          <div className="dynamic-cards-grid">
            {objectsData.map((item, idx) => (
              <ObjectInput
                key={item.id}
                label={`Object #${item.id}`}
                objectNumber={item.id}
                previewUrl={item.previewUrl}
                onImageSelected={(blob, url) => handleImageSelected(idx, blob, url)}
                onImageRemoved={() => handleImageRemoved(idx)}
              />
            ))}
          </div>

          {/* Auto-Enabled Compare Trigger */}
          <div className="multi-action-toolbar">
            <div className="ready-indicator">
              <span className="indicator-dot" style={{ backgroundColor: allImagesReady ? '#2fd9a8' : '#ffb627' }}></span>
              <span>
                {readyCount} of {objectCount} Photos Ready
              </span>
            </div>

            <button
              className={`btn btn--primary btn--large ${allImagesReady ? 'btn--pulse' : 'btn--disabled'}`}
              onClick={handleCompare}
              disabled={!allImagesReady}
            >
              {allImagesReady ? '⚡ Compare Objects Now' : `Add ${objectCount - readyCount} More Photo(s) to Compare`}
            </button>
          </div>
        </div>
      )}

      {status === 'loading' && (
        <div className="progress-container">
          <AnalysisProgress title={`Analyzing & Comparing ${objectCount} Physical Objects...`} />
        </div>
      )}

      {status === 'error' && (
        <div className="error-display-card">
          <div className="error-icon">⚠️</div>
          <h3>Multi-Object Analysis Alert</h3>
          <p>{errorMessage}</p>
          <button className="btn btn--primary" onClick={() => setStatus('idle')}>
            Edit Object Photos
          </button>
        </div>
      )}

      {status === 'success' && comparisonResult && (
        <div className="result-container-flow">
          <div className="reset-bar">
            <button className="btn btn--ghost btn--sm" onClick={handleStartOver}>
              ← Start New Comparison
            </button>
          </div>

          <MultiResultDashboard data={comparisonResult} onStartOver={handleStartOver} />
        </div>
      )}
    </div>
  )
}
