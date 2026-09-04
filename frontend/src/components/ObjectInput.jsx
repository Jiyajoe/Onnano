import React, { useState, useRef } from 'react'
import CameraCapture from './CameraCapture.jsx'

export default function ObjectInput({
  label = 'Object Photo',
  objectNumber = 1,
  previewUrl = null,
  onImageSelected,
  onImageRemoved,
  required = true,
}) {
  const fileInputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [showCamera, setShowCamera] = useState(false)

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      onImageSelected(file, url)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file)
      onImageSelected(file, url)
    }
  }

  const handleCameraCapture = (blob, dataUrl) => {
    onImageSelected(blob, dataUrl)
    setShowCamera(false)
  }

  return (
    <div className="object-input-card">
      <div className="object-input-header">
        <div className="object-badge">
          <span className="object-num">#{objectNumber}</span>
          <span className="object-label">{label}</span>
        </div>
        {previewUrl && (
          <span className="status-pill status-pill--ready">✓ Ready</span>
        )}
      </div>

      {!previewUrl ? (
        <div className="dual-input-options">
          {/* Option A: Drag and Drop Upload */}
          <div
            className={`upload-dropzone ${isDragging ? 'upload-dropzone--dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <div className="dropzone-icon">📁</div>
            <div className="dropzone-text">
              <strong>Upload Photo</strong>
              <p>Drag & drop or click to browse</p>
            </div>
            <span className="file-formats-tag">JPG, PNG, WEBP</span>
          </div>

          <div className="dual-input-divider">
            <span className="divider-line"></span>
            <span className="divider-badge">OR</span>
            <span className="divider-line"></span>
          </div>

          {/* Option B: Open Camera */}
          <div className="camera-trigger-zone" onClick={() => setShowCamera(true)}>
            <div className="camera-trigger-icon">📷</div>
            <div className="camera-trigger-text">
              <strong>Open Camera</strong>
              <p>Capture live from your webcam</p>
            </div>
            <button className="btn btn--camera-launch" type="button">
              Launch Viewfinder
            </button>
          </div>
        </div>
      ) : (
        /* Image Preview with Replace & Remove Controls */
        <div className="object-preview-container">
          <div className="object-preview-media">
            <img src={previewUrl} alt={`Object ${objectNumber}`} className="object-preview-img" />
            <div className="preview-overlay-tag">Object #{objectNumber} Loaded</div>
          </div>
          <div className="object-preview-controls">
            <button
              className="btn btn--secondary btn--sm"
              onClick={() => fileInputRef.current?.click()}
              title="Replace with another uploaded file"
            >
              🔄 Replace Photo
            </button>
            <button
              className="btn btn--secondary btn--sm"
              onClick={() => setShowCamera(true)}
              title="Replace with camera capture"
            >
              📷 Retake with Camera
            </button>
            <button
              className="btn btn--danger-ghost btn--sm"
              onClick={onImageRemoved}
              title="Remove this object photo"
            >
              🗑️ Remove
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
          </div>
        </div>
      )}

      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  )
}
