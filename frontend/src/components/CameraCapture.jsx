import React, { useRef, useState, useEffect } from 'react'

export default function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [stream, setStream] = useState(null)
  const [capturedDataUrl, setCapturedDataUrl] = useState(null)
  const [capturedBlob, setCapturedBlob] = useState(null)
  const [cameraError, setCameraError] = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let activeStream = null
    async function startCamera() {
      setIsLoading(true)
      setCameraError(null)
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })
        activeStream = mediaStream
        setStream(mediaStream)
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(() => {})
          }
          videoRef.current.play().catch(() => {})
        }
        setIsLoading(false)
      } catch (err) {
        setIsLoading(false)
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setCameraError('Camera permission was denied. Please allow camera access in your browser settings, or upload a photo instead.')
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setCameraError('No camera found on this device. Please upload an image instead.')
        } else {
          setCameraError('Camera is currently unavailable. Please upload an image instead.')
        }
      }
    }

    startCamera()

    return () => {
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  const handleTakePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current

    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    canvas.toBlob(
      (blob) => {
        if (blob) {
          const dataUrl = canvas.toDataURL('image/jpeg', 0.92)
          setCapturedBlob(blob)
          setCapturedDataUrl(dataUrl)
        }
      },
      'image/jpeg',
      0.92
    )
  }

  const handleRetake = () => {
    setCapturedBlob(null)
    setCapturedDataUrl(null)
  }

  const handleUsePhoto = () => {
    if (capturedBlob && capturedDataUrl) {
      onCapture(capturedBlob, capturedDataUrl)
      if (onClose) onClose()
    }
  }

  return (
    <div className="camera-modal-backdrop">
      <div className="camera-modal-card">
        <div className="camera-modal-header">
          <div className="camera-header-left">
            <span className="camera-live-dot"></span>
            <h3>Live Camera Capture</h3>
          </div>
          <button className="camera-close-btn" onClick={onClose} title="Close camera">
            ✕
          </button>
        </div>

        <div className="camera-viewport-wrap">
          {isLoading && (
            <div className="camera-state-box">
              <div className="camera-spinner"></div>
              <p>Initializing camera...</p>
            </div>
          )}

          {cameraError && (
            <div className="camera-error-box">
              <div className="camera-error-icon">📷⚠️</div>
              <h4>Camera Unavailable</h4>
              <p>{cameraError}</p>
              <button className="btn btn--primary" onClick={onClose} style={{ marginTop: '12px' }}>
                Use Photo Upload Instead
              </button>
            </div>
          )}

          {!cameraError && !capturedDataUrl && (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-live-video"
              />
              <div className="camera-viewfinder-overlay">
                <div className="viewfinder-grid">
                  <div className="viewfinder-crosshair"></div>
                </div>
                <p className="viewfinder-hint">Center the physical object inside the frame</p>
              </div>
            </>
          )}

          {capturedDataUrl && (
            <div className="camera-preview-captured">
              <img src={capturedDataUrl} alt="Captured preview" className="camera-preview-img" />
              <div className="captured-badge">✓ Captured Frame</div>
            </div>
          )}

          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {!cameraError && (
          <div className="camera-modal-footer">
            {!capturedDataUrl ? (
              <button className="btn btn--shutter" onClick={handleTakePhoto} disabled={isLoading}>
                <span className="shutter-inner"></span>
                <span>Capture Object</span>
              </button>
            ) : (
              <div className="camera-actions-row">
                <button className="btn btn--ghost" onClick={handleRetake}>
                  🔄 Retake
                </button>
                <button className="btn btn--primary btn--large" onClick={handleUsePhoto}>
                  ✓ Use This Photo
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
