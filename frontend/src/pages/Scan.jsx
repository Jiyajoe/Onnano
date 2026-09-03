import React, { useRef, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useSession } from '../App.jsx'
import Camera from '../components/Camera.jsx'
import AnalysisLoader from '../components/AnalysisLoader.jsx'
import { api, ApiError } from '../services/api.js'

const ERROR_TEXT = {
  camera_permission_denied: 'Camera permission was denied. Please allow camera access to scan objects.',
  no_camera: 'No camera was found on this device.',
}

export default function Scan() {
  const navigate = useNavigate()
  const location = useLocation()
  const { session, updateSession } = useSession()
  const isVerifyFlow = location.state?.verify === true

  const cameraRef = useRef(null)
  const [status, setStatus] = useState('idle') // idle | loading | error
  const [error, setError] = useState(null)

  if (!session.mode) {
    navigate('/mode')
    return null
  }

  const instructions = session.mode === 'single'
    ? 'Place the object clearly inside the frame, on a plain background.'
    : 'Place all the objects clearly inside the frame, spaced apart.'

  const handleCameraError = (code) => {
    setStatus('error')
    setError(ERROR_TEXT[code] || 'Something went wrong with the camera.')
  }

  const handleScan = async () => {
    if (!cameraRef.current) return
    setStatus('loading')
    setError(null)
    try {
      const { blob, dataUrl } = await cameraRef.current.capture()
      if (!blob) throw new ApiError('Could not capture a frame. Try again.', 'capture_failed')

      if (isVerifyFlow) {
        const data = session.mode === 'single'
          ? await api.verifySingle(blob)
          : await api.verifyMultiple(blob, session.analysis?.group_a?.length || 0, session.analysis?.group_b?.length || 0)
        updateSession({ verification: data, verifyPreviewImage: dataUrl })
        navigate('/verify', { state: { justScanned: true } })
      } else {
        const data = session.mode === 'single'
          ? await api.analyzeSingle(blob)
          : await api.analyzeMultiple(blob)
        updateSession({ analysis: data, previewImage: dataUrl })
        navigate('/result')
      }
    } catch (err) {
      setStatus('error')
      setError(err instanceof ApiError ? err.message : 'Something went wrong while processing the image. Please try again.')
    }
  }

  return (
    <div className="page page--scan">
      <p className="page-eyebrow">Step 2 of 3</p>
      <h1 className="page-title">{isVerifyFlow ? 'Scan the result' : 'Point the camera'}</h1>

      {status === 'loading' ? (
        <AnalysisLoader />
      ) : (
        <>
          <Camera ref={cameraRef} instructions={instructions} onError={handleCameraError} />
          {status === 'error' && error && <p className="scan-error">{error}</p>}
          <button className="btn btn--primary btn--large" onClick={handleScan}>
            Scan
          </button>
        </>
      )}
    </div>
  )
}
