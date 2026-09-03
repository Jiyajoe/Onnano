import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from 'react'

const Camera = forwardRef(function Camera({ instructions, onError }, ref) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [permissionState, setPermissionState] = useState('requesting') // requesting | granted | denied | no-camera

  useEffect(() => {
    let cancelled = false

    async function start() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setPermissionState('no-camera')
        onError && onError('no_camera')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setPermissionState('granted')
        setReady(true)
      } catch (err) {
        setPermissionState('denied')
        onError && onError('camera_permission_denied')
      }
    }

    start()
    return () => {
      cancelled = true
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useImperativeHandle(ref, () => ({
    capture: () => {
      const video = videoRef.current
      if (!video || !ready) return null
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve({ blob, dataUrl: canvas.toDataURL('image/jpeg', 0.92) }), 'image/jpeg', 0.92)
      })
    },
  }))

  if (permissionState === 'no-camera') {
    return (
      <div className="camera-fallback">
        <p>No camera was found on this device.</p>
        <p className="camera-fallback-sub">You can still try again after connecting a camera.</p>
      </div>
    )
  }

  if (permissionState === 'denied') {
    return (
      <div className="camera-fallback">
        <p>Camera permission was denied.</p>
        <p className="camera-fallback-sub">Allow camera access in your browser settings, then reload.</p>
      </div>
    )
  }

  return (
    <div className="camera-wrap">
      <video ref={videoRef} className="camera-video" playsInline muted />
      <div className="camera-overlay">
        <div className="camera-guide" />
        <p className="camera-instructions">{instructions}</p>
      </div>
      {!ready && <div className="camera-loading">Starting camera…</div>}
    </div>
  )
})

export default Camera
