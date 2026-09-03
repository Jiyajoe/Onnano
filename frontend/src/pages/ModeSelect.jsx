import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useSession } from '../App.jsx'
import ModeSelector from '../components/ModeSelector.jsx'

export default function ModeSelect() {
  const navigate = useNavigate()
  const { updateSession } = useSession()

  const handleSelect = (mode) => {
    updateSession({ mode, analysis: null, verification: null, previewImage: null })
    navigate('/scan')
  }

  return (
    <div className="page">
      <p className="page-eyebrow">Step 1 of 3</p>
      <h1 className="page-title">What are we dividing today?</h1>
      <p className="page-lead">Pick the mode that matches what's on the table.</p>
      <ModeSelector onSelect={handleSelect} />
    </div>
  )
}
