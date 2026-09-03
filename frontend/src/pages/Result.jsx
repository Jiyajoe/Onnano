import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useSession } from '../App.jsx'
import { SingleResultCard, MultipleResultCard } from '../components/ResultCard.jsx'
import Verdict from '../components/Verdict.jsx'

export default function Result() {
  const navigate = useNavigate()
  const { session } = useSession()

  if (!session.analysis) {
    navigate('/mode')
    return null
  }

  const { analysis, mode } = session

  return (
    <div className="page page--result">
      <p className="page-eyebrow">Step 3 of 3</p>
      {mode === 'single' ? <SingleResultCard data={analysis} /> : <MultipleResultCard data={analysis} />}
      <Verdict text={analysis.verdict} disclaimer={analysis.disclaimer} />

      <div className="page-actions">
        <button className="btn btn--primary btn--large" onClick={() => navigate('/scan', { state: { verify: true } })}>
          {mode === 'single' ? 'Verify After Cutting' : 'Verify'}
        </button>
        <button className="btn btn--ghost" onClick={() => navigate('/mode')}>
          Start Over
        </button>
      </div>
    </div>
  )
}
