import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useSession } from '../App.jsx'
import FairnessScore from '../components/FairnessScore.jsx'
import Verdict from '../components/Verdict.jsx'

export default function Verify() {
  const navigate = useNavigate()
  const { session } = useSession()

  if (!session.verification) {
    navigate('/mode')
    return null
  }

  const { verification, mode } = session

  return (
    <div className="page page--result">
      <h1 className="page-title">Final Verdict</h1>

      <div className="result-card">
        {verification.annotated_image && (
          <img className="result-image" src={verification.annotated_image} alt="Verified division" />
        )}

        {mode === 'single' ? (
          <div className="split-bar">
            <div className="split-bar-piece split-bar-piece--a" style={{ flexGrow: verification.piece1_percentage }}>
              <span>Piece A</span>
              <strong>{verification.piece1_percentage.toFixed(1)}%</strong>
            </div>
            <div className="split-bar-piece split-bar-piece--b" style={{ flexGrow: verification.piece2_percentage }}>
              <span>Piece B</span>
              <strong>{verification.piece2_percentage.toFixed(1)}%</strong>
            </div>
          </div>
        ) : (
          <div className="group-grid">
            <div className="group-card group-card--a">
              <h3>Sibling A</h3>
              <p className="group-share">{verification.group_a_count} objects · {verification.group_a_percentage.toFixed(1)}%</p>
            </div>
            <div className="group-card group-card--b">
              <h3>Sibling B</h3>
              <p className="group-share">{verification.group_b_count} objects · {verification.group_b_percentage.toFixed(1)}%</p>
            </div>
          </div>
        )}

        {verification.note && <p className="result-hint">{verification.note}</p>}

        <FairnessScore score={verification.fairness_score} classification={verification.classification} />
      </div>

      <Verdict text={verification.verdict} disclaimer={verification.disclaimer} />

      <div className="page-actions">
        <button className="btn btn--primary btn--large" onClick={() => navigate('/mode')}>
          Fight Cancelled. Start Over 😂⚖️
        </button>
      </div>
    </div>
  )
}
