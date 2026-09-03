import React from 'react'

export default function FairnessScore({ score, classification }) {
  const pct = Math.max(0, Math.min(100, score))
  const tierClass =
    pct >= 98 ? 'tier-0' : pct >= 95 ? 'tier-1' : pct >= 90 ? 'tier-2' : 'tier-3'

  return (
    <div className={`fairness-score ${tierClass}`}>
      <div className="fairness-score-ring" style={{ '--pct': pct }}>
        <span className="fairness-score-number">{pct.toFixed(1)}</span>
        <span className="fairness-score-percent">%</span>
      </div>
      <div className="fairness-score-label">
        <span className="fairness-score-emoji">{classification?.emoji}</span>
        {classification?.label}
      </div>
    </div>
  )
}
