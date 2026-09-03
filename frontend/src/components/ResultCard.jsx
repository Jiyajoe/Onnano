import React from 'react'
import FairnessScore from './FairnessScore.jsx'

export function SingleResultCard({ data }) {
  return (
    <div className="result-card">
      <h2 className="result-heading">Fairest Cut</h2>
      {data.annotated_image && (
        <img className="result-image" src={data.annotated_image} alt="Detected object with recommended cutting line" />
      )}
      <div className="split-bar">
        <div className="split-bar-piece split-bar-piece--a" style={{ flexGrow: data.piece1_percentage }}>
          <span>Piece A</span>
          <strong>{data.piece1_percentage.toFixed(1)}%</strong>
        </div>
        <div className="split-bar-piece split-bar-piece--b" style={{ flexGrow: data.piece2_percentage }}>
          <span>Piece B</span>
          <strong>{data.piece2_percentage.toFixed(1)}%</strong>
        </div>
      </div>
      <p className="result-hint">✂️ Cut along the red line shown above.</p>
      <FairnessScore score={data.fairness_score} classification={data.classification} />
    </div>
  )
}

export function MultipleResultCard({ data }) {
  return (
    <div className="result-card">
      <h2 className="result-heading">{data.object_count} Objects Detected</h2>
      {data.annotated_image && (
        <img className="result-image" src={data.annotated_image} alt="Detected objects split into two groups" />
      )}
      <div className="group-grid">
        <div className="group-card group-card--a">
          <h3>Sibling A</h3>
          <div className="group-chips">
            {data.group_a.map((obj) => (
              <span key={obj.id} className="group-chip">#{obj.id}</span>
            ))}
          </div>
          <p className="group-share">Estimated Share: {data.group_a_percentage.toFixed(1)}%</p>
        </div>
        <div className="group-card group-card--b">
          <h3>Sibling B</h3>
          <div className="group-chips">
            {data.group_b.map((obj) => (
              <span key={obj.id} className="group-chip">#{obj.id}</span>
            ))}
          </div>
          <p className="group-share">Estimated Share: {data.group_b_percentage.toFixed(1)}%</p>
        </div>
      </div>
      <p className="result-hint">Strategy used: <strong>{data.strategy.replace('_', ' ')}</strong></p>
      <FairnessScore score={data.fairness_score} classification={data.classification} />
    </div>
  )
}
