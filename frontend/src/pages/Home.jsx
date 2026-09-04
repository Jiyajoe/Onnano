import React from 'react'
import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="home-hero-container">


      {/* Hero Headline */}
      <h1 className="home-main-heading">
        <span className="text-gradient" style={{ fontSize: '80px' }}>ഒന്നാണോ?
        </span>
      </h1>
      <p className="home-lead-text">
        Same aano? AI parayatte.
      </p>

      {/* Two Core Modes Cards */}
      <div className="home-modes-grid">
        {/* Mode 1 Card */}
        <Link to="/split" className="home-mode-card home-mode-card--split">
          <div className="card-top-icon">✂️</div>
          <div className="card-badge">MODE 1</div>
          <h2 className="card-title">Split One Object</h2>
          <p className="card-desc">
            Upload or take a photo of one object.
            AI finds the object, straightens it, and splits it into equal parts. </p>

          <div className="card-feature-tags">
            <span>📷 Upload or Capture</span>
            <span>🔄 Straighten Object</span>
            <span>✂️ Split into Equal Parts</span>
          </div>

          <div className="card-cta">
            <span>Launch Single Slicer</span>
            <span className="cta-arrow">➔</span>
          </div>
        </Link>

        {/* Mode 2 Card */}
        <Link to="/multiple" className="home-mode-card home-mode-card--multi">
          <div className="card-top-icon">⚖️</div>
          <div className="card-badge card-badge--multi">MODE 2</div>
          <h2 className="card-title">Divide Multiple Objects</h2>
          <p className="card-desc">
            AI checks how similar they are and tells you how closely they match.   </p>

          <div className="card-feature-tags">
            <span>⚖️ Compare up to 8 objects</span>
            <span>🔑 Match based on shape, color, texture</span>
            <span>🤖 Malayalam AI Verdict</span>
          </div>

          <div className="card-cta card-cta--multi">
            <span>Launch Multi-Object Engine</span>
            <span className="cta-arrow">➔</span>
          </div>
        </Link>
      </div>

      {/* Tech Stack Banner */}
      <div className="home-tech-banner">
        <span className="tech-label">✨ AI + kurachu magic + kurachu thalla..</span>

      </div>
    </div>
  )
}
