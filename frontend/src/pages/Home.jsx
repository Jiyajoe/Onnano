import React from 'react'
import { useNavigate } from 'react-router-dom'

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="page page--home">
      <div className="home-badge">AI SIBLING FAIRNESS JUDGE</div>
      <h1 className="home-title">ഒന്നാണോ?</h1>
      <p className="home-subtitle">Same aano? AI parayatte.</p>
      <p className="home-tagline">Sibling fight undo? AI decide cheyyatte.</p>

      <div className="home-scale" aria-hidden="true">
        <svg viewBox="0 0 220 160" className="scale-svg">
          <line x1="110" y1="10" x2="110" y2="140" stroke="var(--gold)" strokeWidth="6" strokeLinecap="round" />
          <line x1="30" y1="40" x2="190" y2="40" stroke="var(--gold)" strokeWidth="6" strokeLinecap="round" />
          <line x1="30" y1="40" x2="30" y2="80" stroke="var(--gold)" strokeWidth="4" strokeLinecap="round" />
          <line x1="190" y1="40" x2="190" y2="70" stroke="var(--gold)" strokeWidth="4" strokeLinecap="round" />
          <path d="M10 80 A20 20 0 0 0 50 80 Z" fill="var(--mint)" />
          <path d="M170 70 A20 20 0 0 0 210 70 Z" fill="var(--coral)" />
          <circle cx="110" cy="140" r="10" fill="var(--gold)" />
          <rect x="95" y="150" width="30" height="8" rx="3" fill="var(--gold)" />
        </svg>
      </div>

      <button className="btn btn--primary btn--large" onClick={() => navigate('/mode')}>
        ⚖️ Start Fair Share
      </button>
    </div>
  )
}
