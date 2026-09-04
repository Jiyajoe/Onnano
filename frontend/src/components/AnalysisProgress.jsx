import React, { useState, useEffect } from 'react'

const PIPELINE_STAGES = [
  { icon: '🔍', label: 'Understanding image & quality check', pct: 12 },
  { icon: '🎯', label: 'Selecting object & refining mask', pct: 24 },
  { icon: '🤖', label: 'Identifying object & specific type', pct: 36 },
  { icon: '📐', label: 'Measuring visible geometry & contours', pct: 48 },
  { icon: '🔄', label: 'Correcting orientation & canonical alignment', pct: 60 },
  { icon: '🎨', label: 'Analyzing color distributions (HSV / LAB)', pct: 72 },
  { icon: '🧵', label: 'Analyzing surface texture & entropy', pct: 84 },
  { icon: '🔑', label: 'Matching features with RANSAC verification', pct: 94 },
  { icon: '🧠', label: 'Validating confidence & generating verdict', pct: 100 },
]

export default function AnalysisProgress({ title = 'Analyzing Physical Objects...' }) {
  const [currentStageIdx, setCurrentStageIdx] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStageIdx((prev) => (prev < PIPELINE_STAGES.length - 1 ? prev + 1 : prev))
    }, 240)

    return () => clearInterval(interval)
  }, [])

  const currentStage = PIPELINE_STAGES[currentStageIdx]

  return (
    <div className="analysis-progress-card">
      <div className="progress-scanner-halo">
        <div className="scanner-gavel">⚡</div>
        <div className="scanner-ring scanner-ring--1"></div>
        <div className="scanner-ring scanner-ring--2"></div>
      </div>

      <h2 className="progress-title">{title}</h2>
      <p className="progress-subtitle">Hybrid AI Semantic Understanding + Computer Vision Pipeline</p>

      {/* Main progress bar */}
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${currentStage.pct}%` }}
        ></div>
      </div>
      <div className="progress-percent-label">{currentStage.pct}% Complete</div>

      {/* 9 Pipeline steps list */}
      <div className="pipeline-steps-list">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isDone = idx < currentStageIdx
          const isCurrent = idx === currentStageIdx
          return (
            <div
              key={idx}
              className={`pipeline-step-item ${isDone ? 'step--done' : ''} ${
                isCurrent ? 'step--current' : ''
              }`}
            >
              <span className="step-icon">{stage.icon}</span>
              <span className="step-label">{stage.label}</span>
              <span className="step-status">
                {isDone ? '✓' : isCurrent ? '...' : ''}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
