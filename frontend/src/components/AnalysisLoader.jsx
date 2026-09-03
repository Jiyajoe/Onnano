import React, { useEffect, useState } from 'react'

const STAGES = [
  '🔍 Detecting objects...',
  '📐 Measuring...',
  '⚖️ Comparing...',
  '🧮 Finding fairest division...',
  '🤖 Preparing verdict...',
]

export default function AnalysisLoader() {
  const [stage, setStage] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setStage((s) => (s + 1) % STAGES.length)
    }, 750)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="analysis-loader">
      <div className="gavel-spinner" aria-hidden="true">⚖️</div>
      <p className="analysis-stage">{STAGES[stage]}</p>
      <p className="analysis-personality">AI has entered the family dispute.</p>
    </div>
  )
}
