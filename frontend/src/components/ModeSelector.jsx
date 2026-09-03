import React from 'react'

export default function ModeSelector({ onSelect }) {
  return (
    <div className="mode-grid">
      <button className="mode-card mode-card--split" onClick={() => onSelect('single')}>
        <span className="mode-card-icon">🍫</span>
        <span className="mode-card-title">Split One Object</span>
        <span className="mode-card-sub">One object, two siblings.</span>
        <span className="mode-card-examples">Chocolate · Cake · Fruit · Pizza</span>
      </button>
      <button className="mode-card mode-card--divide" onClick={() => onSelect('multiple')}>
        <span className="mode-card-icon">🧸</span>
        <span className="mode-card-title">Divide Multiple Objects</span>
        <span className="mode-card-sub">Many objects, two fair groups.</span>
        <span className="mode-card-examples">Toys · Pencils · Cards · Candies</span>
      </button>
    </div>
  )
}
