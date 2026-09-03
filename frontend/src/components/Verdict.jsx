import React from 'react'

export default function Verdict({ text, disclaimer }) {
  return (
    <div className="verdict-box">
      <p className="verdict-text">{text}</p>
      {disclaimer && <p className="verdict-disclaimer">{disclaimer}</p>}
    </div>
  )
}
