import React from 'react'
import { NavLink, Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <div className="brand-logo-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 3v18" />
              <path d="M3 12h18" />
              <circle cx="12" cy="12" r="3" fill="#ffb627" />
            </svg>
          </div>
          <div className="brand-text-group">
            <span className="brand-title">ഒന്നാണോ?</span>
            <span className="brand-badge">Same aano? AI parayatte</span>
          </div>
        </Link>

        <nav className="navbar-nav">
          <NavLink
            to="/split"
            className={({ isActive }) => `nav-link ${isActive ? 'nav-link--active' : ''}`}
          >
            <span className="nav-link-icon">✂️</span>
            <span>Split One Object</span>
          </NavLink>

          <NavLink
            to="/multiple"
            className={({ isActive }) => `nav-link ${isActive ? 'nav-link--active' : ''}`}
          >
            <span className="nav-link-icon">⚖️</span>
            <span>Divide Multiple Objects</span>
          </NavLink>
        </nav>
      </div>
    </header>
  )
}
