import React, { Component } from 'react'
import { HashRouter, Routes, Route, Link } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import SplitOneObject from './pages/SplitOneObject.jsx'
import MultipleObjects from './pages/MultipleObjects.jsx'

class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('UI Rendering Error caught by ErrorBoundary:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '60px 20px', maxWidth: '640px', margin: '0 auto', textAlign: 'center', color: '#f4f1ff' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⚠️</div>
          <h2 style={{ fontSize: '24px', marginBottom: '12px' }}>Something went wrong while rendering this view</h2>
          <p style={{ color: '#a3a1cf', marginBottom: '24px' }}>{this.state.error?.message || 'An unexpected rendering error occurred.'}</p>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null })
              window.location.href = '#/'
            }}
            style={{
              padding: '12px 24px',
              backgroundColor: '#ffb627',
              color: '#15183d',
              border: 'none',
              borderRadius: '8px',
              fontWeight: '700',
              cursor: 'pointer',
            }}
          >
            Return to Home Screen
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <Navbar />
        <main className="app-main-content">
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/split" element={<SplitOneObject />} />
              <Route path="/multiple" element={<MultipleObjects />} />
              <Route path="*" element={<Home />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </HashRouter>
  )
}
