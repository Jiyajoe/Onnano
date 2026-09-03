import React, { useState } from 'react'
import { HashRouter, Routes, Route, Outlet, useOutletContext } from 'react-router-dom'
import Home from './pages/Home.jsx'
import ModeSelect from './pages/ModeSelect.jsx'
import Scan from './pages/Scan.jsx'
import Result from './pages/Result.jsx'
import Verify from './pages/Verify.jsx'

function Shell() {
  // Shared session state carried across the flow: which mode was picked,
  // and the most recent analysis result (so Result/Verify can read it
  // without re-fetching or hitting the back button into a dead end).
  const [session, setSession] = useState({
    mode: null,          // 'single' | 'multiple'
    previewImage: null,  // data URL of what the user scanned
    analysis: null,      // last /analyze response
    verification: null,  // last /verify response
  })

  const updateSession = (patch) => setSession((prev) => ({ ...prev, ...patch }))

  return <Outlet context={{ session, updateSession }} />
}

export function useSession() {
  return useOutletContext()
}

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<Home />} />
          <Route path="/mode" element={<ModeSelect />} />
          <Route path="/scan" element={<Scan />} />
          <Route path="/result" element={<Result />} />
          <Route path="/verify" element={<Verify />} />
        </Route>
      </Routes>
    </HashRouter>
  )
}
