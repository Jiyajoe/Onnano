/**
 * api.js - Client API service for Onnano AI/CV endpoints.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export class ApiError extends Error {
  constructor(message, status = 500) {
    super(message)
    this.status = status
  }
}

export const api = {
  health: async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      return res.ok
    } catch {
      return false
    }
  },

  /**
   * Single Object AI Analysis & N-way Slicing
   */
  analyzeObject: async (imageBlob, partsCount = 4) => {
    const form = new FormData()
    form.append('file', imageBlob, 'object.jpg')
    form.append('parts', partsCount.toString())

    let res
    try {
      res = await fetch(`${API_BASE}/api/analyze-object`, {
        method: 'POST',
        body: form,
      })
    } catch {
      throw new ApiError('Cannot connect to AI Backend server. Please ensure the backend is running.')
    }

    const data = await res.json()
    if (!res.ok) {
      throw new ApiError(data.detail || 'Failed to analyze object.', res.status)
    }
    return data
  },

  /**
   * Re-slice single object with custom parts count
   */
  divideObject: async (imageBlob, partsCount = 4) => {
    const form = new FormData()
    form.append('file', imageBlob, 'object.jpg')
    form.append('parts', partsCount.toString())

    let res
    try {
      res = await fetch(`${API_BASE}/api/divide-object`, {
        method: 'POST',
        body: form,
      })
    } catch {
      throw new ApiError('Cannot connect to AI Backend server.')
    }

    const data = await res.json()
    if (!res.ok) {
      throw new ApiError(data.detail || 'Failed to divide object.', res.status)
    }
    return data
  },

  /**
   * Multi-Object Comparison across N images
   */
  compareObjects: async (imageBlobsList) => {
    const form = new FormData()
    imageBlobsList.forEach((blob, idx) => {
      form.append('files', blob, `object_${idx + 1}.jpg`)
    })

    let res
    try {
      res = await fetch(`${API_BASE}/api/compare-objects`, {
        method: 'POST',
        body: form,
      })
    } catch {
      throw new ApiError('Cannot connect to AI Backend server. Please check your connection.')
    }

    const data = await res.json()
    if (!res.ok) {
      throw new ApiError(data.detail || 'Failed to compare objects.', res.status)
    }
    return data
  },
}
