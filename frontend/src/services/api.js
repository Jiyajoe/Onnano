const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiError extends Error {
  constructor(message, code) {
    super(message)
    this.code = code
  }
}

async function postImage(path, blob, extraFields = {}) {
  const form = new FormData()
  form.append('file', blob, 'scan.jpg')
  Object.entries(extraFields).forEach(([key, value]) => {
    form.append(key, value)
  })

  let response
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: 'POST',
      body: form,
    })
  } catch (err) {
    throw new ApiError('Could not reach the AI backend. Please check your connection and try again.', 'backend_unavailable')
  }

  let data
  try {
    data = await response.json()
  } catch (err) {
    throw new ApiError('The server sent back something unexpected. Please try again.', 'invalid_response')
  }

  if (!response.ok) {
    throw new ApiError(data.detail || 'Something went wrong. Please try again.', 'server_error')
  }

  return data
}

export const api = {
  health: async () => {
    try {
      const res = await fetch(`${API_URL}/health`)
      return res.ok
    } catch {
      return false
    }
  },
  analyzeSingle: (blob) => postImage('/analyze/single', blob),
  analyzeMultiple: (blob, maxObjects = 20) => postImage('/analyze/multiple', blob, { max_objects: maxObjects }),
  verifySingle: (blob) => postImage('/verify/single', blob),
  verifyMultiple: (blob, expectedA = 0, expectedB = 0) =>
    postImage('/verify/multiple', blob, { expected_group_a: expectedA, expected_group_b: expectedB }),
}

export { ApiError, API_URL }
