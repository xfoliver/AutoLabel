const BASE = import.meta.env.VITE_API_BASE || ''

export async function fetchJson(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}: ${text}`)
  }
  return res.json()
}

export function createRunStream(prompt, onMessage) {
  const url = `${BASE}/api/run?prompt=${encodeURIComponent(prompt)}`
  const es = new EventSource(url)

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
      if (data.type === 'done' || data.type === 'error') {
        es.close()
      }
    } catch {
      onMessage({ type: 'log', line: event.data })
    }
  }

  es.onerror = () => {
    onMessage({ type: 'error', message: 'SSE 连接异常' })
    es.close()
  }

  return () => es.close()
}

export const api = {
  health: () => fetchJson('/api/health'),
  tasks: () => fetchJson('/api/tasks'),
  taskDetail: (id) => fetchJson(`/api/tasks/${id}`),
  datasets: () => fetchJson('/api/datasets'),
  models: () => fetchJson('/api/models'),
  templates: () => fetchJson('/api/templates'),
}
