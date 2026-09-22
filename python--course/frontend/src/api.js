const BASE = import.meta.env.VITE_API_BASE || '/api'

async function req(path, { method = 'GET', body, timeout = 90000 } = {}) {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), timeout)
  try {
    const res = await fetch(BASE + path, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    })
    const text = await res.text()
    let data = null
    try { data = text ? JSON.parse(text) : null } catch { data = { raw: text } }
    if (!res.ok) {
      const msg =
        (data && (data.detail?.message || data.detail || data.error || data.message)) ||
        `HTTP ${res.status}`
      const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
      err.payload = data
      err.status = res.status
      throw err
    }
    return data
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`请求超时（>${Math.round(timeout / 1000)}s）`)
    if (e.name === 'TypeError') throw new Error('无法连接后端，请确认 FastAPI 已启动')
    throw e
  } finally {
    clearTimeout(t)
  }
}

export const api = {
  health: () => req('/health'),
  tools: () => req('/tools'),
  rediscover: () => req('/tools/rediscover', { method: 'POST' }),
  callTool: (name, args = {}, extra = {}) =>
    req(`/tools/${encodeURIComponent(name)}/call`, { method: 'POST', body: { args, ...extra } }),
  callMany: (calls) => req('/tools/call_many', { method: 'POST', body: { calls }, timeout: 180000 }),
  rediscoverTools: () => req('/tools/rediscover', { method: 'POST' }),

  pendingChunks: (limit = 20, previewChars = 10) =>
    req(`/chunks/pending?limit=${limit}&preview_chars=${previewChars}`),
  reingest: (chunkId) =>
    req(`/chunks/${chunkId}/reingest`, { method: 'POST', timeout: 330000 }),

  documents: () => req('/documents?limit=60'),
  chunks: () => req('/chunks?limit=60'),
  graph: () => req('/graph', { timeout: 90000 }),
  stats: () => req('/stats', { timeout: 90000 }),
  llmTest: () => req('/llm/test', { timeout: 180000 }),
  parse: (raw) => req('/llm/parse', { method: 'POST', body: { raw } }),

  ingestSentence: (text, source = 'sentence') =>
    req('/ingest/sentence', { method: 'POST', body: { text, source }, timeout: 660000 }),
}

/** SSE 一句话入库：实时推送 LLM 流式增量与各阶段事件 */
export function ingestSentenceStream(text, source = 'sentence', onEvent, onDone) {
  const ctrl = new AbortController()
  ;(async () => {
    try {
      const res = await fetch(BASE + '/ingest/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source }),
        signal: ctrl.signal,
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let idx
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const chunk = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          if (chunk.startsWith(':')) continue // keep-alive
          const line = chunk.replace(/^data:\s*/, '')
          try {
            const evt = JSON.parse(line)
            if (evt.event === 'done') { onDone?.(evt.data); return }
            onEvent?.(evt.event, evt.data)
          } catch { /* 忽略不完整行 */ }
        }
      }
      onDone?.({ ok: false, error: '流已结束' })
    } catch (e) {
      if (e.name !== 'AbortError') onDone?.({ ok: false, error: e.message })
    }
  })()
  return () => ctrl.abort()
}

export default api
