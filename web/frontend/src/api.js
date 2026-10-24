/** 知识星云 · 前端 API 客户端。
 *
 * 只封装 web/app.py 里真实注册的端点；调用方不自己拼 URL。
 * 所有写请求都可能先吃到 409 embedding_lock_mismatch（换模型/换维度后
 * 第一次写库），统一交给 withEmbeddingGuard：询问用户、带 confirm_rebuild
 * 重试一次，语义与旧单文件前端的 withEmbeddingGuard 一致。
 */

const BASE = import.meta.env.VITE_API_BASE || '/api'

/** 后端 HTTPException(detail={...}) 时的可读错误。 */
function detailMessage(data, fallback) {
  if (!data) return fallback
  const d = data.detail
  if (typeof d === 'string') return d
  if (d && typeof d === 'object') return d.message || d.code || JSON.stringify(d)
  return data.error || data.message || fallback
}

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

export function isLockMismatch(err) {
  const d = err && err.payload && err.payload.detail
  return Boolean(err && err.status === 409 && d && d.code === 'embedding_lock_mismatch')
}

export function lockMismatchText(err) {
  const d = err.payload && err.payload.detail
  if (!d) return err.message
  const locked = d.locked || {}
  const current = d.current || {}
  return [
    d.message || '嵌入锁定不一致',
    `库内锁定：${locked.model || '?'} / ${locked.dimension || '?'} 维`,
    `当前配置：${current.model || '?'} / ${current.dimension || '?'} 维`,
    '确认后将重建向量投影并全量重灌（confirm_rebuild=true）。',
  ].join('\n')
}

async function req(path, { method = 'GET', body, form, timeout = 90000, query } = {}) {
  const url = new URL(BASE + path, window.location.origin)
  for (const [k, v] of Object.entries(query || {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
  }
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeout)
  try {
    const res = await fetch(url.pathname + url.search, {
      method,
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : form,
      signal: ctrl.signal,
    })
    const text = await res.text()
    let data = null
    try { data = text ? JSON.parse(text) : null } catch { data = { raw: text } }
    if (!res.ok) {
      throw new ApiError(detailMessage(data, `HTTP ${res.status}`), res.status, data)
    }
    return data
  } catch (e) {
    if (e instanceof ApiError) throw e
    if (e.name === 'AbortError') throw new ApiError(`请求超时（>${Math.round(timeout / 1000)}s）`, 0, null)
    if (e.name === 'TypeError') throw new ApiError('无法连接后端，请确认 FastAPI 已启动（python -m web.app）', 0, null)
    throw e
  } finally {
    clearTimeout(timer)
  }
}

/** 给写请求套嵌入锁闸门：409 → 询问 → confirm_rebuild=true 重试一次。 */
export async function withEmbeddingGuard(action, { confirm = true } = {}) {
  try {
    return await action(false)
  } catch (err) {
    if (!isLockMismatch(err)) throw err
    if (!confirm || !window.confirm(lockMismatchText(err))) throw err
    return action(true)
  }
}

export const api = {
  /* ---- 星云图 / 检索 ---- */
  health: () => req('/health'),
  graph: (since = -1, at = '') => req('/graph', { query: { since, at } }),
  graphRag: (payload) => req('/graph-rag', { method: 'POST', body: payload, timeout: 180000 }),

  /* ---- 对话 ---- */
  chat: (payload) => req('/chat', { method: 'POST', body: payload, timeout: 300000 }),

  /* ---- 入库 ---- */
  ingestFile: (file, opts = {}) => withEmbeddingGuard(
    (confirmRebuild) => {
      const form = new FormData()
      form.append('file', file)
      return req('/ingest', {
        method: 'POST', form, timeout: 600000,
        query: confirmRebuild ? { confirm_rebuild: 'true' } : {},
      })
    }, opts),
  addKnowledge: (payload, opts = {}) => withEmbeddingGuard(
    (confirmRebuild) => req('/knowledge', {
      method: 'POST', body: payload, timeout: 600000,
      query: confirmRebuild ? { confirm_rebuild: 'true' } : {},
    }), opts),
  addImageKnowledge: (payload, opts = {}) => withEmbeddingGuard(
    (confirmRebuild) => {
      const form = new FormData()
      form.append('file', payload.file)
      form.append('text', payload.text || '')
      form.append('captured_at', payload.captured_at || '')
      return req('/knowledge/image', {
        method: 'POST', form, timeout: 600000,
        query: confirmRebuild ? { confirm_rebuild: 'true' } : {},
      })
    }, opts),
  addFact: (body) => req('/facts', { method: 'POST', body }),

  /* ---- 入库任务队列 ---- */
  jobs: (status = '', limit = 30) => req('/knowledge/jobs', { query: { status, limit } }),
  retryJob: (jobId) => req(`/knowledge/jobs/${encodeURIComponent(jobId)}/retry`, { method: 'POST', timeout: 600000 }),

  /* ---- 文档中心 ---- */
  documents: (params = {}) => req('/documents', { query: params }),
  document: (documentId) => req(`/documents/${encodeURIComponent(documentId)}`, { timeout: 120000 }),
  revectorize: (documentId) => withEmbeddingGuard(
    (confirmRebuild) => req(`/documents/${encodeURIComponent(documentId)}/revectorize`, {
      method: 'POST', timeout: 600000,
      query: confirmRebuild ? { confirm_rebuild: 'true' } : {},
    }),
  ),
  exportUrl: () => BASE + '/export',
  importFile: (file) => withEmbeddingGuard(
    (confirmRebuild) => {
      const form = new FormData()
      form.append('file', file)
      return req('/import', {
        method: 'POST', form, timeout: 600000,
        query: confirmRebuild ? { confirm_rebuild: 'true' } : {},
      })
    },
  ),

  /* ---- 监控 / 运维 ---- */
  stats: () => req('/stats'),
  reconcile: () => req('/reconcile', { timeout: 120000 }),
  reconcileRepair: (repair) => req('/reconcile', { method: 'POST', body: { repair } }),
  seed: () => req('/seed', { method: 'POST', timeout: 300000 }),
  rebuildEmbedding: () => req('/embedding/rebuild', { method: 'POST', query: { confirm_rebuild: 'true' }, timeout: 600000 }),
}

export default api