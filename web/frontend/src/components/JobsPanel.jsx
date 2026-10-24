import React, { useCallback, useEffect, useState } from 'react'
import { api, isLockMismatch, lockMismatchText } from '../api.js'
import { errMessage, clip } from '../util.js'

/** 入库任务队列：后台一句话入库的历史与状态，支持失败重试。 */
export default function JobsPanel({ refreshKey, onDataChanged }) {
  const [status, setStatus] = useState('')
  const [rows, setRows] = useState(null)
  const [available, setAvailable] = useState(true)
  const [busy, setBusy] = useState({})
  const [msg, setMsg] = useState('')

  const load = useCallback(async () => {
    try {
      const r = await api.jobs(status, 50)
      setRows(r.items || [])
      setAvailable(r.available !== false)
    } catch (e) {
      setMsg(errMessage(e))
    }
  }, [status])

  useEffect(() => { load() }, [load, refreshKey])

  const retry = useCallback(async (jobId) => {
    if (busy[jobId]) return
    setBusy((b) => ({ ...b, [jobId]: true }))
    setMsg('')
    try {
      await api.retryJob(jobId)
      setMsg(`已重新入队：${jobId}`)
      load()
      onDataChanged && onDataChanged()
    } catch (e) {
      setMsg(isLockMismatch(e) ? lockMismatchText(e) : errMessage(e))
    }
    setBusy((b) => ({ ...b, [jobId]: false }))
  }, [busy, load, onDataChanged])

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <div>
          <h2>入库任务队列 · ingest_jobs</h2>
          <p className="hint" style={{ margin: 0 }}>
            一句话入库（异步模式）提交后立即返回，状态持久化在 SQLite，页面重启后仍可查询与重试。
          </p>
        </div>
        <div className="row tight">
          <select style={{ width: 140 }} value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="">全部状态</option>
            <option value="pending">排队中</option>
            <option value="running">正在入库</option>
            <option value="done">成功</option>
            <option value="failed">失败</option>
          </select>
          <button className="btn sm" onClick={load}>刷新</button>
        </div>
      </div>

      {!available && <div className="empty">当前使用内存库，持久化队列不可用（SQLite 库下可用）。</div>}
      {rows && rows.length === 0 && available && <div className="empty">还没有入库任务</div>}

      <div className="list" style={{ marginTop: 12 }}>
        {(rows || []).map((j) => (
          <div className="item" key={j.job_id}>
            <div className="head">
              <span className={`pill ${j.status === 'done' ? 'ok' : j.status === 'failed' ? 'err' : 'accent2'}`}>
                {j.label || j.status}
              </span>
              <span className="pill">{j.kind}</span>
              <span className="pill">尝试 {j.attempts} 次</span>
              <span className="id">{j.job_id}</span>
              <button
                className={`btn sm ${j.retryable ? 'primary' : ''}`}
                disabled={!j.retryable || busy[j.job_id]}
                onClick={() => retry(j.job_id)}
              >
                {busy[j.job_id] ? <span className="spinner" /> : '重试'}
              </button>
            </div>
            <div className="body">{j.text}</div>
            {j.error && <div className="hint" style={{ color: 'var(--err)', marginTop: 6 }}>{j.error}</div>}
            {j.result && Object.keys(j.result).length > 0 && (
              <pre className="out" style={{ marginTop: 8, maxHeight: 200 }}>{clip(JSON.stringify(j.result, null, 2), 4000)}</pre>
            )}
            <div className="hint" style={{ margin: '6px 0 0' }}>创建 {j.created_at || '-'} · 更新 {j.updated_at || '-'}</div>
          </div>
        ))}
      </div>
      {msg && <pre className="out" style={{ marginTop: 10 }}>{msg}</pre>}
    </div>
  )
}