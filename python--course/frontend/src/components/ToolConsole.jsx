import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

/**
 * 工具台：从 /api/tools 动态拿工具 schema，自动生成调用表单。
 * 新增工具无需改前端 —— discover 发现后这里自动多一个按钮。
 */
export default function ToolConsole({ tools, onRefreshTools }) {
  const [sel, setSel] = useState(null)
  const [form, setForm] = useState({})
  const [out, setOut] = useState(null)
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  const [promptPreview, setPromptPreview] = useState('')

  useEffect(() => {
    api.tools().then((r) => setPromptPreview(r.prompt_preview || '')).catch(() => {})
  }, [])

  useEffect(() => {
    if (tools && tools.length && !sel) setSel(tools[0].name)
  }, [tools, sel])

  const tool = (tools || []).find((t) => t.name === sel)
  useEffect(() => {
    if (!tool) return
    const init = {}
    tool.params.forEach((p) => { if (p.default !== null && p.default !== undefined) init[p.name] = p.default })
    setForm(init)
    setOut(null); setErr('')
  }, [tool])

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }))

  const coerce = (p, raw) => {
    if (raw === '' || raw === null || raw === undefined) return undefined
    if (p.type === 'integer') return Number(raw)
    if (p.type === 'number') return Number(raw)
    if (p.type === 'boolean') return raw === true || raw === 'true'
    if (p.type === 'array') {
      if (typeof raw === 'string') return raw.split(/[,，]/).map(s => s.trim()).filter(Boolean)
      return raw
    }
    return raw
  }

  const run = async () => {
    if (!tool) return
    setBusy(true); setErr(''); setOut(null)
    const args = {}
    tool.params.forEach((p) => {
      const v = coerce(p, form[p.name])
      if (v !== undefined && v !== '') args[p.name] = v
      else if (p.type === 'boolean') args[p.name] = false
    })
    try {
      const r = await api.callTool(tool.name, args)
      setOut(r)
    } catch (e) {
      setErr(e.message)
      setOut(e.payload || null)
    }
    setBusy(false)
  }

  const grouped = (tools || []).reduce((acc, t) => {
    const g = (t.tags && t.tags[0]) || '其他'
    ;(acc[g] = acc[g] || []).push(t)
    return acc
  }, {})

  return (
    <>
      <div className="grid2">
        <div className="card">
          <h2>工具台 · 动态表单</h2>
          <p className="hint">
            表单由后端 <code>/api/tools</code> 返回的 schema 自动生成（含必填/可选、默认值、枚举、取值范围）。
            在 <code>app/tools/</code> 下新增一个 .py 文件，点「重新发现」即可自动登记并出现在这里。
          </p>
          <div className="row" style={{ marginBottom: 10 }}>
            <button className="btn sm" onClick={async () => { await api.rediscoverTools(); onRefreshTools && onRefreshTools() }}>
              重新发现工具
            </button>
            <span className="pill">已登记 {(tools || []).length} 个</span>
          </div>
          {Object.entries(grouped).map(([g, list]) => (
            <div key={g} style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 10.5, color: 'var(--txt-dim)', marginBottom: 5, letterSpacing: 0.6 }}>{g.toUpperCase()}</div>
              <div className="tool-pick">
                {list.map((t) => (
                  <button key={t.name} className={sel === t.name ? 'sel' : ''} onClick={() => setSel(t.name)}>
                    <b>{t.name}</b>
                    <i>{t.params.filter(p => p.required).length} 必填 / {t.params.length} 参数</i>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="card">
          <h2>{sel || '选择工具'}</h2>
          <p className="hint" style={{ minHeight: 32 }}>{tool ? tool.description : ''}</p>
          {tool && tool.params.length === 0 && <p className="hint">该工具无参数，直接点执行。</p>}
          {tool && tool.params.map((p) => (
            <label className="fld" key={p.name}>
              <span>
                {p.name}
                {p.required && <span className="req">*</span>}
                <span className="en">{p.type}{p.enum ? ' · ' + p.enum.join('/') : ''}{p.optional === false ? '' : ' · 可省略'}</span>
              </span>
              {p.type === 'boolean' ? (
                <select value={String(!!form[p.name])} onChange={(e) => set(p.name, e.target.value === 'true')}>
                  <option value="true">true</option><option value="false">false</option>
                </select>
              ) : p.enum ? (
                <select value={form[p.name] ?? ''} onChange={(e) => set(p.name, e.target.value)}>
                  <option value="">（省略，用默认值）</option>
                  {p.enum.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              ) : (p.type === 'array' || p.type === 'object' || (p.description && p.description.length > 40)) ? (
                <textarea value={form[p.name] ?? ''} onChange={(e) => set(p.name, e.target.value)}
                          placeholder={`${p.description}${p.required ? '（必填）' : '（可选，可省略）'}`} />
              ) : (
                <input value={form[p.name] ?? ''} onChange={(e) => set(p.name, e.target.value)}
                       type={p.type === 'integer' || p.type === 'number' ? 'number' : 'text'}
                       placeholder={`${p.description}${p.default !== null ? `（默认 ${p.default}）` : ''}`} />
              )}
            </label>
          ))}
          <div className="row">
            <button className="btn primary" onClick={run} disabled={!tool || busy}>
              {busy ? <><span className="spinner" />执行中…</> : '执行工具'}
            </button>
            {out && <span className="pill ok">耗时 {out.elapsed_ms}ms</span>}
            {err && <span className="pill err">失败</span>}
          </div>
          {err && <p className="hint" style={{ color: 'var(--err)', marginTop: 8 }}>{err}</p>}
          {out && <pre className="out" style={{ marginTop: 10 }}>{JSON.stringify(out.result ?? out, null, 2)}</pre>}
        </div>
      </div>

      <div className="card">
        <h2>自动生成的系统提示词</h2>
        <p className="hint" style={{ margin: 0 }}>
          以下内容由 <code>registry.describe()</code> 动态生成 —— 新增工具后无需手动维护，注册即出现。
        </p>
        <pre className="prompt" style={{ marginTop: 10 }}>{promptPreview || '加载中…'}</pre>
      </div>
    </>
  )
}
