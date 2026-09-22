import React, { useState } from 'react'
import { api, ingestSentenceStream } from '../api.js'

const SAMPLES = [
  '2024年5月，张医生在北京协和医院为李主任完成了首例机器人辅助心脏手术。',
  '胰岛素注射液由诺和诺德生产，用于治疗糖尿病引起的血糖升高。',
  '昨天，王伟在杭州把项目交付给了阿里巴巴的测试团队。',
]

export default function IngestPanel({ onChanged }) {
  const [text, setText] = useState('')
  const [stream, setStream] = useState('')
  const [events, setEvents] = useState([])
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [mode, setMode] = useState('stream')   // stream | json

  const push = (ev) => setEvents((e) => [...e.slice(-120), ev])

  const doIngestStream = async (t) => {
    setBusy(true); setStream(''); setEvents([]); setResult(null)
    const stop = ingestSentenceStream(
      t,
      'sentence',
      (stage, data) => {
        push({ stage, data, ok: true })
        if (stage === 'llm_delta') setStream((s) => s + (data.delta || ''))
      },
      (done) => {
        setBusy(false)
        if (done && done.ok) {
          push({ stage: 'done', data, ok: true })
          setResult(done.result || done)
        } else {
          push({ stage: 'done', data, ok: false })
        }
        onChanged && onChanged()
      }
    )
    return stop
  }

  const doIngestJson = async (t) => {
    setBusy(true); setStream(''); setEvents([]); setResult(null)
    push({ stage: 'request', ok: true, data: { text: t.slice(0, 120) } })
    try {
      const r = await api.ingestSentence(t)
      setResult(r)
      push({ stage: 'done', ok: true, data: r })
    } catch (e) {
      setResult({ ok: false, error: e.message })
      push({ stage: 'error', ok: false, data: { error: e.message } })
    }
    setBusy(false)
    onChanged && onChanged()
  }

  const submit = () => {
    const t = text.trim()
    if (!t || busy) return
    if (mode === 'stream') doIngestStream(t)
    else doIngestJson(t)
  }

  const short = (stage) => {
    const m = {
      document: 'sqlite 写入原始文档', queued: 'chunk 进入 ingest_queue', existing_entities: '读取 neo4j 已有实体',
      llm_start: 'LLM 开始流式抽取', llm_delta: 'LLM 增量输出', llm_end: 'LLM 抽取完成',
      parsed: '解析器解析完成', time_resolved: '时间标记处理', qdrant_ok: 'qdrant 向量入库成功',
      qdrant_fail: 'qdrant 入库失败', neo4j_ok: 'neo4j 图谱入库成功', neo4j_fail: 'neo4j 入库失败',
      graph_written: '实体与关系已写入 neo4j', promoted: 'chunk 转正进 chunks 表',
      stayed_in_queue: '仍在队列中等待重试', warn: '警告', done: '完成',
    }
    return m[stage] || stage
  }

  return (
    <>
      <div className="card">
        <h2>一句话入库</h2>
        <p className="hint">
          输入一句话 → LLM 先读取 neo4j 已有实体（<code>get_all_entities</code>）→ 按提示词抽取实体与关系 →
          <code>parse_llm_output</code> 解析 → 没有时间则调用 <code>get_current_time</code> 兜底 →
          qdrant 与 neo4j 两条线并行入库 → 都成功才转正进 <code>chunks</code> 表。
        </p>
        <textarea
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.ctrlKey && e.key === 'Enter') submit() }}
          placeholder="输入要入库的那句话……（Ctrl+Enter 提交）"
        />
        <div className="row" style={{ marginTop: 10 }}>
          <button className="btn primary" onClick={submit} disabled={!text.trim() || busy}>
            {busy ? <><span className="spinner" />入库中…</> : '开始入库'}
          </button>
          <select style={{ width: 170 }} value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="stream">SSE 流式（看 LLM 增量）</option>
            <option value="json">一次性返回</option>
          </select>
          {SAMPLES.map((s) => (
            <button key={s} className="btn sm ghost" onClick={() => setText(s)} disabled={busy}>{s.slice(0, 12)}…</button>
          ))}
          <button className="btn sm ghost" onClick={() => { setText(''); setStream(''); setEvents([]); setResult(null) }} disabled={busy}>清空</button>
        </div>
      </div>

      {(stream || events.length > 0) && (
        <div className="card">
          <h2>LLM 流式输出 · 实时抽取过程</h2>
          <div className="stream-box">{stream || <span style={{ opacity: 0.5 }}>等待 LLM 输出……</span>}</div>
          <div className="events">
            {events.map((e, i) => (
              <div className={`ev ${e.ok ? '' : 'bad'}`} key={i}>
                <span className="st">{short(e.stage)}</span>
                <span style={{ opacity: 0.9 }}>
                  {e.stage === 'llm_delta' ? `(+${(e.data.delta || '').length} 字符)` :
                    (e.stage === 'documents' ? e.data.document_id :
                    (e.stage === 'existing_entities' ? `共 ${e.data.count} 个已有实体` :
                    (e.stage === 'parsed' ? `实体 ${e.data.entities} / 关系 ${e.data.relations}（复用 ${e.data.reused_entities}）` :
                    (e.stage === 'time_resolved' ? `${e.data.source === 'llm' ? '句子含时间' : '句子无时间，取系统时间'} → ${e.data.time}` :
                    (e.stage === 'qdrant_ok' ? `维度 ${e.data.dim}` :
                    (e.stage === 'neo4j_ok' ? `实体 ${e.data.entities} / 关系 ${e.data.relations}` :
                    (e.stage === 'done' ? (e.data.promoted !== false ? 'chunk 已转正进 chunks 表' : (e.data.error || '未转正')) :
                    (e.data && e.data.message ? e.data.message : ''))))))))}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {result && (
        <div className="grid2">
          <div className="card">
            <h2>抽取结果</h2>
            <p className="hint" style={{ margin: 0 }}>
              {result.qdrant === 'success' ? <span className="pill ok">qdrant ✓</span> : <span className="pill err">qdrant ✗</span>}
              {' '}
              {result.neo4j === 'success' ? <span className="pill ok">neo4j ✓</span> : <span className="pill err">neo4j ✗</span>}
              {' '}
              {result.promoted ? <span className="pill ok">已转正进 chunks 表</span> : <span className="pill warn">仍在队列中</span>}
              {' '}<span className="pill">时间来源：{result.time_source === 'llm' ? '句子本身' : '系统时间兜底'}</span>
              {result.time && <span className="pill accent">{result.time}</span>}
            </p>
            <div className="stat-grid" style={{ marginTop: 12 }}>
              <div className="stat"><div className="k">新增实体</div><div className="v a">{result.extracted?.new_entities ?? 0}</div></div>
              <div className="stat"><div className="k">复用实体</div><div className="v b">{result.extracted?.reused_entities ?? 0}</div></div>
              <div className="stat"><div className="k">关系三元组</div><div className="v o">{result.extracted?.relations?.length ?? 0}</div></div>
              <div className="stat"><div className="k">总耗时</div><div className="v">{result.elapsed_ms ?? '-'}ms</div></div>
            </div>
          </div>
          <div className="card">
            <h2>解析出的实体与关系</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginTop: 10 }}>
              {(result.extracted?.relations || []).map((r, i) => (
                <div className="hit" key={i} style={{ marginBottom: 0 }}>
                  <div className="top">
                    <span>{r.time || '无时间'}</span>
                    <span className={`pill ${r.directed ? 'accent' : ''}`}>{r.directed ? '有向 →' : '无向 —'}</span>
                  </div>
                  <div className="body" style={{ fontSize: 12.5 }}>
                    <b style={{ color: '#c7d2fe' }}>{r.source}</b>
                    <span style={{ color: 'var(--accent)', margin: '0 6px' }}>—{r.predicate}→</span>
                    <b style={{ color: '#c7d2fe' }}>{r.target}</b>
                  </div>
                </div>
              ))}
              {(result.extracted?.relations || []).length === 0 && <div className="empty">未抽出关系</div>}
            </div>
            <div style={{ marginTop: 10, display: 'flex', gap: 5, flexWrap: 'wrap' }}>
              {(result.extracted?.entities || []).map((e, i) => (
                <span className={`pill ${e.reused ? 'accent' : ''}`} key={i} title={e.reused ? '复用的已有实体' : '新建实体'}>
                  {e.reused ? '↺ ' : '＋ '}{e.name} · {e.type}{e.time ? ' · ' + e.time : ''}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {result && result.llm_raw && (
        <div className="card">
          <h2>LLM 原始输出（交给 parse_llm_output 的输入）</h2>
          <pre className="out" style={{ marginTop: 10 }}>{result.llm_raw}</pre>
        </div>
      )}
    </>
  )
}
