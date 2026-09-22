import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { kindColor, kindLabel, domainColor, clip } from '../util.js'

/** 力导向星云图：领域=恒星、实体=行星、事实/知识块=卫星。
 *
 * 纯 Canvas 2D 手写布局（不引三方库）：斥力 + 弹簧 + 向心力，
 * 位置按节点 id 跨次刷新保留（delta 刷新不重排），支持滚轮缩放、
 * 拖拽平移、拖拽单个节点、hover 提示与点击选中。
 */
export default function NebulaGraph({
  nodes = [],
  links = [],
  height = 620,
  showLabels = true,
  highlight = null,
  selectedId = '',
  onSelect,
}) {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const simRef = useRef({ items: new Map(), edges: [], alpha: 1, raf: 0 })
  const viewRef = useRef({ x: 0, y: 0, k: 1 })
  const dragRef = useRef(null)
  const [tip, setTip] = useState(null)
  const [hoverId, setHoverId] = useState('')

  const idOf = useCallback((v) => (typeof v === 'object' && v !== null ? v.id : v), [])

  // ---------------- 布局初始化 / 增量更新 ----------------
  useEffect(() => {
    const sim = simRef.current
    const next = new Map()
    for (const n of nodes) {
      const prev = sim.items.get(n.id)
      next.set(n.id, prev ? { ...prev, ...n } : {
        ...n,
        x: (Math.random() - 0.5) * 420,
        y: (Math.random() - 0.5) * 420,
        vx: 0,
        vy: 0,
        r: radiusOf(n),
        pinned: false,
      })
    }
    sim.items = next
    sim.edges = links
      .map((e) => ({ ...e, s: idOf(e.source), t: idOf(e.target) }))
      .filter((e) => next.has(e.s) && next.has(e.t))
    sim.alpha = Math.max(sim.alpha, 0.55)
  }, [nodes, links, idOf])

  // ---------------- 物理步进 ----------------
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')
    const sim = simRef.current

    const step = () => {
      const items = [...sim.items.values()]
      const byId = sim.items
      sim.alpha *= 0.995
      if (sim.alpha < 0.02) sim.alpha = 0.02

      // 斥力（O(n²) 但节点量级在千级以内可接受；大图靠 LOD 少画不少算）
      for (let i = 0; i < items.length; i += 1) {
        const a = items[i]
        for (let j = i + 1; j < items.length; j += 1) {
          const b = items[j]
          let dx = a.x - b.x
          let dy = a.y - b.y
          let d2 = dx * dx + dy * dy
          if (d2 < 1) { dx = (Math.random() - 0.5) * 2; dy = (Math.random() - 0.5) * 2; d2 = 1 }
          const min = a.r + b.r + 26
          const f = Math.min(2600 / d2, 3.2)
          if (d2 < min * min) {
            const d = Math.sqrt(d2) || 1
            const push = ((min - d) / d) * 0.35
            dx *= push; dy *= push
            if (!a.pinned) { a.vx += dx; a.vy += dy }
            if (!b.pinned) { b.vx -= dx; b.vy -= dy }
          }
          const d = Math.sqrt(d2) || 1
          if (!a.pinned) { a.vx += (dx / d) * f; a.vy += (dy / d) * f }
          if (!b.pinned) { b.vx -= (dx / d) * f; b.vy -= (dy / d) * f }
        }
      }

      // 弹簧 + 层级引力（恒星把它的行星拉住）
      for (const e of sim.edges) {
        const a = byId.get(e.s)
        const b = byId.get(e.t)
        if (!a || !b) continue
        const dx = b.x - a.x
        const dy = b.y - a.y
        const d = Math.hypot(dx, dy) || 1
        const target = a.r + b.r + (a.kind === 'domain' || b.kind === 'domain' ? 48 : 92)
        const f = ((d - target) / d) * 0.045
        if (!a.pinned) { a.vx += dx * f; a.vy += dy * f }
        if (!b.pinned) { b.vx -= dx * f; b.vy -= dy * f }
      }

      for (const n of items) {
        if (n.pinned) { n.vx = 0; n.vy = 0; continue }
        if (n.parent && byId.has(n.parent)) {
          const p = byId.get(n.parent)
          n.vx += (p.x - n.x) * 0.0022
          n.vy += (p.y - n.y) * 0.0022
        }
        n.vx += (0 - n.x) * 0.0016
        n.vy += (0 - n.y) * 0.0016
        n.vx *= 0.86
        n.vy *= 0.86
        n.x += n.vx * sim.alpha
        n.y += n.vy * sim.alpha
      }

      draw(ctx, canvas, sim, viewRef.current, showLabels, highlight, selectedId, hoverId)
      sim.raf = requestAnimationFrame(step)
    }
    sim.raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(sim.raf)
  }, [showLabels, highlight, selectedId, hoverId])

  // ---------------- 尺寸 ----------------
  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return undefined
    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const rect = wrap.getBoundingClientRect()
      canvas.width = Math.max(1, Math.round(rect.width * dpr))
      canvas.height = Math.max(1, Math.round(rect.height * dpr))
      const ctx = canvas.getContext('2d')
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      canvas.__cssWidth = rect.width
      canvas.__cssHeight = rect.height
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [])

  // ---------------- 交互 ----------------
  const toWorld = useCallback((evt) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const v = viewRef.current
    return {
      x: (evt.clientX - rect.left - rect.width / 2 - v.x) / v.k,
      y: (evt.clientY - rect.top - rect.height / 2 - v.y) / v.k,
    }
  }, [])

  const pick = useCallback((evt) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const v = viewRef.current
    const mx = evt.clientX - rect.left - rect.width / 2 - v.x
    const my = evt.clientY - rect.top - rect.height / 2 - v.y
    let best = null
    for (const n of simRef.current.items.values()) {
      const dx = n.x * v.k - mx
      const dy = n.y * v.k - my
      const r = Math.max(n.r * v.k, 9)
      if (dx * dx + dy * dy <= r * r) {
        if (!best || n.r > best.r) best = n
      }
    }
    return best
  }, [])

  const onPointerDown = (evt) => {
    const hit = pick(evt)
    if (hit) {
      dragRef.current = { kind: 'node', id: hit.id, startX: evt.clientX, startY: evt.clientY }
      hit.pinned = true
    } else {
      dragRef.current = { kind: 'pan', x: evt.clientX, y: evt.clientY }
    }
    evt.currentTarget.setPointerCapture(evt.pointerId)
  }

  const onPointerMove = (evt) => {
    const d = dragRef.current
    if (!d) {
      const hit = pick(evt)
      setHoverId(hit ? hit.id : '')
      if (hit) {
        const rect = wrapRef.current.getBoundingClientRect()
        setTip({ x: evt.clientX - rect.left + 14, y: evt.clientY - rect.top + 14, node: hit })
      } else {
        setTip(null)
      }
      return
    }
    if (d.kind === 'pan') {
      const v = viewRef.current
      v.x += evt.clientX - d.x
      v.y += evt.clientY - d.y
      d.x = evt.clientX
      d.y = evt.clientY
      return
    }
    const n = simRef.current.items.get(d.id)
    if (n) {
      const p = toWorld(evt)
      n.x = p.x
      n.y = p.y
      n.vx = 0
      n.vy = 0
    }
  }

  const endDrag = (evt) => {
    const d = dragRef.current
    dragRef.current = null
    if (d && d.kind === 'node') {
      const start = { x: d.startX, y: d.startY }
      const moved = Math.hypot(evt.clientX - start.x, evt.clientY - start.y) > 4
      if (!moved && onSelect) onSelect(simRef.current.items.get(d.id))
    }
    if (evt.currentTarget.releasePointerCapture && evt.pointerId !== undefined) {
      try { evt.currentTarget.releasePointerCapture(evt.pointerId) } catch { /* 已释放 */ }
    }
  }

  const onWheel = (evt) => {
    evt.preventDefault()
    const v = viewRef.current
    const factor = evt.deltaY < 0 ? 1.12 : 1 / 1.12
    v.k = Math.min(4, Math.max(0.15, v.k * factor))
  }

  const zoomBy = (factor) => {
    const v = viewRef.current
    v.k = Math.min(4, Math.max(0.15, v.k * factor))
  }

  const legend = useMemo(() => {
    const seen = new Map()
    for (const n of nodes) {
      const key = n.kind === 'domain' ? 'domain' : (n.kind || 'other')
      if (!seen.has(key)) seen.set(key, kindColor(key))
    }
    return [...seen.entries()]
  }, [nodes])

  return (
    <div className="graph-wrap" ref={wrapRef} style={{ height }}>
      <canvas
        ref={canvasRef}
        aria-label="知识星云图"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onWheel={onWheel}
      />
      <div className="graph-legend">
        <div className="row" style={{ gap: 10 }}>
          <span>缩放 {Math.round(viewRef.current.k * 100)}%</span>
          <button className="btn sm ghost" onClick={() => zoomBy(1.25)}>＋</button>
          <button className="btn sm ghost" onClick={() => zoomBy(0.8)}>－</button>
        </div>
        {legend.map(([kind, color]) => (
          <div className="li" key={kind}>
            <span className="dot" style={{ background: color }} />
            <span>{kindLabel(kind)}</span>
          </div>
        ))}
        <div className="li" style={{ marginTop: 2, opacity: 0.8 }}>恒星=领域 行星=实体 卫星=事实/知识块</div>
      </div>
      {tip && (
        <div className="graph-tip" style={{ left: tip.x, top: tip.y }}>
          <div className="t">{tip.node.title}</div>
          <div>{kindLabel(tip.node.kind)} · {tip.node.domain}</div>
          {tip.node.date && <div>日期 {tip.node.date}</div>}
          {tip.node.meta && tip.node.meta.aliases && tip.node.meta.aliases.length > 0 && (
            <div>别名 {tip.node.meta.aliases.join('、')}</div>
          )}
          {tip.node.content && <div style={{ opacity: 0.8 }}>{clip(tip.node.content, 80)}</div>}
        </div>
      )}
    </div>
  )
}

function radiusOf(n) {
  const base = { domain: 15, entity: 8.5, fact: 5.5, chunk: 4.5, note: 4, event: 6 }[n.kind] || 5
  const importance = typeof n.importance === 'number' ? n.importance : 0.5
  return base + importance * 5
}

function draw(ctx, canvas, sim, view, showLabels, highlight, selectedId, hoverId) {
  const w = canvas.__cssWidth || canvas.width
  const h = canvas.__cssHeight || canvas.height
  ctx.clearRect(0, 0, w, h)
  ctx.save()
  ctx.translate(w / 2 + view.x, h / 2 + view.y)
  ctx.scale(view.k, view.k)

  const hlNodes = highlight && highlight.nodes
  const hlEdges = highlight && highlight.edges
  const nodes = [...sim.items.values()]
  const count = nodes.length
  // LOD：节点多时只画连接的边 + 近处标签，避免糊成一片
  const lod = count > 900 ? 0 : count > 420 ? 1 : 2

  ctx.lineWidth = 1 / view.k
  for (const e of sim.edges) {
    const a = sim.items.get(e.s)
    const b = sim.items.get(e.t)
    if (!a || !b) continue
    const key = `${e.s}->${e.t}`
    const on = hlEdges ? hlEdges.has(key) : null
    if (lod < 2 && !on) continue
    const faded = Boolean(hlEdges) && !on
    ctx.strokeStyle = faded ? 'rgba(120,150,210,0.05)' : (on ? 'rgba(94,234,212,0.85)' : 'rgba(130,160,220,0.22)')
    ctx.lineWidth = (on ? 1.8 : 1) / view.k
    ctx.beginPath()
    ctx.moveTo(a.x, a.y)
    ctx.lineTo(b.x, b.y)
    ctx.stroke()
    if (showLabels && view.k > 1.15 && lod >= 1 && (on || !faded)) {
      const mx = (a.x + b.x) / 2
      const my = (a.y + b.y) / 2
      ctx.fillStyle = faded ? 'rgba(147,163,191,0.25)' : 'rgba(147,163,191,0.9)'
      ctx.font = `${10 / view.k}px system-ui`
      ctx.textAlign = 'center'
      ctx.fillText(clip(e.relation || '', 12), mx, my)
    }
  }

  for (const n of nodes) {
    const on = hlNodes ? hlNodes.has(n.id) : null
    const faded = Boolean(hlNodes) && !on
    const selected = selectedId && selectedId === n.id
    const hovered = hoverId === n.id
    const color = n.kind === 'domain' ? (n.color || domainColor(n.title)) : kindColor(n.kind)
    const r = n.r * (selected || hovered ? 1.35 : 1)
    ctx.globalAlpha = faded ? 0.18 : 1
    if (n.kind === 'domain' || selected || on) {
      const glow = ctx.createRadialGradient(n.x, n.y, r * 0.4, n.x, n.y, r * 3.2)
      glow.addColorStop(0, hexAlpha(color, 0.32))
      glow.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = glow
      ctx.beginPath()
      ctx.arc(n.x, n.y, r * 3.2, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2)
    ctx.fill()
    if (selected || hovered) {
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2 / view.k
      ctx.stroke()
    }
    const wantLabel = showLabels && (lod >= 2 || n.kind === 'domain' || n.kind === 'entity') && !faded
    if (wantLabel) {
      ctx.fillStyle = faded ? 'rgba(147,163,191,0.35)' : 'rgba(230,236,247,0.92)'
      const size = (n.kind === 'domain' ? 12 : 10.5) / view.k
      ctx.font = `${n.kind === 'domain' ? 650 : 500} ${size}px system-ui`
      ctx.textAlign = 'center'
      ctx.fillText(clip(n.title, n.kind === 'domain' ? 14 : 12), n.x, n.y + r + size)
    }
    ctx.globalAlpha = 1
  }
  ctx.restore()
}

function hexAlpha(hex, alpha) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '')
  if (!m) return `rgba(148,163,184,${alpha})`
  const v = parseInt(m[1], 16)
  return `rgba(${(v >> 16) & 255},${(v >> 8) & 255},${v & 255},${alpha})`
}