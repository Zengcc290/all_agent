import React, { useEffect, useRef, useState, useCallback } from 'react'

const TYPE_COLORS = {
  人物: '#f472b6', 人员: '#f472b6', 公司: '#60a5fa', 组织: '#a78bfa', 机构: '#a78bfa',
  地点: '#34d399', 产品: '#fbbf24', 时间: '#22d3ee', 事件: '#fb7185', 疾病: '#f87171',
  药物: '#4ade80', 法律: '#c084fc', 金融: '#facc15', 科技: '#38bdf8', 概念: '#94a3b8',
}
const PALETTE = ['#f472b6', '#60a5fa', '#a78bfa', '#34d399', '#fbbf24', '#22d3ee',
                 '#fb7185', '#f87171', '#4ade80', '#c084fc', '#facc15', '#38bdf8']

export function colorOfType(t) {
  if (!t) return '#94a3b8'
  if (TYPE_COLORS[t]) return TYPE_COLORS[t]
  let h = 0
  for (let i = 0; i < t.length; i++) h = (h * 31 + t.charCodeAt(i)) >>> 0
  return PALETTE[h % PALETTE.length]
}

/**
 * 实体星球：自研 canvas 力导向图。
 * - 有向关系画箭头（→），无向关系画直线（—）
 * - 关系线上标注 predicate 与 time
 * - 支持拖拽节点、滚轮缩放、空白处拖拽平移
 */
export default function ForceGraph({ nodes = [], links = [], height = 620, showLabels = true }) {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  const stateRef = useRef({ nodes: [], links: [], transform: { x: 0, y: 0, k: 1 }, alpha: 1 })
  const dragRef = useRef(null)
  const panRef = useRef(null)
  const [tip, setTip] = useState(null)
  const [hover, setHover] = useState(null)
  const [dim, setDim] = useState({ w: 900, h: height })

  /* ---------------- 尺寸自适应 ---------------- */
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setDim({ w: Math.max(320, r.width), h: Math.max(320, r.height) })
    })
    ro.observe(el)
    const r = el.getBoundingClientRect()
    setDim({ w: Math.max(320, r.width), h: Math.max(320, r.height) })
    return () => ro.disconnect()
  }, [])

  /* ---------------- 构建/增量更新模拟数据 ---------------- */
  useEffect(() => {
    const st = stateRef.current
    const prev = new Map(st.nodes.map((n) => [n.id, n]))

    // 度 -> 半径
    const deg = new Map()
    links.forEach((l) => {
      deg.set(l.source, (deg.get(l.source) || 0) + 1)
      deg.set(l.target, (deg.get(l.target) || 0) + 1)
    })

    const nextNodes = nodes.map((n) => {
      const old = prev.get(n.id)
      const d = deg.get(n.id) || 0
      const r = 6 + Math.min(16, Math.sqrt(d) * 3.1)
      if (old) return { ...old, ...n, r, deg: d }
      const a = Math.random() * Math.PI * 2
      const rad = 40 + Math.random() * Math.min(dim.w, dim.h) * 0.3
      return {
        ...n, r, deg: d,
        x: dim.w / 2 + Math.cos(a) * rad,
        y: dim.h / 2 + Math.sin(a) * rad,
        vx: 0, vy: 0,
      }
    })
    const ids = new Set(nextNodes.map((n) => n.id))
    const nextLinks = links
      .filter((l) => ids.has(l.source) && ids.has(l.target))
      .map((l) => ({ ...l, sid: l.source, tid: l.target }))

    st.nodes = nextNodes
    st.links = nextLinks
    st.alpha = Math.max(st.alpha, 0.55)
  }, [nodes, links, dim.w, dim.h])

  /* ---------------- 力导向模拟 + 渲染 ---------------- */
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let raf = 0
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    canvas.width = dim.w * dpr
    canvas.height = dim.h * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const draw = () => {
      const st = stateRef.current
      const ns = st.nodes, ls = st.links
      const cx = st.transform.x + st.transform.k * (dim.w / 2)
      const cy = st.transform.y + st.transform.k * (dim.h / 2)

      // --- 物理：斥力 / 弹簧 / 向心力 ---
      if (st.alpha > 0.01) {
        st.alpha *= 0.973
        const cx0 = dim.w / 2, cy0 = dim.h / 2
        for (let i = 0; i < ns.length; i++) {
          const a = ns[i]
          if (dragRef.current && dragRef.current.node === a) continue
          a.vx = (a.vx || 0) * 0.82; a.vy = (a.vy || 0) * 0.82
          for (let j = i + 1; j < ns.length; j++) {
            const b = ns[j]
            let dx = a.x - b.x, dy = a.y - b.y
            let d2 = dx * dx + dy * dy
            if (d2 < 1) { dx = (Math.random() - 0.5) * 2; dy = (Math.random() - 0.5) * 2; d2 = dx * dx + dy * dy }
            if (d2 > 600 * 600) continue
            const f = (4200 / d2) * st.alpha
            const d = Math.sqrt(d2)
            const fx = (dx / d) * f, fy = (dy / d) * f
            a.vx += fx; a.vy += fy
            b.vx -= fx; b.vy -= fy
          }
          // 向心力（星球引力）
          a.vx += (cx0 - a.x) * 0.0016 * st.alpha * 3
          a.vy += (cy0 - a.y) * 0.0016 * st.alpha * 3
        }
        const byId = new Map(ns.map((n) => [n.id, n]))
        for (const l of ls) {
          const a = byId.get(l.sid), b = byId.get(l.tid)
          if (!a || !b) continue
          let dx = b.x - a.x, dy = b.y - a.y
          const d = Math.max(1, Math.hypot(dx, dy))
          const target = 74 + (l.weight || 0) * 7
          const f = (d - target) * 0.019 * st.alpha
          const fx = (dx / d) * f, fy = (dy / d) * f
          if (dragRef.current?.node !== a) { a.vx += fx; a.vy += fy }
          if (dragRef.current?.node !== b) { b.vx -= fx; b.vy -= fy }
        }
        for (const a of ns) {
          if (dragRef.current && dragRef.current.node === a) continue
          a.x += Math.max(-22, Math.min(22, a.vx))
          a.y += Math.max(-22, Math.min(22, a.vy))
        }
      }

      // --- 绘制 ---
      ctx.clearRect(0, 0, dim.w, dim.h)

      // 背景星尘
      ctx.fillStyle = 'rgba(255,255,255,0.55)'
      for (let i = 0; i < 46; i++) {
        const sx = ((i * 9301 + 49297) % 233280) / 233280 * dim.w
        const sy = ((i * 4523 + 128) % 233280) / 233280 * dim.h
        const tw = 0.4 + 0.6 * Math.abs(Math.sin(Date.now() / 900 + i))
        ctx.globalAlpha = 0.12 + tw * 0.16
        ctx.fillRect(sx, sy, 1.4, 1.4)
      }
      ctx.globalAlpha = 1

      const toScreen = (p) => ({
        x: p.x * st.transform.k + cx, y: p.y * st.transform.k + cy,
      })

      // 关系线
      const byId = new Map(ns.map((n) => [n.id, n]))
      for (const l of ls) {
        const A = byId.get(l.sid), B = byId.get(l.tid)
        if (!A || !B) continue
        const a = toScreen(A), b = toScreen(B)
        const col = hover === l.id ? 'rgba(94,234,212,0.95)' : 'rgba(130,160,220,0.34)'
        ctx.strokeStyle = col
        ctx.lineWidth = (hover === l.id ? 2.1 : 1.15) * st.transform.k
        ctx.setLineDash(l.directed ? [] : [5, 4])
        ctx.beginPath()
        ctx.moveTo(a.x, a.y)
        ctx.lineTo(b.x, b.y)
        ctx.stroke()
        ctx.setLineDash([])

        if (l.directed) {
          const ang = Math.atan2(b.y - a.y, b.x - a.x)
          const rA = (A.r || 7) * st.transform.k + 2
          const tipX = b.x - Math.cos(ang) * (B.r || 7) * st.transform.k - 1
          const tipY = b.y - Math.sin(ang) * (B.r || 7) * st.transform.k - 1
          const bx = a.x + Math.cos(ang) * rA, by = a.y + Math.sin(ang) * rA
          ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tipX, tipY); ctx.stroke()
          const s = 7 * st.transform.k
          ctx.fillStyle = col
          ctx.beginPath()
          ctx.moveTo(tipX, tipY)
          ctx.lineTo(tipX - Math.cos(ang - 0.42) * s, tipY - Math.sin(ang - 0.42) * s)
          ctx.lineTo(tipX - Math.cos(ang + 0.42) * s, tipY - Math.sin(ang + 0.42) * s)
          ctx.closePath(); ctx.fill()
        }

        // 关系标注
        if (showLabels && st.transform.k > 0.62 && l.predicate) {
          const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2
          const txt = l.time ? `${l.predicate}·${l.time}` : l.predicate
          ctx.font = `${Math.max(8, 9.5 * st.transform.k)}px "PingFang SC", "Microsoft YaHei", sans-serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          const w = ctx.measureText(txt).width + 7
          ctx.fillStyle = 'rgba(6,11,22,0.82)'
          ctx.beginPath()
          ctx.roundRect(mx - w / 2, my - 7.5, w, 15, 7)
          ctx.fill()
          ctx.fillStyle = hover === l.id ? '#5eead4' : 'rgba(190,210,240,0.82)'
          ctx.fillText(txt, mx, my)
        }
      }

      // 节点
      for (const n of ns) {
        const p = toScreen(n)
        const r = Math.max(2.4, (n.r || 7) * st.transform.k)
        const c = colorOfType(n.type)
        const isHover = hover === n.id

        // 光晕
        const g = ctx.createRadialGradient(p.x, p.y, r * 0.25, p.x, p.y, r * 2.7)
        g.addColorStop(0, c + 'cc')
        g.addColorStop(0.45, c + '33')
        g.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.globalAlpha = isHover ? 0.95 : 0.6
        ctx.fillStyle = g
        ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.7, 0, Math.PI * 2); ctx.fill()
        ctx.globalAlpha = 1

        // 星球本体
        const bg = ctx.createRadialGradient(p.x - r * 0.32, p.y - r * 0.36, r * 0.15, p.x, p.y, r)
        bg.addColorStop(0, '#ffffff')
        bg.addColorStop(0.32, c)
        bg.addColorStop(1, shade(c, -0.5))
        ctx.fillStyle = bg
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill()
        ctx.strokeStyle = isHover ? '#fff' : 'rgba(255,255,255,0.28)'
        ctx.lineWidth = isHover ? 2 : 0.9
        ctx.stroke()

        if (showLabels && st.transform.k > 0.45) {
          ctx.font = `600 ${Math.max(10, 12 * st.transform.k)}px "PingFang SC", "Microsoft YaHei", sans-serif`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          ctx.fillStyle = 'rgba(6,11,22,0.8)'
          const w = ctx.measureText(n.name).width
          ctx.beginPath()
          ctx.roundRect(p.x - w / 2 - 4, p.y + r + 3.5, w + 8, 15, 7)
          ctx.fill()
          ctx.fillStyle = isHover ? '#ffffff' : 'rgba(226,236,250,0.92)'
          ctx.fillText(n.name, p.x, p.y + r + 6)
        }
      }

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [dim.w, dim.h, hover, showLabels])

  /* ---------------- 交互：拖拽 / 平移 / 缩放 ---------------- */
  const pick = useCallback((mx, my) => {
    const st = stateRef.current
    const cx = st.transform.x + st.transform.k * (dim.w / 2)
    const cy = st.transform.y + st.transform.k * (dim.h / 2)
    let best = null, bd = Infinity
    for (const n of st.nodes) {
      const x = n.x * st.transform.k + cx, y = n.y * st.transform.k + cy
      const d = Math.hypot(mx - x, my - y)
      const r = Math.max(7, n.r * st.transform.k) + 6
      if (d < r && d < bd) { bd = d; best = n }
    }
    return best
  }, [dim.w, dim.h])

  const pos = (e) => {
    const r = canvasRef.current.getBoundingClientRect()
    const t = e.touches ? e.touches[0] : e
    return { x: t.clientX - r.left, y: t.clientY - r.top }
  }

  const onDown = (e) => {
    const { x, y } = pos(e)
    const n = pick(x, y)
    if (n) {
      dragRef.current = { node: n, ox: x, oy: y }
      stateRef.current.alpha = Math.max(stateRef.current.alpha, 0.25)
      setTip(null)
    } else {
      panRef.current = { x, y, tx: stateRef.current.transform.x, ty: stateRef.current.transform.y }
      setTip(null)
    }
  }

  const onMove = (e) => {
    const { x, y } = pos(e)
    const st = stateRef.current
    if (dragRef.current) {
      const { node, ox, oy } = dragRef.current
      node.x = (x - st.transform.x - st.transform.k * (dim.w / 2)) / st.transform.k + (x - ox) * 0
      // 简化：直接把屏幕位移换算到模型坐标
      node.x = (x - st.transform.x - st.transform.k * (dim.w / 2)) / st.transform.k
      node.y = (y - st.transform.y - st.transform.k * (dim.h / 2)) / st.transform.k
      node.vx = 0; node.vy = 0
      st.alpha = Math.max(st.alpha, 0.2)
      e.preventDefault()
      return
    }
    if (panRef.current) {
      st.transform.x = panRef.current.tx + (x - panRef.current.x)
      st.transform.y = panRef.current.ty + (y - panRef.current.y)
      e.preventDefault()
      return
    }
    // hover 检测
    const n = pick(x, y)
    setHover(n ? n.id : null)
    if (n) {
      const cx = st.transform.x + st.transform.k * (dim.w / 2)
      const cy = st.transform.y + st.transform.k * (dim.h / 2)
      setTip({
        x: n.x * st.transform.k + cx + (n.r || 7) + 12,
        y: n.y * st.transform.k + cy - 10,
        node: n,
      })
    } else {
      setTip(null)
    }
  }

  const onUp = () => { dragRef.current = null; panRef.current = null }

  /**
   * 缩放 + 平移。
   * 注意：这里刻意「不」用 JSX 上的 onWheel / onTouchMove。
   * React 17+ 把 wheel / touchmove 挂到 root 时是 passive 监听器，
   * 在合成事件里调 e.preventDefault() 会被浏览器静默忽略，
   * 于是滚轮缩放的同时整页跟着滚 —— 实体星球被滑出视口。
   * 所以改用原生 addEventListener + { passive: false }，才能真正阻止默认行为。
   */
  const zoomAt = useCallback((e) => {
    e.preventDefault()
    const st = stateRef.current
    const r = canvasRef.current?.getBoundingClientRect()
    if (!r) return
    const x = e.clientX - r.left, y = e.clientY - r.top
    const k0 = st.transform.k
    // deltaMode: 0=像素, 1=行, 2=页；非像素时归一化成像素，避免一次跳太大
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1
    const d = e.deltaY * unit
    const k = Math.min(4.5, Math.max(0.16, k0 * (d < 0 ? 1.13 : 0.885)))
    st.transform.k = k
    st.transform.x = x - (x - st.transform.x) * (k / k0)
    st.transform.y = y - (y - st.transform.y) * (k / k0)
  }, [])

  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    // passive: false 才能 preventDefault 掉页面滚动
    c.addEventListener('wheel', zoomAt, { passive: false })

    // 触摸拖拽/平移时同样要阻止页面滚动（React 的 onTouchMove 也是 passive，拦不住）
    const blockScroll = (e) => {
      // 只在多指或正在拖拽/平移时拦截，单指浏览手势不拦
      if (dragRef.current || panRef.current || e.touches.length > 1) {
        e.preventDefault()
        onMove(e)
      }
    }
    c.addEventListener('touchmove', blockScroll, { passive: false })
    return () => {
      c.removeEventListener('wheel', zoomAt)
      c.removeEventListener('touchmove', blockScroll)
    }
  }, [zoomAt])

  const reset = () => {
    stateRef.current.transform = { x: 0, y: 0, k: 1 }
    stateRef.current.alpha = 0.7
  }

  const legend = {}
  nodes.forEach((n) => { legend[n.type] = (legend[n.type] || 0) + 1 })
  const legendKeys = Object.keys(legend).sort((a, b) => legend[b] - legend[a]).slice(0, 9)

  return (
    <div className="graph-wrap" ref={wrapRef} style={{ height }}>
      <canvas
        ref={canvasRef}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => { onUp(); setHover(null); setTip(null) }}
        onTouchStart={onDown}
        onTouchEnd={onUp}
      />
      {tip && tip.node && (
        <div className="graph-tip" style={{ left: Math.min(tip.x, dim.w - 250), top: Math.max(6, tip.y) }}>
          <div className="t">{tip.node.name}</div>
          <div>类型：<b style={{ color: colorOfType(tip.node.type) }}>{tip.node.type || '未知'}</b></div>
          {tip.node.time && <div>时间：{tip.node.time}</div>}
          <div>关联关系：{tip.node.deg || 0} 条</div>
          <div style={{ opacity: 0.6, marginTop: 3, fontSize: 10.5 }}>{tip.node.id}</div>
        </div>
      )}
      {legendKeys.length > 0 && (
        <div className="graph-legend">
          {legendKeys.map((k) => (
            <div className="li" key={k}>
              <span className="dot" style={{ background: colorOfType(k) }} />
              <span>{k}</span>
              <span style={{ marginLeft: 'auto', opacity: 0.7 }}>{legend[k]}</span>
            </div>
          ))}
          <div className="li" style={{ marginTop: 4, borderTop: '1px solid rgba(120,150,210,.2)', paddingTop: 5 }}>
            <span style={{ opacity: 0.85 }}>实线→ 有向 · 虚线— 无向</span>
          </div>
        </div>
      )}
      <div style={{ position: 'absolute', right: 12, top: 12, display: 'flex', gap: 7 }}>
        <button className="btn sm" onClick={reset}>重置视图</button>
      </div>
    </div>
  )
}

function shade(hex, amt) {
  const h = hex.replace('#', '')
  const n = h.length === 3 ? h.split('').map(c => c + c).join('') : h
  let r = parseInt(n.slice(0, 2), 16), g = parseInt(n.slice(2, 4), 16), b = parseInt(n.slice(4, 6), 16)
  r = Math.max(0, Math.min(255, Math.round(r + 255 * amt)))
  g = Math.max(0, Math.min(255, Math.round(g + 255 * amt)))
  b = Math.max(0, Math.min(255, Math.round(b + 255 * amt)))
  return `#${[r, g, b].map(v => v.toString(16).padStart(2, '0')).join('')}`
}
