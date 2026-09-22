/** 通用格式化与配色助手（与后端 constants.py / tool/graph_snapshot.py 保持一致）。 */

export const NEBULA_PALETTE = [
  '#38bdf8', '#c084fc', '#f43f5e', '#fbbf24',
  '#34d399', '#60a5fa', '#f472b6', '#a3e635',
]

export const DEFAULT_DOMAIN = '未分类'

const CRC_TABLE = (() => {
  const table = new Int32Array(256)
  for (let n = 0; n < 256; n += 1) {
    let c = n
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
    table[n] = c
  }
  return table
})()

/** 与后端 zlib.crc32 逐位一致的 CRC-32，保证领域颜色跨端稳定。 */
export function crc32(bytes) {
  let crc = -1
  for (let i = 0; i < bytes.length; i += 1) {
    crc = (crc >>> 8) ^ CRC_TABLE[(crc ^ bytes[i]) & 0xff]
  }
  return (crc ^ -1) >>> 0
}

export function domainColor(name) {
  const text = (name || DEFAULT_DOMAIN) || DEFAULT_DOMAIN
  const bytes = new TextEncoder().encode(text)
  return NEBULA_PALETTE[crc32(bytes) % NEBULA_PALETTE.length]
}

/** 节点类型（kind）的固定色：领域=恒星、实体=行星、事实/文档/备注=卫星。 */
export function kindColor(kind) {
  switch (kind) {
    case 'domain': return '#fbbf24'
    case 'entity': return '#38bdf8'
    case 'fact': return '#f472b6'
    case 'chunk': return '#a3e635'
    case 'note': return '#c084fc'
    case 'event': return '#f43f5e'
    default: return '#94a3b8'
  }
}

export const KIND_LABEL = {
  domain: '领域',
  entity: '实体',
  fact: '事实',
  chunk: '知识块',
  note: '备注',
  event: '事件',
}

export function kindLabel(kind) {
  return KIND_LABEL[kind] || kind || '节点'
}

export function fmtScore(v) {
  if (v === null || v === undefined || v === '') return '-'
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  return n.toFixed(3)
}

export function fmtBytes(n) {
  if (!n && n !== 0) return '-'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export function clip(text, max = 60) {
  const s = String(text || '')
  return s.length > max ? `${s.slice(0, max)}…` : s
}

export function errMessage(e) {
  return e && e.message ? e.message : String(e)
}