"""验证滚轮缩放不再带动页面滚动。

原理：在真实浏览器环境里没法跑 headless（没装 playwright），
所以改为静态验证 + 逻辑验证：
  1. JSX 上确实没有 onWheel / onTouchMove（React 合成事件是 passive 的，拦不住）
  2. 原生 addEventListener 确实带了 { passive: false }
  3. zoomAt 里确实调了 preventDefault 且参数是鼠标坐标换算
  4. canvas CSS 里有 touch-action: none
  5. 容器有 overscroll-behavior: contain

运行：python _verify_wheel_fix.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASS, FAIL = [], []


def ck(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   -> {detail}" if (detail and not cond) else ""))


jsx = (ROOT / "frontend" / "src" / "components" / "ForceGraph.jsx").read_text(encoding="utf-8")
css = (ROOT / "frontend" / "src" / "styles.css").read_text(encoding="utf-8")

print("=== 1. JSX 上不能再有 onWheel / onTouchMove（合成事件是 passive，preventDefault 无效）===")
ck("JSX 无 onWheel", not re.search(r"\bonWheel\s*=", jsx), "仍然挂了 React onWheel")
ck("JSX 无 onTouchMove", not re.search(r"\bonTouchMove\s*=", jsx), "仍然挂了 React onTouchMove")

print("\n=== 2. 原生监听必须带 passive:false ===")
m = re.search(r"addEventListener\(\s*['\"]wheel['\"]\s*,\s*(\w+)\s*,\s*\{\s*passive:\s*false\s*\}\s*\)", jsx)
ck("wheel 以 passive:false 注册", bool(m), "没找到 passive:false 的 wheel 监听")
tm = re.search(r"addEventListener\(\s*['\"]touchmove['\"]\s*,\s*\w+\s*,\s*\{\s*passive:\s*false\s*\}\s*\)", jsx)
ck("touchmove 以 passive:false 注册", bool(tm), "没找到 passive:false 的 touchmove 监听")

print("\n=== 3. 清理函数要成对移除 ===")
rm = re.findall(r"removeEventListener\(\s*['\"](\w+)['\"]", jsx)
ck("wheel/touchmove 都有移除", "wheel" in rm and "touchmove" in rm, str(rm))

print("\n=== 4. zoomAt 内必须 preventDefault ===")
zm = re.search(r"const zoomAt\s*=\s*useCallback\(\(e\)\s*=>\s*\{(.*?)\n  \}, \[\]\)", jsx, re.S)
ck("找到 zoomAt 定义", bool(zm))
if zm:
    body = zm.group(1)
    ck("zoomAt 调用了 preventDefault", "preventDefault()" in body)
    ck("zoomAt 读鼠标坐标（getBoundingClientRect）", "getBoundingClientRect" in body)
    ck("zoomAt 有缩放范围钳制", "Math.min(" in body and "Math.max(" in body)
    ck("zoomAt 处理了 deltaMode（非像素单位）", "deltaMode" in body)

print("\n=== 5. CSS 兜底 ===")
ck("canvas 有 touch-action: none", re.search(r"\.graph-wrap canvas\s*\{[^}]*touch-action:\s*none", css))
ck("容器有 overscroll-behavior", re.search(r"\.graph-wrap\s*\{[^}]*overscroll-behavior:\s*contain", css))

print("\n=== 6. 滚轮只影响缩放，不改节点坐标（避免误拖）===")
if zm:
    body = zm.group(1)
    ck("zoomAt 只改 transform，不直接改 node.x/y",
       "st.transform.k =" in body and not re.search(r"node\.x\s*=", body))

print(f"\n=== 结果: {len(PASS)} PASS / {len(FAIL)} FAIL ===")
for f in FAIL:
    print("  -", f)
sys.exit(0 if not FAIL else 1)
