// main.js — Dance Pulse 前端
// 双轨: /resolve 解析直链 -> <video> 直接播放 (播放轨);
//       WS /ws/youtube 发 start -> 后端 asyncio FFmpeg 拉流推理 (推理轨, 1Hz)。
// 延迟口径 (严格):
//   TTFF (首帧) = 点击播放 -> 首个渲染帧 (已预缓冲时走 requestVideoFrameCallback,
//                 否则 loadeddata; performance.now 打点), 目标 <500ms
//   E2E (端到端) = 服务端抽到该帧 (frame_ts, 画面事件时刻) -> 前端渲染完预测, 目标 <1000ms
//   TTFP (首条预测) = start -> 第一条非 warmup pred, 目标 <=4s (含解析+起流+暖机)
import Hls from 'hls.js'

const $ = (id) => document.getElementById(id)
const playerA = $('video')
const startBtn = $('startBtn')
const stopBtn = $('stopBtn')
const urlInput = $('urlInput')
const statusEl = $('status')

const wsProto = location.protocol === 'https:' ? 'wss://' : 'ws://'
const WS_URL = wsProto + location.host + '/ws/youtube'

let ws = null
let tStart = 0        // start 点击时刻 (performance.now)
let ttffDone = false  // TTFF 只记一次 (loadeddata / rVFC 双路径)
let ttfpDone = false  // TTFP 只记第一条
let hls = null        // hls.js 实例 (HLS 兜底)
let lastPred = null   // 最近一条预测 (供人工反馈标注)
let playing = false   // 会话进行中: 预缓冲不得抢占正在播放的 <video>
let lastStart = null  // 最近一次 WS start 的载荷 (进度条拖动后带 start_at 重启推理轨)

// ---------------------------------------------------------------------------
// 概率条 (6 类: 5 舞种 + no_dance, Unknown 合并行按需渲染)
// ---------------------------------------------------------------------------
const barEls = {}

// no_dance -> "非舞蹈" 显示映射
function zhName(n) {
  return (n === 'no_dance' || n === 'Unknown / No Dance') ? '非舞蹈' : n
}

function ensureBars(names) {
  const box = $('bars')
  for (const n of names) {
    if (barEls[n]) continue
    const row = document.createElement('div')
    row.className = 'bar-row'
    row.innerHTML =
      `<div class="bar-label">${zhName(n)}</div>` +
      `<div class="bar-track"><div class="bar-fill"></div></div>` +
      `<div class="bar-val">0.0%</div>`
    box.appendChild(row)
    barEls[n] = { fill: row.querySelector('.bar-fill'), val: row.querySelector('.bar-val') }
  }
}

function renderProbs(probs) {
  ensureBars(Object.keys(probs))
  for (const [n, p] of Object.entries(probs)) {
    const el = barEls[n]
    el.fill.style.width = (p * 100).toFixed(1) + '%'
    el.val.textContent = (p * 100).toFixed(1) + '%'
  }
}

// ---------------------------------------------------------------------------
// 延迟指标
// ---------------------------------------------------------------------------
function setMetric(boxId, elId, ms, limit) {
  const box = $(boxId)
  $(elId).textContent = ms.toFixed(0) + ' ms'
  box.classList.remove('bad', 'ok')
  box.classList.add(ms < limit ? 'ok' : 'bad')
}

function setStatus(text, ok) {
  statusEl.textContent = text
  statusEl.classList.toggle('ok', !!ok)
}

// ---------------------------------------------------------------------------
// 播放轨: 直链 -> <video>
// ---------------------------------------------------------------------------
function markFirstFrame(tNow) {
  // TTFF 打点只记一次 (rVFC / loadeddata / rAF 多路径互斥)。
  // tNow: rVFC 回调携带的帧呈现时刻 (DOMHighResTimeStamp), 比回调执行时再取
  // performance.now() 更精确; 事件/无参路径退回 performance.now()
  if (ttffDone) return
  ttffDone = true
  const tFrame = typeof tNow === 'number' ? tNow : performance.now()
  setMetric('ttffBox', 'ttff', tFrame - tStart, 500)
}

// 双播放器轮换: active = 当前播放, idle = 预缓冲下一个 URL 的隐藏播放器。
// 换 URL 时把已缓冲好的 idle 直接切前台 play(), 首帧只剩解码呈现 (~50ms);
// 预缓冲挂在 idle 上也不再偷换 active 的 src (换视频首帧不再等现拉流 2s+)
const playerB = document.createElement('video')
playerB.playsInline = true
playerB.muted = true
playerB.controls = true
playerB.style.display = 'none'
playerA.parentNode.insertBefore(playerB, playerA.nextSibling)
let active = playerA
let idle = playerB

function srcOf(el) { return el.getAttribute('src') || el._hlsUrl || '' }

function attachSource(el, playUrl) {
  if (el === active) destroyHls()
  // HLS (m3u8) 在 Chrome 需 hls.js; MP4 直链 video.src 直接播 (首帧最快路径)
  const isHls = playUrl.includes('m3u8')
  if (isHls && Hls.isSupported()) {
    const inst = new Hls({ maxBufferLength: 4, maxMaxBufferLength: 8 })
    inst.loadSource(playUrl)
    inst.attachMedia(el)
    inst.on(Hls.Events.ERROR, (_, data) => setStatus('播放错误: ' + data.details))
    if (el === active) hls = inst
    el._hlsUrl = playUrl
  } else {
    el._hlsUrl = ''
    el.src = playUrl
    el.preload = 'auto'   // 立即开始缓冲, 但不播放 (用户手势后才允许出声)
    el.load()
  }
}

function swapPlayers() {
  active.pause()
  active.style.display = 'none'
  idle.style.display = ''
  destroyHls()   // 旧 active 的 HLS 实例随切换释放
  const t = active; active = idle; idle = t
}

// 预缓冲: URL 一解析就挂到闲置播放器开始拉流/缓冲, 正在播的 active 完全不受影响
function prefetchMedia(playUrl) {
  if (srcOf(active) === playUrl || srcOf(idle) === playUrl) return
  if (playUrl.includes('m3u8')) return   // HLS (直播) 无需预缓冲
  attachSource(idle, playUrl)
}

function playDirect(playUrl) {
  if (srcOf(active) !== playUrl) {
    if (!playUrl.includes('m3u8') && srcOf(idle) === playUrl) {
      swapPlayers()                 // 已预缓冲: 直接切前台 (换 URL 首帧 <500ms 的关键路径)
    } else {
      attachSource(active, playUrl) // 兜底: 未命中预缓冲, 现拉
    }
  }
  // TTFF: 点击 -> 首个真实渲染帧。rVFC 只在帧真正呈现到合成器时回调, 且回调的
  // now 即帧呈现时刻 (loadeddata 在预缓冲期就触发过、也不代表可见帧, 精度口径不如 rVFC)
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    active.requestVideoFrameCallback((now) => markFirstFrame(now))
  } else if (active.readyState >= 2) {
    // 无 rVFC 的浏览器: 数据已缓冲时 loadeddata 早已触发过、不会再响,
    // 用双 rAF 对齐 play() 后首帧呈现 (第一个 rAF 进入帧周期, 第二个在该帧绘制后)
    requestAnimationFrame(() => requestAnimationFrame(() => markFirstFrame()))
  } else {
    active.addEventListener('loadeddata', markFirstFrame, { once: true })
  }
  active.play().catch(() => {})
}

function destroyHls() {
  if (hls) { hls.destroy(); hls = null }
}

// ---------------------------------------------------------------------------
// 推理轨: WS /ws/youtube
// ---------------------------------------------------------------------------
function connectWS() {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === WebSocket.OPEN) return resolve()
    ws = new WebSocket(WS_URL)
    ws.onopen = () => resolve()
    ws.onerror = () => reject(new Error('WebSocket 连接失败'))
    ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data))
    ws.onclose = () => {
      playing = false
      startBtn.disabled = false
      stopBtn.disabled = true
      setStatus('连接已断开')
    }
  })
}

function handleMessage(msg) {
  if (msg.type === 'started') {
    // 播放已在点击时启动 (双轨并行), 这里只更新状态/标题
    if (msg.title) setStatus('已连接: ' + msg.title, true)
    // 不禁用 start: 支持"不停止直接换视频"——再点一次即无缝切换 (服务端自动停旧起新)
    startBtn.disabled = false
    stopBtn.disabled = false
  } else if (msg.type === 'warmup') {
    setStatus(`推理预热中 ${msg.frames}/16 …`)
  } else if (msg.type === 'pred') {
    if (!ttfpDone) {
      ttfpDone = true
      setMetric('ttfpBox', 'ttfp', performance.now() - tStart, 4000)
    }
    const isNoDance = msg.pred === 'no_dance' || msg.pred === 'Unknown / No Dance'
    $('predName').textContent = isNoDance ? '非舞蹈' : msg.pred
    $('predConf').textContent = isNoDance
      ? `置信度 ${(msg.confidence * 100).toFixed(1)}% (判定为非舞蹈)`
      : `置信度 ${(msg.confidence * 100).toFixed(1)}%`
    renderProbs(msg.probs)
    // E2E: 画面事件时刻 (frame_ts, 服务端抽帧入队) -> 前端渲染完成
    setMetric('e2eBox', 'e2e', Date.now() - msg.frame_ts * 1000, 1000)
    // 人工反馈: 记住这条预测, 等用户标注
    lastPred = msg
    $('fbOkBtn').disabled = false
    $('fbBadBtn').disabled = false
    $('fbStatus').textContent = ''
  } else if (msg.type === 'error') {
    setStatus('错误: ' + msg.msg)
    // 会话已死 (拉流/推理异常或模型未就绪): 恢复按钮, 允许重试/换视频
    playing = false
    startBtn.disabled = false
    stopBtn.disabled = true
  }
}

// ---------------------------------------------------------------------------
// 人工反馈标注: POST /feedback -> logs/feedback.jsonl
// ---------------------------------------------------------------------------
async function sendFeedback(label) {
  if (!lastPred) return
  $('fbOkBtn').disabled = true
  $('fbBadBtn').disabled = true
  try {
    const r = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        label,
        pred: lastPred.pred,
        confidence: lastPred.confidence,
        probs: lastPred.probs,
        frame_ts: lastPred.frame_ts,
        video_url: urlInput.value.trim(),
        client_ts: Date.now() / 1000,
      }),
    })
    if (!r.ok) throw new Error('HTTP ' + r.status)
    $('fbStatus').textContent = label === 'correct' ? '已记录: 准确 ✓' : '已记录: 有误 (待人工复核)'
  } catch (e) {
    $('fbStatus').textContent = '记录失败: ' + e.message
    $('fbOkBtn').disabled = false
    $('fbBadBtn').disabled = false
  }
}

$('fbOkBtn').addEventListener('click', () => sendFeedback('correct'))
$('fbBadBtn').addEventListener('click', () => sendFeedback('wrong'))

// ---------------------------------------------------------------------------
// 控制
// ---------------------------------------------------------------------------
// 预解析: URL 一输入就后台解析, 使 yt-dlp 的 ~1.9s 不占用 TTFF / TTFP 关键路径
let resolveCache = null   // {url, play_url, title}
let resolvePending = null

function resolveVideo(url, verbose) {
  url = canonicalUrl(url)  // 归一化 (去掉 &list= 等参数): 同一视频不同写法共享缓存
  if (resolveCache && resolveCache.url === url) return Promise.resolve(resolveCache)
  if (resolvePending && resolvePending.url === url) return resolvePending.promise
  // sessionStorage 复用: 刷新页面后重播同一直链 (有效期 ~6h) 也秒开, TTFF 不再吃解析耗时
  const ss = ssGet(url)
  if (ss) {
    resolveCache = { url, ...ss }
    prefetchMedia(resolveCache.play_url)   // 预缓冲到闲置播放器, 播放中也安全
    return Promise.resolve(resolveCache)
  }
  // 超时自动重试 (共 3 次 x 30s): 慢解析的原任务在服务端继续跑完并进缓存,
  // 重试通常秒回缓存 —— 直接换 URL 即使 YouTube 限流也能换成功, 而不是报错放弃
  const p = (async () => {
    for (let attempt = 1; ; attempt++) {
      try {
        const r = await fetch('/resolve?url=' + encodeURIComponent(url), { signal: AbortSignal.timeout(30000) })
        if (!r.ok) {
          // 透出服务端真实原因 (yt-dlp 具体错误), 不再只显示 "HTTP 400"
          const d = await r.json().catch(() => null)
          throw new Error(d && d.detail ? String(d.detail).replace(/^解析失败: /, '') : 'HTTP ' + r.status)
        }
        const j = await r.json()
        resolveCache = { url, ...j }
        ssSet(url, j)
        // 预缓冲: 解析完即开始拉媒体, TTFF 只剩 play->出帧。
        // 挂到闲置播放器, 播放中换 URL 也能提前缓冲好 (换视频首帧 <500ms)
        prefetchMedia(resolveCache.play_url)
        return resolveCache
      } catch (e) {
        const retryable = e.name === 'TimeoutError' || e.name === 'TypeError'
        if (retryable && attempt < 3) {
          if (verbose) setStatus(`解析较慢, 自动重试中 (${attempt}/3)…`)
          continue
        }
        if (resolvePending && resolvePending.url === url) resolvePending = null  // 失败清 pending, 允许重试
        throw retryable
          ? new Error('解析超时(已自动重试 3 次)——YouTube 解析持续较慢, 请稍后再试')
          : e
      }
    }
  })()
  resolvePending = { url, promise: p }
  return p
}

// 归一化 YouTube URL -> 缓存键统一 (粘贴带 &list= 的链接与纯 watch?v= 共享预解析)
function canonicalUrl(url) {
  const m = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]{11})/)
  return m ? 'https://www.youtube.com/watch?v=' + m[1] : url
}

// resolve 结果 sessionStorage 缓存 (5.5h < 直链 6h 有效期): 页面刷新后重播也秒开
function ssGet(url) {
  try {
    const raw = sessionStorage.getItem('dp_res:' + url)
    if (!raw) return null
    const o = JSON.parse(raw)
    return Date.now() - o.ts < 5.5 * 3600e3 ? o.data : null
  } catch { return null }
}
function ssSet(url, data) {
  try { sessionStorage.setItem('dp_res:' + url, JSON.stringify({ ts: Date.now(), data })) } catch {}
}

// 预解析时机 (预热): 打字 300ms 防抖 / 粘贴立即 / 失焦立即 —— 把解析+媒体预缓冲
// 藏进用户操作间隙, 点"播放并推理"时命中缓存, TTFF 关键路径只剩 play()。
// 正则闸门只放行完整 YouTube URL, 避免逐键触发垃圾解析请求
const YT_RE = /^https?:\/\/(www\.)?(youtube\.com\/(watch\?v=|shorts\/|embed\/)|youtu\.be\/)[\w-]{11}/
let prefetchTimer = null
function warmupNow() {
  const url = urlInput.value.trim()
  if (YT_RE.test(url)) resolveVideo(url).catch(() => {})   // 静默预热, 失败不打扰用户
}
urlInput.addEventListener('input', () => {
  clearTimeout(prefetchTimer)
  prefetchTimer = setTimeout(warmupNow, 300)
})
urlInput.addEventListener('paste', () => setTimeout(warmupNow, 0))  // paste 时 value 尚未落位, 下一宏任务再读
urlInput.addEventListener('blur', warmupNow)   // 光标离开输入框立即预热
urlInput.addEventListener('change', warmupNow) // Enter 提交场景 (change 先于 blur 触发)

startBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim()
  if (!url) { setStatus('请先粘贴 YouTube URL'); return }
  tStart = performance.now()
  ttffDone = false
  ttfpDone = false
  startBtn.disabled = true
  stopBtn.disabled = false
  setStatus('解析视频中…')
  try {
    // 预解析命中则此处瞬间返回; 未命中则等待 (仍计入 TTFF, 但输入时的 change 事件通常已预热)
    const info = await resolveVideo(url, true)
    playing = true
    // 双轨并行: 播放轨立即起播 (TTFF 关键路径不再等 WS/推理轨)
    playDirect(info.play_url)
    // 推理轨: WS start 带已解析直链, 服务端免重复解析
    await connectWS()
    lastStart = { url, play_url: info.play_url, title: info.title }
    ws.send(JSON.stringify({
      type: 'start', ...lastStart,   // 跳过服务端重复解析
    }))
  } catch (e) {
    setStatus('解析失败: ' + e.message)
    playing = false
    startBtn.disabled = false
    stopBtn.disabled = true
  }
})

stopBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'stop' }))
  playing = false
  destroyHls()
  for (const el of [playerA, playerB]) {   // 双播放器一起清 (含闲置预缓冲)
    el.pause()
    el._hlsUrl = ''
    el.removeAttribute('src')
    el.load()
  }
  setStatus('已停止')
  startBtn.disabled = false
  stopBtn.disabled = true
})

// 进度条拖动 (seeked): 推理轨带 start_at 重启, 同步到新播放位置。
// 防抖 500ms —— 拖动过程会连续触发 seeked, 只在停稳后重启一次
let seekTimer = null
function onSeeked(e) {
  if (e.target !== active) return   // 只响应当前播放器的拖动
  clearTimeout(seekTimer)
  seekTimer = setTimeout(() => {
    if (playing && lastStart && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'start', ...lastStart, start_at: active.currentTime }))
    }
  }, 500)
}
playerA.addEventListener('seeked', onSeeked)
playerB.addEventListener('seeked', onSeeked)
