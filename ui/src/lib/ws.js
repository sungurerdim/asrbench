/** WebSocket client helpers for asrbench live streams. */

const WS_BASE = `ws://${location.host}`

/**
 * Subscribe to activity logs.
 * @param {(entry: object) => void} onMessage
 * @returns {WebSocket}
 */
export function connectLogs(onMessage) {
  const ws = new WebSocket(`${WS_BASE}/ws/logs`)
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch {}
  }
  return ws
}

/**
 * Subscribe to VRAM updates (polled every 500ms server-side).
 * @param {(snap: object) => void} onMessage
 * @returns {WebSocket}
 */
export function connectVRAM(onMessage) {
  const ws = new WebSocket(`${WS_BASE}/ws/vram`)
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch {}
  }
  return ws
}

/**
 * Subscribe to live run progress.
 * @param {string} runId
 * @param {(msg: object) => void} onMessage
 * @returns {WebSocket}
 */
export function connectRunLive(runId, onMessage) {
  const ws = new WebSocket(`${WS_BASE}/ws/runs/${runId}/live`)
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)) } catch {}
  }
  return ws
}
