/**
 * WebSocket client helpers for asrbench live streams.
 *
 * All endpoints follow the Phase-4 EventBus convention — the server
 * pushes JSON frames with a ``type`` discriminator. Connect-on-demand;
 * caller is responsible for closing the socket on component teardown.
 */

import type {
  ActivityRecord,
  RunProgressEvent,
  VRAMSnapshot,
} from './types'

const WS_BASE = `ws://${location.host}`

function _connect<T>(path: string, onMessage: (msg: T) => void): WebSocket {
  const ws = new WebSocket(`${WS_BASE}${path}`)
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data) as T)
    } catch {
      /* malformed frame — ignore */
    }
  }
  return ws
}

export function connectActivity(
  onMessage: (entry: ActivityRecord) => void,
): WebSocket {
  return _connect('/ws/activity', onMessage)
}

/** @deprecated — prefer {@link connectActivity}. Kept for pre-0.2 UI code. */
export function connectLogs(
  onMessage: (entry: ActivityRecord) => void,
): WebSocket {
  return _connect('/ws/logs', onMessage)
}

export function connectVRAM(
  onMessage: (snap: VRAMSnapshot) => void,
): WebSocket {
  return _connect('/ws/vram', onMessage)
}

export function connectRun(
  runId: string,
  onMessage: (msg: RunProgressEvent) => void,
): WebSocket {
  return _connect(`/ws/runs/${runId}`, onMessage)
}

export function connectOptimize(
  studyId: string,
  onMessage: (msg: Record<string, unknown>) => void,
): WebSocket {
  return _connect(`/ws/optimize/${studyId}`, onMessage)
}

export function connectDataset(
  datasetId: string,
  onMessage: (msg: Record<string, unknown>) => void,
): WebSocket {
  return _connect(`/ws/datasets/${datasetId}`, onMessage)
}
