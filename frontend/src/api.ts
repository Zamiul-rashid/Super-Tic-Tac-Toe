export type GameState = {
  cells: number[]
  boards: number[]
  turn: number
  forced: number
  result: number | null
  legal: number[]
}

export type Tier = {
  key: string
  label: string
  simulations: number
  leaf_batch: number
  stroke: number
}

export type Health = {
  status: string
  model: { iteration?: number; arch?: string; opset?: number }
  threads: number
  sessions: { active: number; max: number }
  tiers: Tier[]
  default_difficulty: string
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    // The server explains refusals in `detail`; surfacing its words beats
    // inventing our own, especially for the session cap.
    let detail = `${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body; the status will have to do */
    }
    throw new Error(detail)
  }
  return response.status === 204 ? (undefined as T) : response.json()
}

export const getHealth = () => request<Health>('/api/health')

export const newGame = (difficulty: string, humanSide: number) =>
  request<{ session: string; state: GameState; engine_action: number | null }>('/api/game', {
    method: 'POST',
    body: JSON.stringify({ difficulty, human_side: humanSide }),
  })

export const playMove = (session: string, action: number) =>
  request<{ state: GameState; engine_action: number | null }>(
    `/api/game/${session}/move`,
    { method: 'POST', body: JSON.stringify({ action }) },
  )

export const endGame = (session: string) =>
  request<void>(`/api/game/${session}`, { method: 'DELETE' })
