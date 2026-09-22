import { useEffect, useState } from 'react'
import { Board } from './Board'
import { endGame, getGame, getHealth, newGame, playMove } from './api'
import type { GameState, Health } from './api'

const BOARD_NAMES = [
  'top-left', 'top-centre', 'top-right',
  'middle-left', 'centre', 'middle-right',
  'bottom-left', 'bottom-centre', 'bottom-right',
]

const SESSION_KEY = 'sttt.session'

// Browser storage can be absent or throw (private windows, cleared data).
function remember(session: string | null) {
  try {
    if (session) sessionStorage.setItem(SESSION_KEY, session)
    else sessionStorage.removeItem(SESSION_KEY)
  } catch { /* fine: the game just will not survive a refresh */ }
}
function remembered(): string | null {
  const fromUrl = new URLSearchParams(location.search).get('session')
  if (fromUrl) return fromUrl
  try { return sessionStorage.getItem(SESSION_KEY) } catch { return null }
}

function moveLabel(action: number) {
  return `${BOARD_NAMES[Math.floor(action / 9)]} ${(action % 9) + 1}`
}

type Status = { text: string; kind: 'play' | 'over' | 'error' }

function status(state: GameState | null, busy: boolean, humanSide: number, error: string | null): Status {
  if (error) return { text: error, kind: 'error' }
  if (!state) return { text: 'Setting up the board.', kind: 'play' }
  if (state.result !== null) {
    if (state.result === 0) return { text: 'A draw. Nobody took three boards.', kind: 'over' }
    return state.result === humanSide
      ? { text: 'You won.', kind: 'over' }
      : { text: 'The network won.', kind: 'over' }
  }
  if (busy) return { text: 'The network is thinking.', kind: 'play' }
  if (state.forced === -1) return { text: 'Your move, any open board.', kind: 'play' }
  return { text: `Your move, ${BOARD_NAMES[state.forced]} board.`, kind: 'play' }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [session, setSession] = useState<string | null>(null)
  const [state, setState] = useState<GameState | null>(null)
  const [difficulty, setDifficulty] = useState('standard')
  const [humanSide, setHumanSide] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [log, setLog] = useState<{ action: number; side: number }[]>([])
  const [lastEngine, setLastEngine] = useState<number | null>(null)

  async function start(nextDifficulty: string, side: number, previous: string | null) {
    setError(null)
    setBusy(true)
    try {
      if (previous) await endGame(previous).catch(() => undefined)
      const game = await newGame(nextDifficulty, side)
      remember(game.session)
      setSession(game.session)
      setState(game.state)
      setLastEngine(game.engine_action)
      setLog(game.engine_action === null ? [] : [{ action: game.engine_action, side: -side }])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  // A board on arrival, not a start button: the page is the game. A refresh
  // picks the same game back up; an expired one quietly starts fresh.
  useEffect(() => {
    getHealth()
      .then(async (h) => {
        setHealth(h)
        setDifficulty(h.default_difficulty)
        const saved = remembered()
        if (saved) {
          try {
            const game = await getGame(saved)
            const entries = game.history.map((action, i) => ({ action, side: i % 2 === 0 ? 1 : -1 }))
            const last = entries[entries.length - 1]
            setSession(saved)
            setState(game.state)
            setDifficulty(game.difficulty)
            setHumanSide(game.human_side)
            setLog(entries)
            setLastEngine(last && last.side !== game.human_side ? last.action : null)
            return
          } catch {
            remember(null)
          }
        }
        await start(h.default_difficulty, 1, null)
      })
      .catch((e) => setError(e.message))
  }, [])

  async function play(action: number) {
    if (!session || !state) return
    setBusy(true)
    setError(null)
    // Your mark goes down at once. The network's reply is held back for a
    // beat even when the search answers in a tenth of a second: a reply that
    // lands in the same frame as your own move cannot be seen as a reply.
    const cells = state.cells.slice()
    cells[action] = humanSide
    setState({ ...state, cells, forced: -1, legal: [] })
    setLastEngine(null)
    const entries = [{ action, side: humanSide }]
    try {
      const [result] = await Promise.all([
        playMove(session, action),
        new Promise((resolve) => setTimeout(resolve, 450)),
      ])
      if (result.engine_action !== null) entries.push({ action: result.engine_action, side: -humanSide })
      setState(result.state)
      setLastEngine(result.engine_action)
      setLog((previous) => [...previous, ...entries])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const line = status(state, busy, humanSide, error)
  const iteration = health?.model?.iteration
  const tierLabel = health?.tiers.find((t) => t.key === difficulty)?.label ?? difficulty

  return (
    <main className="sheet">
      <aside className="margin">
        <h1 className="wordmark">Super Tic-Tac-Toe</h1>
        {log.length === 0 ? (
          <p className="log-empty">Moves are written here.</p>
        ) : (
          <ol className="log" aria-label="move log">
            {log.map((entry, index) => (
              <li key={index} className={entry.side === humanSide ? 'log-you' : 'log-net'}>
                <span className="log-n">{index + 1}</span>
                {moveLabel(entry.action)}
              </li>
            ))}
          </ol>
        )}
      </aside>

      <div className="play">
        <h1 className="title-mobile">Super Tic-Tac-Toe</h1>

        {state ? (
          <Board state={state} humanSide={humanSide} onPlay={play} busy={busy}
                 lastEngineAction={lastEngine} moveCount={log.length} />
        ) : (
          <div className="board" aria-hidden="true">
            {Array.from({ length: 9 }, (_, b) => (
              <div key={b} className="local">
                {Array.from({ length: 9 }, (_, c) => <span key={c} className="cell" />)}
              </div>
            ))}
          </div>
        )}

        <p className={`status status-${line.kind}`} role="status">{line.text}</p>

        <div className="controls">
          <fieldset className="tiers">
            <legend>Opponent</legend>
            {health?.tiers.map((tier) => (
              <button
                key={tier.key}
                className={`tier${tier.key === difficulty ? ' tier-on' : ''}`}
                aria-pressed={tier.key === difficulty}
                onClick={() => { setDifficulty(tier.key); start(tier.key, humanSide, session) }}
                disabled={busy}
              >
                <svg viewBox="0 0 64 12" aria-hidden="true">
                  <path d="M3 8 C 18 4, 40 9, 61 5" style={{ strokeWidth: tier.stroke }} />
                </svg>
                {tier.label}
              </button>
            ))}
          </fieldset>

          <fieldset className="seats">
            <legend>You play</legend>
            <button className={`seat${humanSide === 1 ? ' seat-on' : ''}`} aria-pressed={humanSide === 1}
                    onClick={() => { setHumanSide(1); start(difficulty, 1, session) }} disabled={busy}>
              first
            </button>
            <button className={`seat${humanSide === -1 ? ' seat-on' : ''}`} aria-pressed={humanSide === -1}
                    onClick={() => { setHumanSide(-1); start(difficulty, -1, session) }} disabled={busy}>
              second
            </button>
          </fieldset>

          <button className="restart" onClick={() => start(difficulty, humanSide, session)} disabled={busy}>
            New game
          </button>
        </div>

        <p className="rules">
          Win a small board to claim its square; take three squares in a line to win.
          The cell you play in sends the network to that board for its next move,
          and its move sends you to yours.
        </p>
        <p className="colophon">
          Playing {tierLabel} against {iteration ? `iteration ${iteration}` : 'the network'}.
        </p>
      </div>
    </main>
  )
}
