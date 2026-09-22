import { useEffect, useState } from 'react'
import { Board } from './Board'
import { endGame, getHealth, newGame, playMove } from './api'
import type { GameState, Health } from './api'

const BOARD_NAMES = [
  'top-left', 'top-centre', 'top-right',
  'middle-left', 'centre', 'middle-right',
  'bottom-left', 'bottom-centre', 'bottom-right',
]

function moveLabel(action: number) {
  return `${BOARD_NAMES[Math.floor(action / 9)]} ${(action % 9) + 1}`
}

function statusLine(state: GameState | null, busy: boolean, humanSide: number) {
  if (!state) return 'Setting up.'
  if (state.result !== null) {
    if (state.result === 0) return 'Drawn. Nobody took three boards.'
    return state.result === humanSide ? 'You won.' : 'The network won.'
  }
  if (busy) return 'The network is thinking.'
  if (state.forced === -1) return 'Your move, any open board.'
  return `Your move, ${BOARD_NAMES[state.forced]} board.`
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [session, setSession] = useState<string | null>(null)
  const [state, setState] = useState<GameState | null>(null)
  const [difficulty, setDifficulty] = useState<string>('standard')
  const [humanSide, setHumanSide] = useState(1)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [log, setLog] = useState<{ action: number; side: number }[]>([])
  const [lastEngine, setLastEngine] = useState<number | null>(null)

  useEffect(() => {
    getHealth()
      .then((h) => { setHealth(h); setDifficulty(h.default_difficulty) })
      .catch((e) => setError(e.message))
  }, [])

  async function start(nextDifficulty = difficulty, side = humanSide) {
    setError(null)
    setBusy(true)
    try {
      if (session) await endGame(session).catch(() => undefined)
      const game = await newGame(nextDifficulty, side)
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

  async function play(action: number) {
    if (!session) return
    setBusy(true)
    setError(null)
    const entries = [{ action, side: humanSide }]
    try {
      const result = await playMove(session, action)
      if (result.engine_action !== null) {
        entries.push({ action: result.engine_action, side: -humanSide })
      }
      setState(result.state)
      setLastEngine(result.engine_action)
      setLog((previous) => [...previous, ...entries])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const iteration = health?.model?.iteration
  const tierLabel = health?.tiers.find((t) => t.key === difficulty)?.label ?? difficulty

  return (
    <main className="sheet">
      <aside className="margin">
        <ol className="log" aria-label="move log">
          {log.map((entry, index) => (
            <li key={index} className={entry.side === humanSide ? 'log-you' : 'log-net'}>
              <span className="log-n">{index + 1}</span>
              {moveLabel(entry.action)}
            </li>
          ))}
        </ol>
      </aside>

      <div className="play">
        {state ? (
          <Board state={state} onPlay={play} busy={busy} lastEngineAction={lastEngine} />
        ) : (
          <div className="board board-empty" aria-hidden="true" />
        )}

        <p className="status">{error ?? statusLine(state, busy, humanSide)}</p>

        <h1 className="title">Super Tic-Tac-Toe</h1>
        <p className="blurb">
          Win a small board to claim its square. Take three squares in a line to win.
          Wherever you play inside a board sends the network to that board next.
        </p>

        <div className="controls">
          <fieldset className="tiers">
            <legend>Opponent</legend>
            {health?.tiers.map((tier) => (
              <button
                key={tier.key}
                className={`tier${tier.key === difficulty ? ' tier-on' : ''}`}
                onClick={() => { setDifficulty(tier.key); start(tier.key, humanSide) }}
                disabled={busy}
              >
                {/* Stroke weight carries the strength; the word carries the meaning. */}
                <svg viewBox="0 0 64 12" aria-hidden="true">
                  <path d="M4 8 C 18 4, 40 9, 60 5" style={{ strokeWidth: tier.stroke }} />
                </svg>
                {tier.label}
              </button>
            ))}
          </fieldset>

          <fieldset className="seats">
            <legend>You play</legend>
            <button className={`seat${humanSide === 1 ? ' seat-on' : ''}`}
                    onClick={() => { setHumanSide(1); start(difficulty, 1) }} disabled={busy}>
              first
            </button>
            <button className={`seat${humanSide === -1 ? ' seat-on' : ''}`}
                    onClick={() => { setHumanSide(-1); start(difficulty, -1) }} disabled={busy}>
              second
            </button>
          </fieldset>

          <button className="restart" onClick={() => start()} disabled={busy}>
            {state ? 'New game' : 'Start'}
          </button>
        </div>

        <p className="colophon">
          Playing {tierLabel} against {iteration ? `iteration ${iteration}` : 'the network'}.
        </p>
      </div>
    </main>
  )
}
