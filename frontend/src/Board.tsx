import type { GameState } from './api'

// Marks are drawn, not typeset: hand-weighted SVG strokes with deliberately
// uneven endpoints and terminals that do not quite meet. A typeface X sitting
// in a grid cell reads as a form field; this reads as a game played on paper.
function Cross() {
  return (
    <svg className="mark mark-x" viewBox="0 0 100 100" aria-hidden="true">
      <path d="M22 19 C 40 38, 62 60, 79 82" />
      <path d="M80 21 C 61 41, 39 61, 21 80" />
    </svg>
  )
}

function Ring() {
  return (
    <svg className="mark mark-o" viewBox="0 0 100 100" aria-hidden="true">
      <path d="M68 26 C 44 12, 18 30, 20 53 C 22 77, 50 90, 70 79 C 88 69, 89 40, 72 27" />
    </svg>
  )
}

// The red pencil circle. It is the only element allowed this colour, and its
// one drawing pass is the only non-user-triggered motion on the page.
function PencilCircle() {
  return (
    <svg className="pencil-circle" viewBox="0 0 120 120" aria-hidden="true">
      <path d="M60 7 C 92 8, 114 32, 113 61 C 112 92, 88 113, 58 113 C 28 113, 6 91, 7 60 C 8 30, 30 8, 61 7 C 70 7, 78 9, 85 12" />
    </svg>
  )
}

type Props = {
  state: GameState
  onPlay: (action: number) => void
  busy: boolean
  lastEngineAction: number | null
}

export function Board({ state, onPlay, busy, lastEngineAction }: Props) {
  const legal = new Set(state.legal)
  const playable = state.result === null && !busy

  return (
    <div className="board" role="grid" aria-label="Super Tic-Tac-Toe board">
      {Array.from({ length: 9 }, (_, b) => {
        const won = state.boards[b]
        const isForced = state.forced === b || (state.forced === -1 && won === 0)
        const classes = [
          'local',
          won !== 0 ? 'local-closed' : '',
          state.forced === b ? 'local-forced' : '',
          state.forced === -1 && won === 0 ? 'local-open' : '',
        ].filter(Boolean).join(' ')

        return (
          <div key={b} className={classes} role="group"
               aria-label={`board ${b + 1}${isForced ? ', playable' : ''}`}>
            {state.forced === b && <PencilCircle />}
            {Array.from({ length: 9 }, (_, c) => {
              const action = b * 9 + c
              const value = state.cells[action]
              const canPlay = playable && legal.has(action)
              return (
                <button
                  key={c}
                  className={`cell${canPlay ? ' cell-open' : ''}${
                    action === lastEngineAction ? ' cell-latest' : ''}`}
                  onClick={() => canPlay && onPlay(action)}
                  disabled={!canPlay}
                  aria-label={`board ${b + 1} cell ${c + 1}`}
                >
                  {value === 1 && <Cross />}
                  {value === -1 && <Ring />}
                </button>
              )
            })}
            {won === 1 && <div className="claim claim-x"><Cross /></div>}
            {won === -1 && <div className="claim claim-o"><Ring /></div>}
          </div>
        )
      })}
    </div>
  )
}
