import type { GameState } from './api'

// Marks are drawn, not typeset: hand-weighted strokes with uneven endpoints
// and terminals that do not quite meet. A typeface X in a grid cell reads as
// a form field; this reads as a game played by hand.
function Cross({ className }: { className: string }) {
  return (
    <svg className={className} viewBox="0 0 100 100" aria-hidden="true">
      <path pathLength={100} d="M22 19 C 40 38, 62 60, 79 82" />
      <path pathLength={100} d="M80 21 C 61 41, 39 61, 21 80" />
    </svg>
  )
}

function Ring({ className }: { className: string }) {
  return (
    <svg className={className} viewBox="0 0 100 100" aria-hidden="true">
      <path pathLength={100} d="M68 26 C 44 12, 18 30, 20 53 C 22 77, 50 90, 70 79 C 88 69, 89 40, 72 27" />
    </svg>
  )
}

function Mark({ value, className }: { value: number; className: string }) {
  return value === 1 ? <Cross className={className} /> : <Ring className={className} />
}

// The red pencil loop: a loose squarish circle that follows the board it is
// drawn around, with the overshoot where a pencil stroke closes on itself.
// pathLength normalises it so the draw animation is one dash of length 100.
function PencilLoop() {
  return (
    <svg className="pencil-loop" viewBox="0 0 120 120" aria-hidden="true">
      <path
        pathLength={100}
        d="M 24 12 C 48 7, 78 6, 104 11 C 112 14, 115 22, 114 34 C 116 60, 115 84, 110 106 C 106 113, 96 114, 84 113 C 60 116, 36 115, 18 111 C 9 108, 6 98, 7 86 C 4 62, 5 38, 9 20 C 11 12, 18 9, 30 9 C 42 8, 56 8, 66 9"
      />
    </svg>
  )
}

type Props = {
  state: GameState
  humanSide: number
  onPlay: (action: number) => void
  busy: boolean
  lastEngineAction: number | null
  moveCount: number
}

export function Board({ state, humanSide, onPlay, busy, lastEngineAction, moveCount }: Props) {
  const legal = new Set(state.legal)
  const playable = state.result === null && !busy
  const sideClass = (value: number) => (value === humanSide ? 'mark-you' : 'mark-net')

  return (
    <div className="board" role="grid" aria-label="Super Tic-Tac-Toe board">
      {Array.from({ length: 9 }, (_, b) => {
        const won = state.boards[b]
        const forced = state.forced === b
        return (
          <div
            key={b}
            className={`local${won !== 0 ? ' local-closed' : ''}`}
            role="group"
            aria-label={`board ${b + 1}${forced ? ', you must play here' : ''}`}
          >
            {/* Cells first: the # lines are drawn with :nth-child, which counts
                every element child, so nothing may precede them. */}
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
                  aria-label={`board ${b + 1}, cell ${c + 1}${value ? ', taken' : ''}`}
                >
                  {value !== 0 && <Mark value={value} className={`mark ${sideClass(value)}`} />}
                  {value === 0 && canPlay && (
                    <Mark value={humanSide} className="mark ghost mark-you" />
                  )}
                </button>
              )
            })}
            {/* Keyed on the move count so the loop redraws every turn, even
                when the network sends you back to the same board. */}
            {forced && <PencilLoop key={moveCount} />}
            {(won === 1 || won === -1) && (
              <div className="claim"><Mark value={won} className={`mark ${sideClass(won)}`} /></div>
            )}
          </div>
        )
      })}
    </div>
  )
}
