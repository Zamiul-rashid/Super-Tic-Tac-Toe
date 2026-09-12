#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <array>

namespace sttt {

// 8 winning lines in 3x3 Tic-Tac-Toe
// Row 0: 0,1,2 -> 0x007
// Row 1: 3,4,5 -> 0x038
// Row 2: 6,7,8 -> 0x1C0
// Col 0: 0,3,6 -> 0x049
// Col 1: 1,4,7 -> 0x092
// Col 2: 2,5,8 -> 0x124
// Diag 0: 0,4,8 -> 0x111
// Diag 1: 2,4,6 -> 0x054
constexpr uint16_t WIN_MASKS[8] = {
    0x007, 0x038, 0x1C0,
    0x049, 0x092, 0x124,
    0x111, 0x054
};

// Compile-time check for winning pattern in 9 bits
constexpr bool check_win_pattern(uint16_t m) {
    for (uint16_t mask : WIN_MASKS) {
        if ((m & mask) == mask) return true;
    }
    return false;
}

// Lookup table for winning 3x3 boards (512 entries)
struct WinTable {
    uint8_t table[512];
    constexpr WinTable() : table{} {
        for (int m = 0; m < 512; ++m) {
            table[m] = check_win_pattern(static_cast<uint16_t>(m)) ? 1 : 0;
        }
    }
    constexpr uint8_t operator[](size_t idx) const {
        return table[idx];
    }
};

constexpr WinTable WIN_TABLE;

// Result representation:
// -1: O win
//  0: Draw
//  1: X win
//  2: In progress (None in Python)
constexpr int8_t RESULT_ONGOING = 2;
constexpr int8_t RESULT_DRAW = 0;
constexpr int8_t RESULT_X = 1;
constexpr int8_t RESULT_O = -1;

#pragma pack(push, 1)
struct BoardState {
    uint16_t x_cells[9];   // 9 bits per board (18 bytes)
    uint16_t o_cells[9];   // 9 bits per board (18 bytes)
    uint16_t macro_x;      // 9 bits: boards won by X (2 bytes)
    uint16_t macro_o;      // 9 bits: boards won by O (2 bytes)
    uint16_t macro_draw;   // 9 bits: boards drawn (2 bytes)
    int8_t turn;           // 1 (X) or -1 (O) (1 byte)
    int8_t forced;         // -1 (any open board) or 0..8 (1 byte)
    int8_t result;         // 1, -1, 0, or RESULT_ONGOING (2) (1 byte)
    uint8_t padding;       // pad to 46 bytes

    static BoardState initial() {
        BoardState s;
        std::memset(&s, 0, sizeof(BoardState));
        s.turn = 1;
        s.forced = -1;
        s.result = RESULT_ONGOING;
        return s;
    }

    inline bool is_terminal() const {
        return result != RESULT_ONGOING;
    }

    inline uint16_t macro_closed() const {
        return macro_x | macro_o | macro_draw;
    }

    // Fast legal move generation into fixed array. Returns count of legal moves.
    inline int get_legal_actions(uint8_t* out_actions) const {
        if (result != RESULT_ONGOING) {
            return 0;
        }

        int count = 0;
        if (forced != -1) {
            // Forced to board 'forced'
            uint16_t open = ~(x_cells[forced] | o_cells[forced]) & 0x1FF;
            uint8_t base = static_cast<uint8_t>(forced * 9);
            while (open) {
                int c = __builtin_ctz(open);
                out_actions[count++] = base + c;
                open &= open - 1;
            }
        } else {
            // Any open board
            uint16_t open_boards = (~macro_closed()) & 0x1FF;
            while (open_boards) {
                int b = __builtin_ctz(open_boards);
                uint16_t open = ~(x_cells[b] | o_cells[b]) & 0x1FF;
                uint8_t base = static_cast<uint8_t>(b * 9);
                while (open) {
                    int c = __builtin_ctz(open);
                    out_actions[count++] = base + c;
                    open &= open - 1;
                }
                open_boards &= open_boards - 1;
            }
        }
        return count;
    }

    inline std::vector<uint8_t> legal_actions() const {
        uint8_t buf[81];
        int n = get_legal_actions(buf);
        return std::vector<uint8_t>(buf, buf + n);
    }

    inline bool is_legal(uint8_t action) const {
        if (action >= 81 || result != RESULT_ONGOING) {
            return false;
        }
        uint8_t b = action / 9;
        uint8_t c = action % 9;

        if (forced != -1 && forced != b) {
            return false;
        }
        if (macro_closed() & (1 << b)) {
            return false;
        }
        uint16_t occ = (x_cells[b] | o_cells[b]) & (1 << c);
        return occ == 0;
    }

    // In-place play for blazing fast rollouts
    inline void play_inplace(uint8_t action) {
        uint8_t b = action / 9;
        uint8_t c = action % 9;
        uint16_t bit = 1 << c;

        if (turn == 1) {
            x_cells[b] |= bit;
            if (WIN_TABLE[x_cells[b]]) {
                macro_x |= (1 << b);
                if (WIN_TABLE[macro_x]) {
                    result = RESULT_X;
                }
            } else if ((x_cells[b] | o_cells[b]) == 0x1FF) {
                macro_draw |= (1 << b);
            }
        } else {
            o_cells[b] |= bit;
            if (WIN_TABLE[o_cells[b]]) {
                macro_o |= (1 << b);
                if (WIN_TABLE[macro_o]) {
                    result = RESULT_O;
                }
            } else if ((x_cells[b] | o_cells[b]) == 0x1FF) {
                macro_draw |= (1 << b);
            }
        }

        uint16_t closed = macro_x | macro_o | macro_draw;
        if (result == RESULT_ONGOING) {
            if (closed == 0x1FF) {
                result = RESULT_DRAW;
            }
        }

        forced = (closed & (1 << c)) ? -1 : c;
        turn = -turn;
    }

    // Pure functional play: returns new BoardState
    inline BoardState play(uint8_t action) const {
        if (!is_legal(action)) {
            throw std::invalid_argument("Illegal action: " + std::to_string(action));
        }
        BoardState next = *this;
        next.play_inplace(action);
        return next;
    }

    // Unpack cells into 81-element array: 0 = empty, 1 = X, -1 = O
    inline void to_cells(int8_t* out) const {
        for (int b = 0; b < 9; ++b) {
            uint16_t x = x_cells[b];
            uint16_t o = o_cells[b];
            for (int c = 0; c < 9; ++c) {
                if (x & (1 << c)) {
                    out[b * 9 + c] = 1;
                } else if (o & (1 << c)) {
                    out[b * 9 + c] = -1;
                } else {
                    out[b * 9 + c] = 0;
                }
            }
        }
    }

    // Unpack boards into 9-element array: 0 = open, 1 = X, -1 = O, 2 = draw
    inline void to_boards(int8_t* out) const {
        for (int b = 0; b < 9; ++b) {
            uint16_t m = 1 << b;
            if (macro_x & m) {
                out[b] = 1;
            } else if (macro_o & m) {
                out[b] = -1;
            } else if (macro_draw & m) {
                out[b] = 2;
            } else {
                out[b] = 0;
            }
        }
    }

    // High performance feature encoding for neural network (289 floats)
    // Matches sttt.learning.encode() exactly:
    // 243 floats: cells in canonical perspective (empty, player, opponent)
    //  36 floats: boards in canonical perspective (open, player, opponent, drawn)
    //  10 floats: one-hot for forced+1 (0..9)
    inline void encode(float* out) const {
        std::memset(out, 0, 289 * sizeof(float));

        const uint16_t* my_cells = (turn == 1) ? x_cells : o_cells;
        const uint16_t* opp_cells = (turn == 1) ? o_cells : x_cells;
        uint16_t my_macro = (turn == 1) ? macro_x : macro_o;
        uint16_t opp_macro = (turn == 1) ? macro_o : macro_x;

        // Channel 0: empty cells (0..80)
        // Channel 1: current player cells (81..161)
        // Channel 2: opponent cells (162..242)
        for (int b = 0; b < 9; ++b) {
            uint16_t my = my_cells[b];
            uint16_t opp = opp_cells[b];
            int base = b * 9;
            for (int c = 0; c < 9; ++c) {
                uint16_t mask = 1 << c;
                int idx = base + c;
                if (my & mask) {
                    out[81 + idx] = 1.0f;
                } else if (opp & mask) {
                    out[162 + idx] = 1.0f;
                } else {
                    out[idx] = 1.0f;
                }
            }
        }

        // Boards:
        // Channel 0: open (243..251)
        // Channel 1: player won (252..260)
        // Channel 2: opp won (261..269)
        // Channel 3: draw (270..278)
        for (int b = 0; b < 9; ++b) {
            uint16_t m = 1 << b;
            if (my_macro & m) {
                out[252 + b] = 1.0f;
            } else if (opp_macro & m) {
                out[261 + b] = 1.0f;
            } else if (macro_draw & m) {
                out[270 + b] = 1.0f;
            } else {
                out[243 + b] = 1.0f;
            }
        }

        // Forced one-hot (279..288)
        out[279 + (forced + 1)] = 1.0f;
    }

    bool operator==(const BoardState& o) const {
        return std::memcmp(this, &o, sizeof(BoardState)) == 0;
    }

    bool operator!=(const BoardState& o) const {
        return !(*this == o);
    }

    std::string render() const {
        int8_t cells[81];
        int8_t boards[9];
        to_cells(cells);
        to_boards(boards);

        auto char_of = [&](int8_t v) -> char {
            if (v == 0) return '.';
            if (v == 1) return 'X';
            if (v == -1) return 'O';
            if (v == 2) return '=';
            return '?';
        };

        std::string s;
        for (int br = 0; br < 3; ++br) {
            for (int r = 0; r < 3; ++r) {
                for (int bc = 0; bc < 3; ++bc) {
                    int b = br * 3 + bc;
                    for (int c = 0; c < 3; ++c) {
                        s += char_of(cells[b * 9 + r * 3 + c]);
                        if (c < 2) s += ' ';
                    }
                    if (bc < 2) s += " | ";
                }
                s += "\n";
            }
            if (br < 2) s += "------+-------+------\n";
        }
        s += "Local boards: ";
        for (int b = 0; b < 9; ++b) {
            s += char_of(boards[b]);
            if (b < 8) s += ' ';
        }
        s += "\nNext board: ";
        if (forced < 0) {
            s += "any open board";
        } else {
            s += std::to_string(forced + 1);
        }
        return s;
    }
};
#pragma pack(pop)

// Fast pseudo-random number generator (XorShift64*)
class FastRng {
    uint64_t state;
public:
    explicit FastRng(uint64_t seed = 88172645463325252ULL) {
        state = seed ? seed : 88172645463325252ULL;
    }
    inline uint64_t next() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }
    inline uint32_t next_u32() {
        return static_cast<uint32_t>(next() >> 32);
    }
    inline uint32_t next_bounded(uint32_t bound) {
        uint64_t mult = static_cast<uint64_t>(next_u32()) * static_cast<uint64_t>(bound);
        return static_cast<uint32_t>(mult >> 32);
    }
};

} // namespace sttt
