#ifndef STTT_C_API_H
#define STTT_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t x_cells[9];
    uint16_t o_cells[9];
    uint16_t macro_x;
    uint16_t macro_o;
    uint16_t macro_draw;
    int8_t turn;
    int8_t forced;
    int8_t result;
    uint8_t padding;
} CBoardState;

void sttt_init_state(CBoardState* state);
void sttt_from_raw(CBoardState* state, const int8_t* cells, const int8_t* boards, int8_t turn, int8_t forced, int8_t result);

int sttt_legal_actions(const CBoardState* state, uint8_t* out_actions);
int sttt_is_legal(const CBoardState* state, uint8_t action);
int sttt_play(const CBoardState* state, uint8_t action, CBoardState* next_state);
void sttt_play_inplace(CBoardState* state, uint8_t action);

void sttt_get_cells(const CBoardState* state, int8_t* out_cells);
void sttt_get_boards(const CBoardState* state, int8_t* out_boards);
int8_t sttt_get_turn(const CBoardState* state);
int8_t sttt_get_forced(const CBoardState* state);
int8_t sttt_get_result(const CBoardState* state);
int sttt_is_terminal(const CBoardState* state);

void sttt_encode(const CBoardState* state, float* out_features);
void sttt_encode_batch(const CBoardState* states, int count, float* out_features);

int8_t sttt_rollout(CBoardState state, uint64_t seed);
void sttt_benchmark_rollouts(int num_games, uint64_t seed, int num_threads, double* out_elapsed, uint64_t* out_total_moves);

float sttt_evaluate(const CBoardState* state);
int sttt_search_alphabeta(const CBoardState* state, int depth, uint64_t budget);

#ifdef __cplusplus
}
#endif

#endif // STTT_C_API_H
