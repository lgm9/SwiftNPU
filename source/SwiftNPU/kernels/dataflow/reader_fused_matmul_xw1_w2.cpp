// reader_fused_matmul_xw1_w2.cpp
//
// Streams tiles for fused two-stage matmul compute kernel:
//
// Compute kernel CB usage:
//   c_0 : X
//   c_1 : W1
//   c_2 : TMP (produced/consumed by compute)
//   c_3 : W2
//   c_16: OUT (written by compute)
//
// This reader fills:
//   CB0 (c_0) with X tiles
//   CB1 (c_1) with W1 tiles
//   CB3 (c_3) with W2 tiles
//
// Runtime args (from host):
//   0: x_addr
//   1: w1_addr
//   2: w2_addr
//   3: Mt
//   4: Kt1
//   5: Nt
//   6: work_offset
//   7: num_output_tiles  (work_per_core)
//   8: Kt2
//
// IMPORTANT NOTE ABOUT COUNTS (matches the fused two-stage compute behavior):
// Your compute kernel Stage2 consumes (num_output_tiles * Kt2) TMP tiles.
// The host code therefore sets: num_hidden_tiles = num_output_tiles * Kt2.
// To match that, Stage1 here computes hidden tiles in the order:
//   for each output tile (m,n) assigned to this core:
//     for kt2 in [0..Kt2):
//       compute hidden tile (m, kt2)  (repeated for each different n)
//
// This is not the most efficient schedule, but it is count-consistent and avoids underflow deadlock.

#include <cstdint>
#include "dataflow_api.h"

using std::uint32_t;

void kernel_main() {
    // --- Runtime args ---
    const uint32_t x_addr         = get_arg_val<uint32_t>(0);
    const uint32_t w1_addr        = get_arg_val<uint32_t>(1);
    const uint32_t w2_addr        = get_arg_val<uint32_t>(2);
    const uint32_t Mt             = get_arg_val<uint32_t>(3);
    const uint32_t Kt1            = get_arg_val<uint32_t>(4);
    const uint32_t Nt             = get_arg_val<uint32_t>(5);
    const uint32_t work_offset    = get_arg_val<uint32_t>(6);
    const uint32_t num_out_tiles  = get_arg_val<uint32_t>(7);
    const uint32_t Kt2            = get_arg_val<uint32_t>(8);

    // --- CB ids (matching tt::CBIndex values) ---
    constexpr uint32_t cb_X  = 0;  // c_0
    constexpr uint32_t cb_W1 = 1;  // c_1
    constexpr uint32_t cb_W2 = 3;  // c_3

    // Tile sizes
    const uint32_t x_tile_bytes  = get_tile_size(cb_X);
    const uint32_t w1_tile_bytes = get_tile_size(cb_W1);
    const uint32_t w2_tile_bytes = get_tile_size(cb_W2);

    // All buffers are DRAM interleaved
    constexpr bool is_dram = true;

    const InterleavedAddrGenFast<is_dram> x_gen = {
        .bank_base_address = x_addr,
        .page_size = x_tile_bytes
    };

    const InterleavedAddrGenFast<is_dram> w1_gen = {
        .bank_base_address = w1_addr,
        .page_size = w1_tile_bytes
    };

    const InterleavedAddrGenFast<is_dram> w2_gen = {
        .bank_base_address = w2_addr,
        .page_size = w2_tile_bytes
    };

    // Helper lambda: read one tile into a CB
    auto read_tile_to_cb = [](uint32_t cb_id,
                              const InterleavedAddrGenFast<is_dram>& gen,
                              uint32_t tile_id) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_addr = get_write_ptr(cb_id);
        noc_async_read_tile(tile_id, gen, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    };

    // -------------------------------------------------------
    // STAGE 1 STREAM: feed X/W1 for hidden tiles
    //
    // For each assigned output tile id -> (m,n):
    //   produce Kt2 hidden tiles for (m, kt2), kt2=0..Kt2-1
    // For each hidden tile (m, kt2), matmul reduction over kt1:
    //   X tile id  = m*Kt1 + kt1
    //   W1 tile id = kt1*Nt + kt2
    // -------------------------------------------------------
    for (uint32_t out_i = 0; out_i < num_out_tiles; ++out_i) {
        const uint32_t out_tile_id = work_offset + out_i;

        // Output tile coordinates in MxN tile grid
        const uint32_t m = out_tile_id / Nt;
        // const uint32_t n = out_tile_id % Nt; // only needed for stage2/W2

        // Produce Kt2 hidden tiles for this output tile (repeated across n)
        for (uint32_t kt2 = 0; kt2 < Kt2; ++kt2) {
            // Reduction for hidden tile (m, kt2) over Kt1
            for (uint32_t kt1 = 0; kt1 < Kt1; ++kt1) {
                const uint32_t x_tile_id  = m * Kt1 + kt1;      // X: (Mt x Kt1)
                const uint32_t w1_tile_id = kt1 * Nt + kt2;     // W1: (Kt1 x Nt)

                read_tile_to_cb(cb_X,  x_gen,  x_tile_id);
                read_tile_to_cb(cb_W1, w1_gen, w1_tile_id);
            }
        }
    }

    // -------------------------------------------------------
    // STAGE 2 STREAM: feed W2 tiles for output tiles
    //
    // For each assigned output tile id -> (m,n):
    //   for kt2=0..Kt2-1:
    //     W2 tile id = kt2*Nt + n   (W2: Nt x Nt)
    // -------------------------------------------------------
    for (uint32_t out_i = 0; out_i < num_out_tiles; ++out_i) {
        const uint32_t out_tile_id = work_offset + out_i;
        const uint32_t n = out_tile_id % Nt;

        for (uint32_t kt2 = 0; kt2 < Kt2; ++kt2) {
            const uint32_t w2_tile_id = kt2 * Nt + n;
            read_tile_to_cb(cb_W2, w2_gen, w2_tile_id);
        }
    }
}