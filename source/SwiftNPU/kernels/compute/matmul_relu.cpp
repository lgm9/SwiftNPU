#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/relu.h"   // ReLU SFPU op

using std::uint32_t;

namespace NAMESPACE {

void MAIN {
    // Runtime args
    // 0: num_out_tiles (how many output tiles this core produces)
    // 1: Kt            (reduction tiles for A@B)
    // 2: act_kind      (0:none, 1:ReLU)
    uint32_t num_out_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt            = get_arg_val<uint32_t>(1);
    uint32_t act_kind      = get_arg_val<uint32_t>(2);

    // CB layout for matmul + optional ReLU
    constexpr tt::CBIndex c_X   = tt::CBIndex::c_0;
    constexpr tt::CBIndex c_B   = tt::CBIndex::c_1;
    constexpr tt::CBIndex c_OUT = tt::CBIndex::c_16;

    mm_init(c_X, c_B, c_OUT);

    if (act_kind == 1) {
        relu_tile_init();
    }

    for (uint32_t i = 0; i < num_out_tiles; ++i) {
        acquire_dst();

        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(c_X, 1);
            cb_wait_front(c_B, 1);

            // Accumulate into DST tile 0
            matmul_tiles(c_X, c_B, 0, 0, 0, false);

            cb_pop_front(c_X, 1);
            cb_pop_front(c_B, 1);
        }

        // Epilogue activation on the accumulated output tile
        if (act_kind == 1) {
            relu_tile(0);     // operates on DST tile 0 in-place
        }

        cb_reserve_back(c_OUT, 1);
        pack_tile(0, c_OUT);
        cb_push_back(c_OUT, 1);

        release_dst();
    }
}

} // namespace NAMESPACE