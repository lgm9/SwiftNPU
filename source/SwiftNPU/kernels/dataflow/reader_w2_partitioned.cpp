#include <cstdint>
#include "dataflow_api.h"
using std::uint32_t;

namespace NAMESPACE {
    void MAIN {
        uint32_t w2_base = get_arg_val<uint32_t>(0);
        uint32_t Kt2 = get_arg_val<uint32_t>(1); // tiles along hidden
        uint32_t Nt = get_arg_val<uint32_t>(2); // tiles along output N
        uint32_t start_id = get_arg_val<uint32_t>(3);
        uint32_t num_tiles = get_arg_val<uint32_t>(4);
        constexpr tt::CBIndex c_W2 = tt::CBIndex::c_3;
        // This mirrors the per-output-tile ordering: for each output tile, feed Kt2 W2 tiles.
        // Address math here assumes W2 laid out in interleaved tiles (Ht x Nt), row-major by tiles.
        for (uint32_t out_idx = 0; out_idx < num_tiles; ++out_idx) {
            uint32_t tile_id = start_id + out_idx; // which output tile ID this core owns
            uint32_t n_col = tile_id % Nt; // column in tile space
            // For each reduction tile over hidden dim
            for (uint32_t k = 0; k < Kt2; ++k) {
                uint32_t w2_tile_index = k * Nt + n_col; // (Ht x Nt) layout
                uint32_t addr = w2_base + w2_tile_index * get_tile_size();
                noc_async_read(addr, c_W2, 1);
                cb_push_back(c_W2, 1);
            }
        }
    }
} // namespace NAMESPACE