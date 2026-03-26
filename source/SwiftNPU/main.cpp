#include <random>
#include <filesystem>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/device.hpp>

#include <fmt/core.h>
#include <x86intrin.h>
#include "cxxopts.hpp"
#include "core.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

struct TimesTaken {
    uint64_t write_time = 0;
    uint64_t exec_time  = 0;
    uint64_t read_time  = 0;
    uint64_t allocation_time = 0;
};

struct WorkItem {
    uint32_t N = 0;
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t cores_required = 0;
};

struct Allocation {
    core_set_t cores;
    uint32_t N = 0;
    uint32_t M = 0;
    uint32_t K = 0;
};


static inline float bf16_to_f32(const bfloat16& x) {
    return x.to_float();
}


static std::vector<bfloat16> zero_pad_matrix_rowmajor(
    const std::vector<bfloat16>& src,
    uint32_t rows,
    uint32_t cols,
    uint32_t padded_rows,
    uint32_t padded_cols) {

    std::vector<bfloat16> dst((size_t)padded_rows * (size_t)padded_cols, bfloat16(0.0f));
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            dst[r * padded_cols + c] = src[r * cols + c];
        }
    }
    return dst;
}

static std::vector<bfloat16> crop_matrix_rowmajor(
    const std::vector<bfloat16>& src,
    uint32_t rows,
    uint32_t cols,
    uint32_t padded_cols) {

    std::vector<bfloat16> dst((size_t)rows * (size_t)cols);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            dst[r * cols + c] = src[r * padded_cols + c];
        }
    }
    return dst;
}

static void print_matrix_bf16_rowmajor(
    std::ostream& os,
    const std::vector<bfloat16>& v,
    uint32_t rows,
    uint32_t cols,
    const std::string& name) {

    os << name << " (" << rows << "x" << cols << ")\n";
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            os << bf16_to_f32(v[r * cols + c]) << (c + 1 == cols ? '\n' : ' ');
        }
    }
}

static std::vector<float> cpu_matmul_ref(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    uint32_t N,
    uint32_t M,
    uint32_t K) {

    std::vector<float> out((size_t)N * (size_t)M, 0.0f);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += bf16_to_f32(a[n * K + k]) * bf16_to_f32(b[k * M + m]);
            }
            out[n * M + m] = acc;
        }
    }
    return out;
}

static void print_matrix_f32_rowmajor(
    std::ostream& os,
    const std::vector<float>& v,
    uint32_t rows,
    uint32_t cols,
    const std::string& name) {

    os << name << " (" << rows << "x" << cols << ")\n";
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            os << v[r * cols + c] << (c + 1 == cols ? '\n' : ' ');
        }
    }
}

static void setup_given_cores(
    Program& program,
    std::shared_ptr<tt::tt_metal::Buffer>& src0_dram_buffer,
    std::shared_ptr<tt::tt_metal::Buffer>& src1_dram_buffer,
    std::shared_ptr<tt::tt_metal::Buffer>& dst_dram_buffer,
    CoreRangeSet& core_rangesets,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t single_tile_size,
    uint32_t num_output_tiles_total,
    uint32_t act_kind) {

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_rangesets, num_output_tiles_total);

    const auto cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t num_input_tiles = 2;

    // CB0: A, CB1: B, CB16: OUT
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    MathFidelity math_fidelity = MathFidelity::HiFi4;

    // Reader
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);

    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "source/SwiftNPU/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);

    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "source/SwiftNPU/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute
    auto compute_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "source/SwiftNPU/kernels/compute/matmul_relu.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {src0_dram_buffer->address(),
                     src1_dram_buffer->address(),
                     Mt,
                     Kt,
                     Nt,
                     work_offset,
                     work_per_core});

                tt_metal::SetRuntimeArgs(
                    program,
                    writer_id,
                    core,
                    {dst_dram_buffer->address(), work_per_core, work_offset});

                // matmul_relu.cpp expects: (num_out_tiles, Kt, act_kind)
                tt_metal::SetRuntimeArgs(
                    program,
                    compute_id,
                    core,
                    {work_per_core, Kt, act_kind});

                work_offset += work_per_core;
            }
        }
    }
}


static void warmup_once(IDevice* device, uint32_t act_kind) {
    CommandQueue& cq = device->command_queue();
    Program program{};

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    const uint32_t N = 32;
    const uint32_t M = 32;
    const uint32_t K = 32;
    const uint32_t Nt = 1;
    const uint32_t Mt = 1;
    const uint32_t Kt = 1;

    std::vector<bfloat16> a_rowmaj(N * K, bfloat16(1.0f));
    std::vector<bfloat16> b_rowmaj(K * M, bfloat16(1.0f));
    auto a_tiled = tilize_nfaces(a_rowmaj, N, K);
    auto b_tiled = tilize_nfaces(b_rowmaj, K, M);
    std::vector<bfloat16> out_tiled(TILE_HEIGHT * TILE_WIDTH, bfloat16(0.0f));

    tt_metal::InterleavedBufferConfig cfg{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_buffer = CreateBuffer(cfg);
    auto src1_buffer = CreateBuffer(cfg);
    auto dst_buffer = CreateBuffer(cfg);

    CoreCoord core{0, 0};
    CoreRangeSet core_range_set({CoreRange(core, core)});

    setup_given_cores(
        program,
        src0_buffer,
        src1_buffer,
        dst_buffer,
        core_range_set,
        Mt, Nt, Kt,
        single_tile_size,
        Nt * Mt,
        act_kind);

    EnqueueWriteBuffer(cq, src0_buffer, a_tiled.data(), false);
    EnqueueWriteBuffer(cq, src1_buffer, b_tiled.data(), false);
    Finish(cq);

    EnqueueProgram(cq, program, true);
    Finish(cq);

    EnqueueReadBuffer(cq, dst_buffer, out_tiled.data(), false);
    Finish(cq);
}

static void scheduler_execute_once(
    IDevice* device,
    std::ofstream& outfile,
    const std::vector<Allocation>& current_allocations,
    TimesTaken& total_times_taken,
    uint32_t act_kind,
    uint32_t debug_option) {

    CommandQueue& cq = device->command_queue();
    Program program{};

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    const int T = (int)current_allocations.size();
    TT_ASSERT(T > 0, "scheduler_execute_once called with empty allocations");

    // Per-tenant host buffers (row-major + tiled + output tiled)
    std::vector<std::vector<bfloat16>> a_rowmaj(T), b_rowmaj(T);
    std::vector<std::vector<bfloat16>> a_tiled(T), b_tiled(T), out_tiled(T);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    // Per-tenant device buffers
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> src0_buffers;
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> src1_buffers;
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> dst_buffers;
    src0_buffers.reserve(T);
    src1_buffers.reserve(T);
    dst_buffers.reserve(T);

    // Create buffers + inputs per tenant
    for (int i = 0; i < T; i++) {
        const uint32_t N = current_allocations[i].N;
        const uint32_t M = current_allocations[i].M;
        const uint32_t K = current_allocations[i].K;

        //Dividing into tile size
        const uint32_t Nt = (N + TILE_HEIGHT - 1) / TILE_HEIGHT;
        const uint32_t Mt = (M + TILE_WIDTH  - 1) / TILE_WIDTH;
        const uint32_t Kt = (K + TILE_WIDTH  - 1) / TILE_WIDTH;

        const uint32_t a_tiles   = Nt * Kt; // A: NxK
        const uint32_t b_tiles   = Kt * Mt; // B: KxM
        const uint32_t out_tiles = Nt * Mt; // Out: NxM

        const uint32_t a_bytes   = single_tile_size * a_tiles;
        const uint32_t b_bytes   = single_tile_size * b_tiles;
        const uint32_t out_bytes = single_tile_size * out_tiles;

        // Host inputs are all ones. If dimensions are not tile-aligned, zero-pad before tilize.
        a_rowmaj[i].resize((size_t)N * (size_t)K);
        b_rowmaj[i].resize((size_t)K * (size_t)M);
        for (auto& v : a_rowmaj[i]) v = bfloat16(dist(rng));
        for (auto& v : b_rowmaj[i]) v = bfloat16(dist(rng));

        const uint32_t padded_N = Nt * TILE_HEIGHT;
        const uint32_t padded_M = Mt * TILE_WIDTH;
        const uint32_t padded_K = Kt * TILE_WIDTH;

        auto a_padded = zero_pad_matrix_rowmajor(a_rowmaj[i], N, K, padded_N, padded_K);
        auto b_padded = zero_pad_matrix_rowmajor(b_rowmaj[i], K, M, padded_K, padded_M);

        a_tiled[i] = tilize_nfaces(a_padded, padded_N, padded_K);
        b_tiled[i] = tilize_nfaces(b_padded, padded_K, padded_M);

        out_tiled[i].resize(out_bytes / sizeof(bfloat16), bfloat16(0.0f));

        // Device buffers
        tt_metal::InterleavedBufferConfig cfg{
            .device = device,
            .size = 0,
            .page_size = single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        cfg.size = a_bytes;
        src0_buffers.push_back(CreateBuffer(cfg));
        cfg.size = b_bytes;
        src1_buffers.push_back(CreateBuffer(cfg));
        cfg.size = out_bytes;
        dst_buffers.push_back(CreateBuffer(cfg));
    }

    // Program setup per tenant
    for (int i = 0; i < T; i++) {
        CoreRangeSet core_range_set = current_allocations[i].cores.to_core_range_set();

        const uint32_t N = current_allocations[i].N;
        const uint32_t M = current_allocations[i].M;
        const uint32_t K = current_allocations[i].K;

        const uint32_t Nt = (N + TILE_HEIGHT - 1) / TILE_HEIGHT;
        const uint32_t Mt = (M + TILE_WIDTH  - 1) / TILE_WIDTH;
        const uint32_t Kt = (K + TILE_WIDTH  - 1) / TILE_WIDTH;

        const uint32_t num_output_tiles_total = Nt * Mt;

        setup_given_cores(
            program,
            src0_buffers[i],
            src1_buffers[i],
            dst_buffers[i],
            core_range_set,
            Mt, Nt, Kt,
            single_tile_size,
            num_output_tiles_total,
            act_kind);
    }

    // Single measured execution
    unsigned int aux = 0;
    uint64_t t0 = 0, t1 = 0;

    t0 = __rdtscp(&aux);
    for (int i = 0; i < T; i++) {
        TT_ASSERT(src0_buffers[i]->size() <= a_tiled[i].size() * sizeof(bfloat16), "A host tiled too small");
        TT_ASSERT(src1_buffers[i]->size() <= b_tiled[i].size() * sizeof(bfloat16), "B host tiled too small");
        EnqueueWriteBuffer(cq, src0_buffers[i], a_tiled[i].data(), false);
        EnqueueWriteBuffer(cq, src1_buffers[i], b_tiled[i].data(), false);
    }
    Finish(cq);
    t1 = __rdtscp(&aux);
    uint64_t write_cycles = t1 - t0;

    t0 = __rdtscp(&aux);
    EnqueueProgram(cq, program, true);
    Finish(cq);
    t1 = __rdtscp(&aux);
    uint64_t exec_cycles = t1 - t0;

    t0 = __rdtscp(&aux);
    for (int i = 0; i < T; i++) {
        TT_ASSERT(dst_buffers[i]->size() <= out_tiled[i].size() * sizeof(bfloat16), "OUT host tiled too small");
        EnqueueReadBuffer(cq, dst_buffers[i], out_tiled[i].data(), false);
    }
    Finish(cq);
    t1 = __rdtscp(&aux);
    uint64_t read_cycles = t1 - t0;

    if (debug_option) {
        for (int i = 0; i < T; i++) {
            const uint32_t N = current_allocations[i].N;
            const uint32_t M = current_allocations[i].M;
            const uint32_t K = current_allocations[i].K;

            const uint32_t Nt = (N + TILE_HEIGHT - 1) / TILE_HEIGHT;
            const uint32_t Mt = (M + TILE_WIDTH  - 1) / TILE_WIDTH;

            const uint32_t padded_N = Nt * TILE_HEIGHT;
            const uint32_t padded_M = Mt * TILE_WIDTH;

            auto out_padded_rowmaj = untilize_nfaces(out_tiled[i], padded_N, padded_M);
            auto out_rowmaj = crop_matrix_rowmajor(out_padded_rowmaj, N, M, padded_M);

            auto golden = cpu_matmul_ref(a_rowmaj[i], b_rowmaj[i], N, M, K);

            outfile << "\n===== workload " << i
                    << " : N=" << N << " M=" << M << " K=" << K << " =====\n";
            print_matrix_bf16_rowmajor(outfile, a_rowmaj[i], N, K, "A");
            print_matrix_bf16_rowmajor(outfile, b_rowmaj[i], K, M, "B");
            print_matrix_f32_rowmajor(outfile, golden, N, M, "CPU golden C");
            print_matrix_bf16_rowmajor(outfile, out_rowmaj, N, M, "Device C");
        }
    }

    outfile << "Write cycles: " << write_cycles << "\n";
    outfile << "Exec  cycles: " << exec_cycles << "\n";
    outfile << "Read  cycles: " << read_cycles << "\n";

    total_times_taken.write_time += write_cycles;
    total_times_taken.exec_time  += exec_cycles;
    total_times_taken.read_time  += read_cycles;
}

static void schedule_matmul_relu_multicore(
    IDevice* device,
    std::ofstream& outfile,
    std::deque<WorkItem>& workload_sizes,
    uint32_t visualize_option,
    uint32_t debug_option,
    uint32_t alloc_algorithm,
    uint32_t act_kind) {

    auto core_grid_hw = device->compute_with_storage_grid_size();
    core_grid_t core_grid(core_grid_hw.x, core_grid_hw.y);
    const int num_total_cores = core_grid_hw.x * core_grid_hw.y;

    TimesTaken total_times_taken{};
    std::vector<Allocation> current_allocations;
    uint32_t num_executions = 0;
    core_grid.free_all();

    while (!workload_sizes.empty()) {
        uint64_t alloc_overhead_this_run = 0;
        uint32_t num_in_use = 0;

        while (!workload_sizes.empty()) {
            WorkItem w = workload_sizes.front();
            uint32_t num_cores_required = w.cores_required;

            // choose rectangle for requested cores
            std::pair<int, int> shape = {1, (int)num_cores_required};
            for (int i = 2; i <= 10; i++) {
                if ((uint32_t)(i * i) > num_cores_required) break;
                if (num_cores_required % (uint32_t)i == 0) shape = {i, (int)(num_cores_required / (uint32_t)i)};
            }

            unsigned int aux = 0;
            uint64_t t0 = 0, t1 = 0;
            core_set_t placed;

            t0 = __rdtscp(&aux);
            if      (alloc_algorithm == 0)  { placed = core_grid.allocate_first_fit(shape.first, shape.second); }
            else if (alloc_algorithm == 1)  { placed = core_grid.allocate_best_fit(shape.first, shape.second); }
            else if (alloc_algorithm == 2)  { placed = core_grid.allocate_LSSA(shape.first, shape.second); }
            else if (alloc_algorithm == 3)  { placed = core_grid.allocate_ASFF(shape.first * shape.second); }
            else if (alloc_algorithm == 4)  { placed = core_grid.allocate_ASBF(shape.first * shape.second); }
            else if (alloc_algorithm == 5)  { placed = core_grid.allocate_NAS(shape.first, shape.second); }
            else if (alloc_algorithm == 6)  { placed = core_grid.allocate_GED(shape.first, shape.second); }
            else {
                placed = core_grid.allocate_first_fit(shape.first, shape.second);
            }
            t1 = __rdtscp(&aux);

            alloc_overhead_this_run += (t1 - t0);

            if (placed.empty()) break;
            num_in_use += num_cores_required;
            workload_sizes.pop_front();
            current_allocations.push_back({placed, w.N, w.M, w.K});
        }

        if (current_allocations.empty()) {
            // Can't place anything at all
            cout << "Error: could not place workload;\n";
            exit(2);
        }

        if (visualize_option) {
            // Visualize
            char ch = 'A';
            char map[32][32];
            std::memset(map, '.', sizeof(map));

            for (const auto& item : current_allocations) {
                for (const auto core : item.cores.cores) {
                    map[core.x][core.y] = ch;
                }
                ch++;
            }

            for (int y = 0; y < core_grid_hw.y; ++y) {
                for (int x = 0; x < core_grid_hw.x; ++x) {
                    outfile << (unsigned char)map[x][y];
                }
                outfile << "\n";
            }
        }
        num_executions++;

        outfile << "Run " << num_executions << " Utilization: " << num_in_use << " / " << num_total_cores << " "
                << 100.0 * num_in_use / num_total_cores << "%\n";

        scheduler_execute_once(
            device, outfile, current_allocations,
            total_times_taken,
            act_kind, debug_option);

        total_times_taken.allocation_time += alloc_overhead_this_run;
        current_allocations.clear();
        core_grid.free_all();
    }

    outfile << "Total executions: " << num_executions << "\n";
    outfile << "Total write cycles: " << total_times_taken.write_time << "\n";
    outfile << "Total read cycles: " << total_times_taken.read_time << "\n";
    outfile << "Total allocation overhead: " << total_times_taken.allocation_time << "\n";
    outfile << "Total exec cycles: " << total_times_taken.exec_time << "\n";
    outfile << "Total transfer cycles: " << (total_times_taken.write_time + total_times_taken.read_time) << "\n";
    outfile << "End-to-end time: " << (total_times_taken.allocation_time + total_times_taken.exec_time +
                                       total_times_taken.write_time + total_times_taken.read_time) << "\n";
    outfile.flush();
}

int main(int argc, char* argv[]) {
    
    cxxopts::Options options("scheduler_matmul_relu", "Single matmul + optional ReLU via matmul_relu.cpp (varying N,M,K per line)");
    options.add_options()
        ("allocs", "Alloc file: lines are 'N M K cores_required'",
            cxxopts::value<std::string>()->default_value("results/shapes.txt"))
        ("out_dir", "Output directory for results",
            cxxopts::value<std::string>()->default_value("results/scheduler_matmul_relu"))
        ("out", "Output file name",
            cxxopts::value<std::string>()->default_value("result_scheduler_matmul_relu.txt"))
        ("visualize_option", "Print allocated cores to result file",
            cxxopts::value<uint32_t>()->default_value("0"))
        ("debug_option", "Print A, B, CPU golden C, and device C to result file",
            cxxopts::value<uint32_t>()->default_value("0"))
        ("alloc_algorithm", "Allocation algorithm",
            cxxopts::value<uint32_t>()->default_value("0"))
        ("workload_size", "Number of workloads (lines) to read",
            cxxopts::value<uint32_t>()->default_value("500"))
        ("act_kind", "0:none, 1:ReLU",
            cxxopts::value<uint32_t>()->default_value("1"));

    auto parse_result = options.parse(argc, argv);

    const std::string ALLOC_FILE   = parse_result["allocs"].as<std::string>();
    const std::string OUTPUT_DIR   = parse_result["out_dir"].as<std::string>();
    const std::string OUTPUT_FILE  = parse_result["out"].as<std::string>();
    const uint32_t    visualize_option = parse_result["visualize_option"].as<uint32_t>();
    const uint32_t    debug_option = parse_result["debug_option"].as<uint32_t>();
    const uint32_t    alloc_algorithm = parse_result["alloc_algorithm"].as<uint32_t>();
    const uint32_t    act_kind     = parse_result["act_kind"].as<uint32_t>();
    const uint32_t    workload_n   = parse_result["workload_size"].as<uint32_t>();

    std::filesystem::create_directories(OUTPUT_DIR);
    std::string out_path = (std::filesystem::path(OUTPUT_DIR) / OUTPUT_FILE).string();
    std::ofstream outfile(out_path);
    if (!outfile.is_open()) {
        std::cerr << "Error: could not open " << out_path << " for writing.\n";
        return 1;
    }

    std::ifstream alloc_file(ALLOC_FILE);
    if (!alloc_file.is_open()) {
        std::cerr << "Error: could not open " << ALLOC_FILE << " for reading.\n";
        return 1;
    }

    std::deque<WorkItem> workload_sizes;

    uint32_t n, m, k, c;
    for (uint32_t i = 0; i < workload_n; i++) {
        if (!(alloc_file >> n >> m >> k >> c)) break;
        workload_sizes.push_back({n, m, k, c});
    }

    // Return on empty shape.txt
    if (workload_sizes.empty()) {
        outfile << "ERROR: alloc file empty, try generating shapes first\n";
        return 1;
    }

    std::cout << "allocs: " << ALLOC_FILE << "\n";
    std::cout << "workloads read: " << workload_sizes.size() << "\n";

    bool pass = true;
    try {
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        warmup_once(device, act_kind);

        schedule_matmul_relu_multicore(
            device,
            outfile,
            workload_sizes,
            visualize_option,
            debug_option,
            alloc_algorithm,
            act_kind);

        pass &= CloseDevice(device);
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n{}\n", e.what());
        throw;
    }

    if (pass) fmt::print("Test Passed\n");
    else TT_THROW("Test Failed");

    TT_ASSERT(pass);
    return 0;
}
