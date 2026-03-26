// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <fmt/core.h>
#include <x86intrin.h>
#include "cxxopts.hpp"
#include <regex>
#include "core.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Reference implementation of matrix multiplication.
// Array A is of size MxK, Array B is of size KxN, and the output C is of size MxN.
// The implementation is bare bones and does not include optimizations such as tiling or vectorization.
// This is intended to be used as a golden reference for testing the Metalium implementation.
void golden_matmul(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j + (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

void multi_core_for_one_tenant(
    CommandQueue& cq,
    Program& program,
    std::shared_ptr<tt::tt_metal::Buffer>& src0_dram_buffer,
    std::shared_ptr<tt::tt_metal::Buffer>& src1_dram_buffer,
    std::shared_ptr<tt::tt_metal::Buffer>& dst_dram_buffer,
    CoreRangeSet& core_rangesets,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t single_tile_size,
    uint32_t num_output_tiles_total) {
    std::cout << "use count: " << src0_dram_buffer.use_count() << "\n" << std::flush;

    std::cout << "Number of cores requested: " << core_rangesets.num_cores() << "\n" << std::flush;
    std::cout << "core_rangesets: " << core_rangesets.str() << "\n" << std::flush;
    std::cout << "Total number of output tiles: " << num_output_tiles_total << "\n" << std::flush;

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_rangesets, num_output_tiles_total);
    std::cout << "all_cores: " << all_cores.str() << "\n" << std::flush;
    std::cout << "core_group_1 size: " << core_group_1.size() << ", work_per_core1: " << work_per_core1 << "\n"
              << std::flush;
    std::cout << "core_group_2 size: " << core_group_2.size() << ", work_per_core2: " << work_per_core2 << "\n"
              << std::flush;
    std::cout << "matmul_multicore_multitenant.cpp: Number of cores allocated: " << num_cores << "\n";

    // Configure Circular Buffers
    // Circular buffers act as staging areas for data movement between DRAM and compute units.
    // Using 2 tiles per circular buffer to allow for double buffering (data movement can be reading from one tile while
    // the compute kernel is using the other tile). This number can be adjusted based on the use case, but generally
    // diminishing returns are observed after several tiles.
    // input tiles count is = 2 so one tile can be read while the other is being processed
    const auto cb_data_format = tt::DataFormat::Float16_b;
    uint32_t num_input_tiles = 2;
    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,  // create on all cores
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size));

    // Create Kernels (Reader, Writer, Compute)
    // - Reader kernel: Handles reading input data from DRAM into circular buffers
    // - Writer kernel: Handles writing output data from circular buffers back to DRAM
    // - Compute kernel: Performs the actual matrix multiplication computation
    // All kernels run across all cores to enable parallel execution
    MathFidelity math_fidelity = MathFidelity::HiFi4;  // High fidelity math for accurate results
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    auto compute_kernel_id = tt_metal::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "matmul/matmul_multi_core/kernels/compute/mm.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = {}});

    // Set Runtime Arguments for Kernels
    // Each core needs to know which portion of the work it's responsible for. We are parallelizing across output
    // tiles - each core computes different output tiles. Runtime arguments can be changed between program executions
    // without recompilation.
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    int i = 0;

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // std::cout << "Core: " << core.str() << ", work_offset: " << work_offset
                //           << ", work_per_core: " << work_per_core << "\n"
                //           << std::flush;
                // Set arguments for the reader kernel (data input)
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {src0_dram_buffer->address(),  // Address of matrix A in DRAM
                     src1_dram_buffer->address(),  // Address of matrix B in DRAM
                     Mt,                           // Number of tiles in M dimension
                     Kt,                           // Number of tiles in K dimension
                     Nt,                           // Number of tiles in N dimension
                     work_offset,                  // Starting offset for this core's work
                     work_per_core});              // Amount of work for this core

                // Set arguments for the writer kernel (data output)
                tt_metal::SetRuntimeArgs(
                    program, writer_id, core, {dst_dram_buffer->address(), work_per_core, work_offset});

                // Set arguments for the compute kernel
                tt_metal::SetRuntimeArgs(
                    program,
                    compute_kernel_id,
                    core,
                    {work_per_core,            // Amount of work for this core
                     Kt});                     // Number of tiles in K dimension for dot product
                work_offset += work_per_core;  // Update offset for next core
                i++;
            }
        }
    }
    std::cout << "marker1" << std::endl << std::flush;
}

/**
 * @brief Multi-core matrix multiplication using SPMD (Single Program, Multiple Data) parallelization.
 *
 * Performs C = A * B matrix multiplication by distributing output tiles across multiple cores.
 * Each core runs the same program but works on different portions of the output matrix,
 * making this a simple and efficient parallelization scheme.
 *
 * The function uses three types of kernels running in parallel:
 * - Reader: Loads input matrix tiles from DRAM into circular buffers
 * - Compute: Performs tile-wise matrix multiplication (A_tile * B_tile = C_tile)
 * - Writer: Stores computed output tiles back to DRAM
 *
 * Work distribution is handled automatically - if output tiles don't divide evenly
 * across cores, some cores get one extra tile to balance the workload.
 *
 * @param a Input matrix A in row-major format (bfloat16 elements)
 * @param b Input matrix B in row-major format (bfloat16 elements)
 * @param output Output matrix C to store A*B result (bfloat16 elements)
 * @param M Number of rows in matrix A and output matrix C
 * @param N Number of columns in matrix B and output matrix C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param device Target device for computation
 *
 * @note Matrix dimensions must be divisible by tile size (32x32) for this implementation
 * @note Uses circular buffers with 2 tiles for double-buffering to overlap compute and data movement
 */
void matmul_multi_core(
    std::vector<bfloat16>& a,
    std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    IDevice* device,
    std::ofstream& outfile,
    std::vector<CoreRangeSet>& core_allocations) {
    // Check if the configuration is valid - matrices must be divisible by tile dimensions
    TT_ASSERT(
        (M * N) % TILE_HW == 0,
        "Matrix dimensions M={} and N={} must be divisible by TILE_HW={} to use this matmul implementation",
        M,
        N,
        TILE_HW);

    // Setup the device and command queue for multi-core execution
    CommandQueue& cq = device->command_queue();
    Program program{};

    auto core_grid = device->compute_with_storage_grid_size();

    std::cout << "core grid" << " x " << core_grid.x << " y " << core_grid.y << std::endl;

    uint32_t NUM_TENANTS = core_allocations.size();
    std::cout << "Number of tenants: " << NUM_TENANTS << "\n";

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    // Get the compute grid size to determine how many cores are available
    // auto core_grid = device->compute_with_storage_grid_size();
    auto num_output_tiles_total = (M * N) / TILE_HW;

    // Use the split_work_to_cores utility function to distribute matrix multiplication work
    // across available cores for efficient SPMD (Single Program, Multiple Data) execution.
    // This function takes the total number of output tiles and available cores, then calculates
    // how to divide the work when it cannot be evenly distributed. It returns two groups of cores:
    // - Primary group: handles more tiles per core
    // - Secondary group: handles fewer tiles per core
    // The secondary group is empty if the work can be evenly distributed across all cores. This
    // approach minimizes workload imbalance between cores for optimal performance.

    // Extracting Matrix dimensions from input/output vectors and converting to tile coordinates.
    // The accelerator works with 32x32 tiles, so we need to convert from element dimensions
    // to tile dimensions for proper addressing and computation.
    const uint32_t Mt = M / TILE_HEIGHT;  // Number of tiles in M dimension
    const uint32_t Kt = K / TILE_WIDTH;   // Number of tiles in K dimension
    const uint32_t Nt = N / TILE_WIDTH;   // Number of tiles in N dimension

    // Create DRAM Buffers for input and output vectors.
    // We allocate DRAM buffers for the input matrices and output matrix.
    // Setting page_size to single_tile_size is the most common configuration for memory buffers in Metalium
    // as it is generic, works for most cases and achieves good performance.
    // Writing data from input vectors to source buffers.
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes

    tt_metal::InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = single_tile_size * Mt * Kt,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = single_tile_size * Nt * Kt,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = single_tile_size * Mt * Nt,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // Creating buffers for each tenant.
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> src0_dram_buffers;
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> src1_dram_buffers;
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> dst_dram_buffers;
    for (int i = 0; i < NUM_TENANTS; i++) {
        src0_dram_buffers.push_back(CreateBuffer(dram_config_A));
        src1_dram_buffers.push_back(CreateBuffer(dram_config_B));
        dst_dram_buffers.push_back(CreateBuffer(dram_config_C));
    }

    // std::cout << "placeholder" << "\n" << std::flush;

    // Create kernels and circular buffers, and enqueue them for each tenant.
    for (int i = 0; i < NUM_TENANTS; i++) {
        std::cout << "i: " << i << "\n" << std::flush;
        auto src0 = src0_dram_buffers[i];
        auto src1 = src1_dram_buffers[i];
        auto dst = dst_dram_buffers[i];

        std::cout << "use count: " << src0_dram_buffers[i].use_count() << "\n" << std::flush;
        multi_core_for_one_tenant(
            cq, program, src0, src1, dst, core_allocations[i], Mt, Nt, Kt, single_tile_size, num_output_tiles_total);
        std::cout << "marker 3" << std::endl << std::flush;
    }
    std::cout << "marker2" << std::endl << std::flush;

    // Launch program & read in output buffer result into the host vector
    // 1. Upload input data to DRAM buffers
    // 2. Execute the program (all kernels run in parallel across cores)
    // 3. Read back the result from DRAM to host memory
    // The 'true' parameter in EnqueueReadBuffer ensures we wait for completion (so when the function
    // returns, the output vector is fully populated).

    // std::cout << "number of cores" << num_cores << "\n";

    // Enqueue the program for execution
    // Measure the time taken for each step and log it to the output file.
    unsigned int aux = 0;
    uint64_t t0 = __rdtscp(&aux);
    uint64_t t1 = __rdtscp(&aux);

    // log dimension and core ranges
    outfile << "dimension: " << M << " x " << M << "\n";
    outfile << "core_ranges: " << "\n";
    for (const auto& range : core_allocations) {
        outfile << "    " << range.str() << "\n";
    }

    // log times taken for moving data to DRAM buffers
    for (int i = 0; i < NUM_TENANTS; i++) {
        EnqueueWriteBuffer(cq, src0_dram_buffers[i], a.data(), true);
        Finish(cq);
        t1 = __rdtscp(&aux);
        outfile << "Time taken to write src0: " << (t1 - t0) << "[Partition " << i << "]" << "\n";
        t0 = t1;

        EnqueueWriteBuffer(cq, src1_dram_buffers[i], b.data(), true);
        Finish(cq);
        t1 = __rdtscp(&aux);
        outfile << "Time taken to write src1: " << (t1 - t0) << "[Partition " << i << "]" << "\n";
        t0 = t1;
    }

    // log time taken for matrix multiplication
    for (int i = 0; i < 31; i++) {  // ----- Timing + batched execution -----
        t0 = __rdtscp(&aux);

        EnqueueProgram(cq, program, true);
        Finish(cq);
        t1 = __rdtscp(&aux);
        outfile << "Time taken: " << (t1 - t0) << "\n";
    }
    // log time taken for moving data from DRAM buffers to output vector
    for (int i = 0; i < NUM_TENANTS; i++) {
        EnqueueReadBuffer(cq, dst_dram_buffers[i], output.data(), true);
        Finish(cq);
        t1 = __rdtscp(&aux);
        outfile << "Time taken last: " << (t1 - t0) << "[Partition " << i << "]" << "\n";
        t0 = t1;
    }
}

///////////////////////////////////////

int main(int argc, char* argv[]) {
    // cxx options
    cxxopts::Options options("tt_loopback", "Loopback test program");
    options.add_options()(
        "allocs",
        "A File containing list of core allocations for each tenant",
        cxxopts::value<std::string>()->default_value("matmul/matmul_multicore_anyshape/allocations.txt"));
    // dim_len is the value of n for n x n matrix multiplication
    options.add_options()("dim_len", "Dimension length for matrices", cxxopts::value<uint32_t>()->default_value("640"));
    options.add_options()(
        "out_dir",
        "Output directory for results",
        cxxopts::value<std::string>()->default_value("results/matmul_multicore_multitenant"));
    options.add_options()(
        "out",
        "Output file name",
        cxxopts::value<std::string>()->default_value("result_matmul_multicore_multitenant.txt"));

    auto parse_result = options.parse(argc, argv);

    const string ALLOC_FILE = parse_result["allocs"].as<std::string>();

    const uint32_t DIM_LEN = parse_result["dim_len"].as<uint32_t>();
    std::cout << "DIM_LEN: " << DIM_LEN << "\n";

    const std::string OUTPUT_DIR = parse_result["out_dir"].as<std::string>();
    const std::string OUTPUT_FILE = parse_result["out"].as<std::string>();

    std::filesystem::create_directories(OUTPUT_DIR);
    std::string path = (std::filesystem::path(OUTPUT_DIR) / OUTPUT_FILE).string();
    std::ofstream outfile(path, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: could not open " << path << " for writing." << std::endl;
        return 1;
    }

    std::ifstream alloc_file(ALLOC_FILE, std::ios::in);
    if (!alloc_file.is_open()) {
        std::cerr << "Error: could not open " << ALLOC_FILE << " for reading." << std::endl;
        return 1;
    }

    std::string line;

    // A line contains coordinates in the format "x,y - x,y"
    std::regex re("\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*-\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*");
    std::smatch match;

    std::vector<core_set_t> core_sets;
    std::vector<CoreRange> current_core_ranges;
    core_set_t core_set;

    while (std::getline(alloc_file, line)) {
        // std::cout << "Processing line: " << line << "\n";
        if (std::regex_match(line, match, re)) {
            // std::cout << "1" << std::endl;
            uint32_t x1, y1, x2, y2;
            x1 = std::stoi(match[1].str());
            y1 = std::stoi(match[2].str());
            x2 = std::stoi(match[3].str());
            y2 = std::stoi(match[4].str());
            if (x1 == x2) {
                if (y1 > y2) {
                    std::swap(y1, y2);  // Ensure y1 is less than or equal to y2
                }
                for (uint32_t y = y1; y <= y2; ++y) {
                    core_set.core_add(x1, y);
                }
            } else if (y1 == y2) {
                if (x1 > x2) {
                    std::swap(x1, x2);  // Ensure x1 is less than or equal to x2
                }
                for (uint32_t x = x1; x <= x2; ++x) {
                    core_set.core_add(x, y1);
                }
            } else {
                std::cerr << "Error: Invalid core range format in line: " << line << std::endl;
                return 1;
            }
        } else if (!core_set.empty()) {
            core_sets.push_back(core_set);
            core_set = core_set_t();  // Reset for the next core range
        } else {
            // std::cout << "3" << std::endl;
            // ignore
            continue;
        }
    }
    if (!core_set.empty()) {
        // std::cout << "4" << std::endl;
        // complete the last core rangeset
        core_sets.push_back(core_set);
    }

    std::vector<CoreRangeSet> core_allocations;
    for (const auto& core_set : core_sets) {
        if (core_set.empty()) {
            continue;  // Skip empty core sets
        }
        core_allocations.push_back(core_set.to_core_range_set());
    }

    std::cout << "Number of tenants: " << core_allocations.size() << "\n";
    // for (const auto& core_range_set : core_allocations) {
    //     std::cout << "Core Range Set: " << core_range_set.str() << "\n";
    // }

    bool pass = true;

    try {
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        // Matrix multiplication in concern: (M x K) x (K x N) = (M x N)
        // Use DIM_LEN for matrix dimensions
        // Create source data with specified matrix dimensions
        const uint32_t M = DIM_LEN;  // Number of rows in matrix A (user-defined)
        const uint32_t N = DIM_LEN;  // Number of columns in matrix B (user-defined)
        const uint32_t K = DIM_LEN;  // Inner dimension for multiplication (user-defined)

        // Ensure that the matrix dimensions are compatible with the tile size
        TT_ASSERT(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
        TT_ASSERT(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
        TT_ASSERT(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

        // Calculate matrix dimensions in tiles for the accelerator
        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Nt = N / TILE_WIDTH;

        // Calculate buffer sizes needed for each matrix in bytes
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2 * 32 * 32 = 2048 bytes
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt;                           // num_tiles of FP16_B

        // Create random input vectors for matrices A and B
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> src0_vec(M * K, 0);  // Matrix A (MxK)
        std::vector<bfloat16> src1_vec(K * N, 0);  // Matrix B (KxN)
        // // Fill with random bfloat16 values
        for (bfloat16& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (bfloat16& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        // Golden Matmul running on CPU (Float) - reference implementation for verification
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

        // Input vector tilizing to match device expected tiled layout
        // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
        // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the
        // device. This transformation groups elements into 32x32 blocks and reorders them in memory so that each
        // tile (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix
        // engine and enables efficient operations on the accelerator.
        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        // the values in tensors are not checked for brevity
        /* Calling the MatMul host program. Read in result into a host vector */
        std::vector<bfloat16> result_vec(dram_buffer_C_size / sizeof(bfloat16));
        matmul_multi_core(src0_vec, src1_vec, result_vec, M, N, K, device, outfile, core_allocations);
        // Reverse the tilization to get the result in the row-major format that the CPU expects
        result_vec = untilize_nfaces(result_vec, M, N);

        fmt::print("Output vector of size {}\n", result_vec.size());

        // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
        // This is a measure of how similar the two vectors are.
        // A PCC close to 1 indicates that the two vectors are very similar.
        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
        TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());

        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
