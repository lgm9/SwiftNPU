#define ALGORITHM_TEST
#include "core.hpp"
#include <x86intrin.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <filesystem>

using namespace std;

int grid_x = 5, grid_y = 5;
vector<core_set_t> V;
vector<uint64_t> E;
char map[100][100];
core_set_t placed;

static uint64_t measure_full_allocation_GED(int n) {
    grid_x = n;
    grid_y = n;
    if (grid_x == 2) {
        grid_x = grid_y = 3;
    }

    core_grid_t main_grid(grid_x, grid_y);
    unsigned int aux = 0;
    uint64_t sum = 0;

    V.clear();
    E.clear();

    while (true) {
        uint64_t t0 = __rdtscp(&aux);
        placed = main_grid.allocate_GED(3, 3);
        uint64_t t1 = __rdtscp(&aux);

        uint64_t dt = t1 - t0;
        E.push_back(dt);
        sum += dt;

        if (placed.empty()) {
            break;
        }
        V.push_back(placed);
    }

    return sum;
}

static uint64_t measure_full_allocation_NAS(int n) {
    grid_x = n;
    grid_y = n;
    if (grid_x == 2) {
        grid_x = grid_y = 3;
    }

    core_grid_t main_grid(grid_x, grid_y);
    unsigned int aux = 0;
    uint64_t sum = 0;

    V.clear();
    E.clear();

    while (true) {
        uint64_t t0 = __rdtscp(&aux);
        placed = main_grid.allocate_NAS(3, 3);
        uint64_t t1 = __rdtscp(&aux);

        uint64_t dt = t1 - t0;
        E.push_back(dt);
        sum += dt;

        if (placed.empty()) {
            break;
        }
        V.push_back(placed);
    }

    return sum;
}

int main(int argc, char* argv[]) {
    std::filesystem::create_directories("results");
    const char* ged_path = "results/vNPU.txt";
    const char* nas_path = "results/NAS.txt";

    FILE* ged_fp = fopen(ged_path, "w");
    FILE* nas_fp = fopen(nas_path, "w");

    if (!ged_fp || !nas_fp) {
        fprintf(stderr, "Failed to open output files: %s, %s\n", ged_path, nas_path);
        if (ged_fp) fclose(ged_fp);
        if (nas_fp) fclose(nas_fp);
        return 1;
    }

    fprintf(ged_fp, "# n grid_size total_cycles\n");
    fprintf(nas_fp, "# n grid_size total_cycles\n");

    for (int n = 2; n <= 11; n++) {
        uint64_t ged_sum = measure_full_allocation_GED(n);
        uint64_t nas_sum = measure_full_allocation_NAS(n);

        if (n >= 3) {
            fprintf(ged_fp, "%d %d %llu\n", n, n, static_cast<unsigned long long>(ged_sum));
            fprintf(nas_fp, "%d %d %llu\n", n, n, static_cast<unsigned long long>(nas_sum));
            printf("n=%d GED=%llu NAS=%llu\n",
                   n,
                   static_cast<unsigned long long>(ged_sum),
                   static_cast<unsigned long long>(nas_sum));
        }
    }

    fclose(ged_fp);
    fclose(nas_fp);
    return 0;
}
