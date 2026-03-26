#pragma once

#include <cstdio>
#include <vector>
#include <set>
#include <deque>
#include <cstdint>
#include <limits>
#include <queue>
#include <functional>
#include <string>
#include <algorithm>

#ifndef ALGORITHM_TEST
#include <fmt/core.h>
#endif

struct core_t {
public:
    int x;
    int y;

    core_t() : x(0), y(0) {}
    core_t(int x, int y) : x(x), y(y) {}

    bool operator<(const core_t& o) const { return (x < o.x) || (x == o.x && y < o.y); }
};

struct core_set_t {
public:
    std::set<core_t> cores;

    core_set_t() : cores() {}

    void core_add(core_t in_core) { cores.insert(in_core); }
    void core_add(int x, int y) { cores.insert(core_t(x, y)); }
    void core_add_rect(int x, int y, int a, int b) {
        for(int i = 0 ; i < a ; i++) for(int j = 0 ; j < b ; j++) {
            core_add(x + i, y + j);
        }
    }

    void core_pop(core_t in_core) { cores.erase(in_core); }
    void core_pop(int x, int y) { cores.erase(core_t(x, y)); }

    bool contains(core_t in_core) const { return (cores.find(in_core) != cores.end()); }
    bool contains(int x, int y)   const { return (cores.find(core_t(x, y)) != cores.end()); }

    core_t get_front() {
        if (cores.empty()) {
            printf("[get_front] empty set\n");
            return core_t();
        }
        return *cores.begin();
    }

    core_t get_front_and_remove() {
        if (cores.empty()) {
            printf("[get_front_and_remove] empty set\n");
            return core_t();
        }
        auto it = cores.begin();
        auto tmp = *it;
        cores.erase(it);
        return tmp;
    }

    bool empty() const { return cores.empty(); }

    #ifndef ALGORITHM_TEST
    CoreRangeSet to_core_range_set() const {
        std::vector<CoreRange> core_ranges;
        for (const auto& core : cores) {
            CoreRange core_range(CoreCoord(core.x, core.y));
            core_ranges.push_back(core_range);
        }
        CoreRangeSet ranges(core_ranges);
        ranges.merge_ranges();
        return ranges;
    }
    #endif
};

struct bf_choice_t {
    bool found = false;
    int score = -1;
    int x0 = -1, y0 = -1;
    int a = 0, b = 0;
};

struct core_grid_t {
    std::vector<std::vector<bool>> grid;
    int xlen;
    int ylen;

    core_grid_t(int x, int y) : xlen(x), ylen(y) {
        grid.reserve(xlen);
        for (int i = 0; i < xlen; i++) {
            grid.push_back(std::vector<bool>(ylen, false));
        }
    }

    bool in_bounds(int x, int y) const { return (0 <= x && x < xlen && 0 <= y && y < ylen); }

    bool rect_in_bounds(int x, int y, int a, int b) const {
        return (a > 0 && b > 0 && x >= 0 && y >= 0 && x + a <= xlen && y + b <= ylen);
    }

    void oob_error(const char* fn, int x, int y) const {
        printf("[%s] Out-of-bounds: (%d,%d) not in [0,%d) x [0,%d)\n", fn, x, y, xlen, ylen);
    }

    bool get(int x, int y) const {
        if (!in_bounds(x, y)) {
            oob_error("get", x, y);
            return false;
        }
        return grid[x][y];
    }

    void allocate(int x, int y) {
        if (!in_bounds(x, y)) {
            oob_error("allocate", x, y);
            return;
        }
        if (grid[x][y]) {
            printf("[allocate] Already allocated: (%d,%d)\n", x, y);
            return;
        }
        grid[x][y] = true;
    }

    void allocate(core_t in_core) { allocate(in_core.x, in_core.y); }

    void allocate(const core_set_t& in_set) {
        for (const auto& c : in_set.cores) {
            if (!in_bounds(c.x, c.y)) {
                oob_error("allocate(set)", c.x, c.y);
                return;
            }
            if (grid[c.x][c.y]) {
                printf("[allocate(set)] Already allocated: (%d,%d)\n", c.x, c.y);
                return;
            }
        }
        for (const auto& c : in_set.cores) {
            grid[c.x][c.y] = true;
        }
    }

    bool check_rect(int x, int y, int a, int b) const {
        if (!rect_in_bounds(x, y, a, b)) return false;
        for (int i = 0; i < a; ++i)
            for (int j = 0; j < b; ++j)
                if (grid[x + i][y + j]) return false;
        return true;
    }

    bool allocate_rect(int x, int y, int a, int b) {
        if (!check_rect(x, y, a, b)) return false;
        for (int i = 0; i < a; i++) for (int j = 0; j < b; j++) {
            grid[x + i][y + j] = true;
        }
        return true;
    }

    bool allocate_rect(int x, int y, int a, int b, core_set_t& placed) {
        if(!allocate_rect(x, y, a, b)) return false;
        placed.core_add_rect(x, y, a, b);
        return true;
    }

    void free(int x, int y) {
        if (!in_bounds(x, y)) {
            oob_error("free", x, y);
            return;
        }
        if (!grid[x][y]) {
            printf("[free] Double free / not allocated: (%d,%d)\n", x, y);
            return;
        }
        grid[x][y] = false;
    }

    void free(core_t in_core) { free(in_core.x, in_core.y); }

    void free(const core_set_t& in_set) {
        for (const auto& c : in_set.cores) {
            if (!in_bounds(c.x, c.y)) {
                oob_error("free(set)", c.x, c.y);
                return;
            }
            if (!grid[c.x][c.y]) {
                printf("[free(set)] Not allocated: (%d,%d)\n", c.x, c.y);
                return;
            }
        }
        for (const auto& c : in_set.cores) {
            grid[c.x][c.y] = false;
        }
    }

    //Silent free core_set_t. ONLY USE WHEN THE CORE SET IS ALLOCATED RECENTLY
    void rollback(const core_set_t& placed) {
        for (const auto& c : placed.cores)
            grid[c.x][c.y] = false;
    }

    void free_all() {
        for (auto& row : grid) {
            std::fill(row.begin(), row.end(), false);
        }
    }

    int count_free() const {
        int cnt = 0;
        for (int i = 0; i < xlen; ++i)
            for (int j = 0; j < ylen; ++j)
                if (!grid[i][j]) cnt++;
        return cnt;
    }

    core_set_t allocate_first_fit(int a, int b) {
        core_set_t placed;
        if (a <= 0 || b <= 0 || a > xlen || b > ylen) {
            return placed;
        }

        std::vector<int> heights(ylen, 0);
        for (int x = 0; x < xlen; ++x) {
            for (int y = 0; y < ylen; ++y) {
                heights[y] = grid[x][y] ? 0 : heights[y] + 1;
            }

            std::deque<int> dq;
            for (int y = 0; y < ylen; ++y) {
                while (!dq.empty() && heights[y] <= heights[dq.back()]) {
                    dq.pop_back();
                }
                dq.push_back(y);
                if (!dq.empty() && dq.front() <= y - b) {
                    dq.pop_front();
                }
                if (y >= b - 1) {
                    int minH = heights[dq.front()];
                    if (minH >= a) {
                        int top_x = x - a + 1;
                        int left_y = y - b + 1;

                        for (int dx = 0; dx < a; ++dx) {
                            for (int dy = 0; dy < b; ++dy) {
                                grid[top_x + dx][left_y + dy] = true;
                                placed.core_add(top_x + dx, left_y + dy);
                            }
                        }

                        return placed;
                    }
                }
            }
        }

        return placed;
    }

    bf_choice_t probe_best_fit(int a, int b, bool walls_as_busy = true) const {
        bf_choice_t best{false, -1, -1, -1, a, b};
        if (a <= 0 || b <= 0 || a > xlen || b > ylen) return best;

        const int XP = xlen + 2, YP = ylen + 2;
        std::vector<std::vector<int>> occ(XP, std::vector<int>(YP, walls_as_busy ? 1 : 0));
        for (int x = 0; x < xlen; ++x)
            for (int y = 0; y < ylen; ++y)
                occ[x + 1][y + 1] = grid[x][y] ? 1 : 0;

        // prefix sum
        std::vector<std::vector<int>> ps(XP + 1, std::vector<int>(YP + 1, 0));
        for (int i = 0; i < XP; ++i)
            for (int j = 0; j < YP; ++j)
                ps[i + 1][j + 1] = ps[i + 1][j] + ps[i][j + 1] - ps[i][j] + occ[i][j];

        auto sumRect = [&](int x1, int y1, int x2, int y2) -> int {
            if (x1 > x2 || y1 > y2) return 0;
            return ps[x2 + 1][y2 + 1] - ps[x1][y2 + 1] - ps[x2 + 1][y1] + ps[x1][y1];
        };

        // vertical free runs (you called it horiz_ok)
        std::vector<std::vector<uint8_t>> vert_ok(xlen, std::vector<uint8_t>(ylen, 0));
        for (int y = 0; y < ylen; ++y) {
            int run = 0;
            for (int x = xlen - 1; x >= 0; --x) {
                run = (!grid[x][y]) ? run + 1 : 0;
                vert_ok[x][y] = (run >= a);
            }
        }

        for (int x0 = 0; x0 <= xlen - a; ++x0) {
            int down_run = 0;
            for (int y = 0; y < ylen; ++y) {
                if (vert_ok[x0][y]) {
                    if (++down_run >= b) {
                        int y0 = y - b + 1;

                        int xp0 = x0 + 1, yp0 = y0 + 1;
                        int top    = sumRect(xp0,       yp0 - 1, xp0 + a - 1, yp0 - 1);
                        int bottom = sumRect(xp0,       yp0 + b, xp0 + a - 1, yp0 + b);
                        int left   = sumRect(xp0 - 1,   yp0,     xp0 - 1,     yp0 + b - 1);
                        int right  = sumRect(xp0 + a,   yp0,     xp0 + a,     yp0 + b - 1);
                        int score  = top + bottom + left + right;

                        if (!best.found ||
                            score > best.score ||
                            (score == best.score && (y0 < best.y0 ||
                                                    (y0 == best.y0 && x0 < best.x0)))) {
                            best.found = true;
                            best.score = score;
                            best.x0 = x0;
                            best.y0 = y0;
                            // a,b already set
                        }
                    }
                } else {
                    down_run = 0;
                }
            }
        }

        return best;
    }

    // Walls_as_busy will consider the wall as a busy core
    core_set_t allocate_best_fit(int a, int b, bool walls_as_busy = true) {
        core_set_t placed;
        auto best = probe_best_fit(a, b, walls_as_busy);
        if (!best.found) return placed;

        for (int dx = 0; dx < best.a; ++dx)
            for (int dy = 0; dy < best.b; ++dy) {
                grid[best.x0 + dx][best.y0 + dy] = true;
                placed.core_add(best.x0 + dx, best.y0 + dy);
            }

        return placed;
    }
    // --------- LSSA helpers (inside core_grid_t) ---------

    // Try to attach SR(e,f) to an already placed SL(c,d) located at (sx, sy).
    // SL occupies rows [sx .. sx+c-1], cols [sy .. sy+d-1].
    // We try up to 8 natural "corner-aligned" attachments around SL.
    // On success, we allocate SR into 'placed' and return true.
    // On failure, 'placed' is unchanged.
    bool attach_SR_around_SL(int sx, int sy, int c, int d, int e, int f, core_set_t& placed) {
        // candidate: (x,y) is top-left of SR (e rows, f cols)
        auto try_pos = [&](int x, int y) -> bool {
            core_set_t tmp;
            if (!check_rect(x, y, e, f)) return false;
            if (!allocate_rect(x, y, e, f, tmp)) return false;
            // merge tmp into placed
            for (const auto& cell : tmp.cores) placed.core_add(cell);
            return true;
        };

        // RIGHT side (aligned to TOP / BOTTOM)
        if (try_pos(sx,             sy + d)) return true;            // right-top
        if (try_pos(sx + c - e,     sy + d)) return true;            // right-bottom

        // LEFT side
        if (try_pos(sx,             sy - f)) return true;            // left-top
        if (try_pos(sx + c - e,     sy - f)) return true;            // left-bottom

        // TOP side
        if (try_pos(sx - e,         sy))     return true;            // top-left
        if (try_pos(sx - e,         sy + d - f)) return true;        // top-right

        // BOTTOM side
        if (try_pos(sx + c,         sy))     return true;            // bottom-left
        if (try_pos(sx + c,         sy + d - f)) return true;        // bottom-right

        return false;
    }

    // Place an L by first placing SL(c,d) via FF scan, then attaching SR(e,f) around it.
    // Returns the union set; empty on failure.
    core_set_t place_L_FF(int c, int d, int e, int f) {
        core_set_t placed;
        if (c <= 0 || d <= 0 || e <= 0 || f <= 0) return placed;
        if (c > xlen || d > ylen || e > xlen || f > ylen) return placed;

        for (int sx = 0; sx + c <= xlen; ++sx) {
            for (int sy = 0; sy + d <= ylen; ++sy) {
                if (!check_rect(sx, sy, c, d)) continue;

                core_set_t tmp;  // tentative SL
                if (!allocate_rect(sx, sy, c, d, tmp)) continue;

                // try to attach SR around this SL
                if (attach_SR_around_SL(sx, sy, c, d, e, f, tmp)) {
                    // success: commit
                    return tmp;
                } else {
                    // rollback SL only
                    rollback(tmp);
                }
            }
        }
        return core_set_t{};
    }

    core_set_t try_L_even_a(int a, int b) {
        core_set_t placed;
        int c = a / 2, e = a / 2;
        for (int f = 1; f <= b; ++f) {
            int d = 2 * b - f;
            if (d <= 0) continue;
            placed = place_L_FF(c, d, e, f);
            if (!placed.empty()) return placed;
        }
        return placed;
    }

    core_set_t try_L_odd_a(int a, int b) {
        core_set_t placed;
        int a_up = (a + 1) / 2;  // ceil
        int a_dn = a / 2;        // floor
        int k_max = b - 1 - a_up;
        for (int k = 0; k <= k_max; ++k) {
            int c = a_up + k;
            int d = b + a_dn - k;
            int e = a_dn - k;
            int f = b - a_up + k;
            if (c <= 0 || d <= 0 || e <= 0 || f <= 0) continue;
            placed = place_L_FF(c, d, e, f);
            if (!placed.empty()) return placed;
        }
        return placed;
    }

    core_set_t try_L_from_b(int a, int b) {
        core_set_t placed;
        if (b % 2 == 0) {
            int c = b / 2, e = b / 2;
            for (int f = 1; f <= a; ++f) {
                int d = 2 * a - f;
                if (d <= 0) continue;
                // transpose
                placed = place_L_FF(d, c, f, e);
                if (!placed.empty()) return placed;
            }
        } else {
            int b_up = (b + 1) / 2, b_dn = b / 2;
            int k_max = a - 1 - b_up;
            for (int k = 0; k <= k_max; ++k) {
                int c = b_up + k;
                int d = a + b_dn - k;
                int e = b_dn - k;
                int f = a - b_up + k;
                if (c <= 0 || d <= 0 || e <= 0 || f <= 0) continue;
                // transpose
                placed = place_L_FF(d, c, f, e);
                if (!placed.empty()) return placed;
            }
        }
        return placed;
    }

    core_set_t try_folds(int a, int b) {
        core_set_t placed;
        if (a % 2 == 0) {
            placed = allocate_first_fit(a / 2, 2 * b);
            if (!placed.empty()) return placed;
        }
        if (b % 2 == 0) {
            placed = allocate_first_fit(2 * a, b / 2);
            if (!placed.empty()) return placed;
        }
        return placed;
    }

    // --------- top-level: LSSA First-Fit ---------
    core_set_t allocate_LSSA(int a, int b) {
        core_set_t placed;
        if (a <= 0 || b <= 0) return placed;
        if (a > xlen && b > xlen) return placed;       // quick sanity
        if (b > ylen && a > ylen) return placed;
        if (count_free() < a * b) return placed;       // capacity check

        // 1) exact a×b
        placed = allocate_first_fit(a, b);
        if (!placed.empty()) return placed;

        // 2) rotation b×a
        placed = allocate_first_fit(b, a);
        if (!placed.empty()) return placed;

        // 3) folds
        placed = try_folds(a, b);
        if (!placed.empty()) return placed;

        // 4) L-shapes derived from side a
        placed = (a % 2 == 0) ? try_L_even_a(a, b) : try_L_odd_a(a, b);
        if (!placed.empty()) return placed;

        // 5) L-shapes derived from side b (transpose families)
        placed = try_L_from_b(a, b);
        if (!placed.empty()) return placed;

        // fail
        return core_set_t{};
    }

    //All shape FF
    core_set_t allocate_ASFF(int n) {
        core_set_t placed;
        
        int a = 2;
        while(1) {
            if(a * a > n) {
                a--;
                break;
            }
            a++;
        }
        while(a) {
            if(n % a == 0) placed = allocate_first_fit(a, n / a);
            if(!placed.empty()) return placed;
            if(n % a == 0) placed = allocate_first_fit(n / a, a);
            if(!placed.empty()) return placed;
            a--;
        }

        return placed;
    }

    core_set_t allocate_ASBF(int n, bool walls_as_busy = true) {
        core_set_t placed;
        if (n <= 0) return placed;
        if (count_free() < n) return placed;

        bf_choice_t best_overall{};  // now safely initialized

        for (int a = 1; a * a <= n; ++a) {
            if (n % a) continue;
            int b = n / a;

            if (a <= xlen && b <= ylen) {
                auto c1 = probe_best_fit(a, b, walls_as_busy);
                if (c1.found &&
                    (!best_overall.found ||
                    c1.score > best_overall.score ||
                    (c1.score == best_overall.score &&
                    (c1.y0 < best_overall.y0 ||
                    (c1.y0 == best_overall.y0 && c1.x0 < best_overall.x0))))) {
                    best_overall = c1;
                }
            }

            if (a != b && b <= xlen && a <= ylen) {
                auto c2 = probe_best_fit(b, a, walls_as_busy);
                if (c2.found &&
                    (!best_overall.found ||
                    c2.score > best_overall.score ||
                    (c2.score == best_overall.score &&
                    (c2.y0 < best_overall.y0 ||
                    (c2.y0 == best_overall.y0 && c2.x0 < best_overall.x0))))) {
                    best_overall = c2;
                }
            }
        }

        if (!best_overall.found) return placed;

        if (allocate_rect(best_overall.x0, best_overall.y0,
                        best_overall.a, best_overall.b, placed)) {
            return placed;
        }

        return core_set_t{};
    }

     int free_neighbors(int x, int y) const {
        static const int dx[4] = {1, -1, 0, 0};
        static const int dy[4] = {0, 0, 1, -1};
        int cnt = 0;
        for (int k = 0; k < 4; ++k) {
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (!in_bounds(nx, ny)) continue;
            if (!grid[nx][ny]) cnt++;
        }
        return cnt;
    }

    core_set_t allocate_GED(int a, int b, bool require_connected = true) {
        core_set_t placed;
        if (a <= 0 || b <= 0) return placed;

        const int k = a * b;
        const int free_total = count_free();
        if (free_total < k) return placed;

        // 1) Try FF on requested shape
        placed = allocate_first_fit(a, b);
        if (!placed.empty()) return placed;

        // 1) Collect all free cores and give them contiguous IDs [0..F-1]
        std::vector<core_t> free_nodes;
        free_nodes.reserve(free_total);
        for (int x = 0; x < xlen; ++x)
            for (int y = 0; y < ylen; ++y)
                if (!grid[x][y])
                    free_nodes.push_back(core_t(x, y));

        const int F = (int)free_nodes.size();

        // 2) Build adjacency matrix for free nodes (4-neighbor mesh)
        std::vector<std::vector<uint8_t>> adjFree(F, std::vector<uint8_t>(F, 0));
        for (int i = 0; i < F; ++i) {
            int x = free_nodes[i].x;
            int y = free_nodes[i].y;
            const int dx[4] = {-1, 1, 0, 0};
            const int dy[4] = {0, 0, -1, 1};
            for (int d = 0; d < 4; ++d) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (!in_bounds(nx, ny)) continue;
                if (grid[nx][ny]) continue; // only free neighbors
                // find its ID in free_nodes: we can do a small linear scan here
                // (F is small in the intended NP use cases). If you want faster,
                // build a map x,y->id above.
                for (int j = 0; j < F; ++j) {
                    if (free_nodes[j].x == nx && free_nodes[j].y == ny) {
                        adjFree[i][j] = adjFree[j][i] = 1;
                        break;
                    }
                }
            }
        }

        // 3) Build virtual a×b mesh adjacency (k nodes)
        std::vector<std::vector<uint8_t>> adjV(k, std::vector<uint8_t>(k, 0));
        auto vid = [b](int i, int j) { return i * b + j; }; // row-major
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                int v = vid(i, j);
                if (i + 1 < a) {
                    int v2 = vid(i + 1, j);
                    adjV[v][v2] = adjV[v2][v] = 1;
                }
                if (j + 1 < b) {
                    int v2 = vid(i, j + 1);
                    adjV[v][v2] = adjV[v2][v] = 1;
                }
            }
        }

        // 4) Helper: check connectivity of a candidate subset (by indices into free_nodes)
        auto subset_connected = [&](const std::vector<int>& nodes) -> bool {
            if (nodes.empty()) return false;
            const int sz = (int)nodes.size();
            std::vector<uint8_t> visited(sz, 0);
            std::queue<int> q;
            visited[0] = 1;
            q.push(0);
            int seen = 1;
            while (!q.empty()) {
                int i = q.front(); q.pop();
                int u = nodes[i];
                for (int j = 0; j < sz; ++j) {
                    if (visited[j]) continue;
                    int v = nodes[j];
                    if (adjFree[u][v]) {
                        visited[j] = 1;
                        q.push(j);
                        ++seen;
                    }
                }
            }
            return (seen == sz);
        };

        // 5) Helper: compute minimum edge edit distance between virtual mesh (adjV)
        //    and candidate physical subgraph induced by 'nodes' (IDs into free_nodes).
        auto compute_min_edit_distance = [&](const std::vector<int>& nodes) -> int {
            const int kk = (int)nodes.size();
            // build adjacency for candidate subgraph
            std::vector<std::vector<uint8_t>> adjP(kk, std::vector<uint8_t>(kk, 0));
            for (int i = 0; i < kk; ++i) {
                int u = nodes[i];
                for (int j = i + 1; j < kk; ++j) {
                    int v = nodes[j];
                    if (adjFree[u][v]) {
                        adjP[i][j] = adjP[j][i] = 1;
                    }
                }
            }

            int best = std::numeric_limits<int>::max();
            std::vector<int> perm(kk, -1);      // virtual idx -> physical idx
            std::vector<uint8_t> used(kk, 0);

            std::function<void(int,int)> dfs_perm = [&](int vIdx, int curCost) {
                if (curCost >= best) return;
                if (vIdx == kk) {
                    if (curCost < best) best = curCost;
                    return;
                }
                for (int pIdx = 0; pIdx < kk; ++pIdx) {
                    if (used[pIdx]) continue;
                    int delta = 0;
                    // cost contribution for edges between vIdx and earlier virtual nodes u < vIdx
                    for (int uIdx = 0; uIdx < vIdx; ++uIdx) {
                        int pu = perm[uIdx];
                        if (pu < 0) continue;
                        uint8_t ev = adjV[vIdx][uIdx];
                        uint8_t ep = adjP[pIdx][pu];
                        if (ev != ep) ++delta;
                    }
                    used[pIdx] = 1;
                    perm[vIdx] = pIdx;
                    dfs_perm(vIdx + 1, curCost + delta);
                    used[pIdx] = 0;
                    perm[vIdx] = -1;
                }
            };

            dfs_perm(0, 0);
            return best;
        };

        // 6) Enumerate all subsets of free_nodes of size k (combinations), bruteforce.
        //    For each subset:
        //      - if require_connected: check connectivity
        //      - compute minimal edit distance and keep the best
        std::vector<int> current;
        current.reserve(k);

        int bestCost = std::numeric_limits<int>::max();
        std::vector<int> bestChoice;

        std::function<void(int,int)> dfs_comb = [&](int start, int depth) {
            if (depth == k) {
                if (require_connected && !subset_connected(current)) return;
                int cost = compute_min_edit_distance(current);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestChoice = current;
                }
                return;
            }
            if (start >= F) return;
            // Not enough remaining nodes to fill k
            if (F - start < k - depth) return;

            for (int i = start; i < F; ++i) {
                current.push_back(i);
                dfs_comb(i + 1, depth + 1);
                current.pop_back();
            }
        };

        dfs_comb(0, 0);

        // 7) If we found nothing, fail
        if (bestChoice.empty()) return placed;

        // 8) Actually allocate the chosen subset in the grid
        for (int idx : bestChoice) {
            const core_t& c = free_nodes[idx];
            grid[c.x][c.y] = true;
            placed.core_add(c.x, c.y);
        }
        return placed;
    }

    core_set_t allocate_NAS(int a, int b) {
        core_set_t placed;
        if (a <= 0 || b <= 0) return placed;

        const int job = a * b;
        if (count_free() < job) return placed; // paper: fail if not enough free PEs :contentReference[oaicite:6]{index=6}

        // 1) Try FF on requested shape
        placed = allocate_first_fit(a, b);
        if (!placed.empty()) return placed;

        // 2) Try ASFF on area (a*b)
        placed = allocate_ASFF(job);
        if (!placed.empty()) return placed;

        // 3) Try LSSA (your implementation already tries rotation/folds/L-shapes)
        placed = allocate_LSSA(a, b);
        if (!placed.empty()) return placed;

        // ---------- 4) Nucleus-based NAS ----------
        // Paper’s nucleus rules are described assuming a >= b >= 2 (they also mention odd-a family) :contentReference[oaicite:7]{index=7}
        // We normalize so A >= B (track if we swapped so we can transpose candidate nuclei).
        int A = a, B = b;
        bool swapped = false;
        if (A < B) { std::swap(A, B); swapped = true; }

        auto try_allocate_nucleus = [&](int c, int d) -> core_set_t {
            // map back if we swapped the request orientation
            int cc = c, dd = d;
            if (swapped) std::swap(cc, dd);
            // FF allocate nucleus rectangle
            return allocate_first_fit(cc, dd);
        };

        auto add_frontier_neighbors = [&](const core_set_t& cur, std::vector<core_t>& frontier, std::vector<std::vector<uint8_t>>& in_frontier) {
            static const int dx[4] = {1, -1, 0, 0};
            static const int dy[4] = {0, 0, 1, -1};
            for (const auto& p : cur.cores) {
                for (int k = 0; k < 4; ++k) {
                    int nx = p.x + dx[k], ny = p.y + dy[k];
                    if (!in_bounds(nx, ny)) continue;
                    if (grid[nx][ny]) continue;        // not free
                    if (in_frontier[nx][ny]) continue; // already queued
                    in_frontier[nx][ny] = 1;
                    frontier.push_back(core_t(nx, ny));
                }
            }
        };

        auto allocated_neighbors_in_job = [&](int x, int y, const core_set_t& cur) -> int {
            static const int dx[4] = {1, -1, 0, 0};
            static const int dy[4] = {0, 0, 1, -1};
            int cnt = 0;
            for (int k = 0; k < 4; ++k) {
                int nx = x + dx[k], ny = y + dy[k];
                if (!in_bounds(nx, ny)) continue;
                if (cur.contains(nx, ny)) ++cnt; // only count neighbors that belong to this job
            }
            return cnt;
        };

        // Generate nucleus candidates.
        // Even-a rule from paper: NS(a/2, b+k), 1<=k<=b :contentReference[oaicite:8]{index=8}
        // In practice (and consistent with their 5x3 -> 2x6 example), trying floor(A/2) works well. :contentReference[oaicite:9]{index=9}
        std::vector<std::pair<int,int>> nucleus_candidates;

        if (A >= 2 && B >= 2) {
            int c = A / 2; // floor(A/2)
            for (int k = 1; k <= B; ++k) {
                int d = B + k;
                if (c > 0 && d > 0) nucleus_candidates.push_back({c, d});
            }

            // Odd-a family (paper): NS depends on k in both dims :contentReference[oaicite:10]{index=10}
            // The exact typography is messy in the HTML extraction, so we add a conservative interpretation:
            // c = floor(A/2) + k, d = B + floor(A/2) - k (keep dims positive).
            if (A % 2 == 1) {
                int half = A / 2;
                for (int k = 1; k <= half; ++k) {
                    int c2 = half + k;
                    int d2 = B + half - k;
                    if (c2 > 0 && d2 > 0) nucleus_candidates.push_back({c2, d2});
                }
            }
        }

        // Try allocating nucleus in candidate order; once allocated, expand via neighbors to reach job size.
        for (auto [c, d] : nucleus_candidates) {
            core_set_t nucleus = try_allocate_nucleus(c, d);
            if (nucleus.empty()) continue;

            placed = nucleus;
            int remaining = job - (int)placed.cores.size();
            if (remaining <= 0) {
                // nucleus already covers >= job; keep it (paper allows internal fragmentation in non-contiguous strategies)
                return placed;
            }

            // Build frontier of free neighbor nodes (paper neighbor definition is 4-neighbor) :contentReference[oaicite:11]{index=11}
            std::vector<core_t> frontier;
            std::vector<std::vector<uint8_t>> in_frontier(xlen, std::vector<uint8_t>(ylen, 0));
            add_frontier_neighbors(placed, frontier, in_frontier);

            while (remaining > 0) {
                if (frontier.empty()) {
                    rollback(placed);
                    placed = core_set_t{};
                    break;
                }

                // pick the frontier node with maximum adjacency to current placed set (keeps cluster tight)
                int best_i = -1;
                int best_score = -1;
                for (int i = 0; i < (int)frontier.size(); ++i) {
                    const auto& cand = frontier[i];
                    if (grid[cand.x][cand.y]) continue; // might have become busy
                    int sc = allocated_neighbors_in_job(cand.x, cand.y, placed);
                    if (sc > best_score ||
                        (sc == best_score && (cand.x < frontier[best_i].x ||
                                            (cand.x == frontier[best_i].x && cand.y < frontier[best_i].y)))) {
                        best_score = sc;
                        best_i = i;
                    }
                }

                if (best_i < 0) {
                    // all candidates are stale/busy now
                    frontier.clear();
                    continue;
                }

                core_t pick = frontier[best_i];
                // remove pick from frontier
                frontier[best_i] = frontier.back();
                frontier.pop_back();

                if (grid[pick.x][pick.y]) continue;

                // allocate it
                grid[pick.x][pick.y] = true;
                placed.core_add(pick);
                --remaining;

                // add its free neighbors to frontier
                core_set_t just_added;
                just_added.core_add(pick);
                add_frontier_neighbors(just_added, frontier, in_frontier);
            }

            if (!placed.empty() && (int)placed.cores.size() == job) return placed;

            // if we broke due to failure, ensure any partial allocation is rolled back
            if (!placed.empty()) rollback(placed);
            placed = core_set_t{};
        }

        // If all nucleus candidates fail
        return core_set_t{};
    }
};