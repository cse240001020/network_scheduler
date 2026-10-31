#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Flow {
    std::uint64_t id{};
    double arrival_ms{};
    std::uint64_t size_bytes{};
};

struct Completion {
    double finish_ms{};
    std::uint64_t size_bytes{};
};

struct Path {
    double capacity_bytes_per_ms{};
    double available_at_ms{};
    std::deque<Completion> pending;
    std::uint64_t pending_bytes{};
};

struct Result {
    Flow flow;
    int path_id{-1};
    bool dropped{false};
    double queue_delay_ms{};
    double service_ms{};
    double finish_ms{};
};

struct Options {
    std::string policy{"least_delay"};
    std::string output{"flows.csv"};
    std::uint64_t flow_count{20000};
    std::uint64_t seed{42};
    double arrival_rate_per_second{500.0};
    std::uint64_t queue_limit_bytes{64ULL * 1024ULL * 1024ULL};
};

constexpr double kBytesPerMegabyte = 1'000'000.0;
const std::vector<double> kPathCapacitiesMbPerSecond{55.0, 35.0, 20.0};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --policy NAME       round_robin | shortest_queue | least_delay\n"
              << "  --flows N           number of generated flows (default: 20000)\n"
              << "  --seed N            workload seed (default: 42)\n"
              << "  --arrival-rate N    average flows per second (default: 500)\n"
              << "  --queue-limit-mb N  per-path waiting-buffer limit (default: 64)\n"
              << "  --output PATH       output CSV (default: flows.csv)\n";
}

std::uint64_t parse_positive_integer(const std::string& value, const std::string& flag) {
    std::size_t parsed = 0;
    const auto result = std::stoull(value, &parsed);
    if (parsed != value.size() || result == 0) {
        throw std::invalid_argument(flag + " expects a positive integer");
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value after " + flag);
        }
        const std::string value = argv[++i];
        if (flag == "--policy") {
            options.policy = value;
        } else if (flag == "--output") {
            options.output = value;
        } else if (flag == "--flows") {
            options.flow_count = parse_positive_integer(value, flag);
        } else if (flag == "--seed") {
            options.seed = parse_positive_integer(value, flag);
        } else if (flag == "--arrival-rate") {
            options.arrival_rate_per_second = std::stod(value);
            if (!(options.arrival_rate_per_second > 0.0)) {
                throw std::invalid_argument(flag + " expects a value greater than zero");
            }
        } else if (flag == "--queue-limit-mb") {
            options.queue_limit_bytes = parse_positive_integer(value, flag) * 1'000'000ULL;
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }
    if (options.policy != "round_robin" && options.policy != "shortest_queue" &&
        options.policy != "least_delay") {
        throw std::invalid_argument("unknown policy: " + options.policy);
    }
    return options;
}

std::vector<Flow> generate_workload(const Options& options) {
    std::mt19937_64 generator(options.seed);
    std::exponential_distribution<double> interarrival(options.arrival_rate_per_second / 1000.0);
    // Parameterize the log-normal distribution to have an approximate 64 KB mean.
    std::lognormal_distribution<double> flow_size(std::log(64'000.0) - 0.5 * 0.9 * 0.9, 0.9);

    std::vector<Flow> flows;
    flows.reserve(static_cast<std::size_t>(options.flow_count));
    double arrival_ms = 0.0;
    for (std::uint64_t id = 0; id < options.flow_count; ++id) {
        arrival_ms += interarrival(generator);
        const auto sampled_size = static_cast<std::uint64_t>(flow_size(generator));
        const auto bounded_size = std::clamp<std::uint64_t>(sampled_size, 512, 4'000'000);
        flows.push_back(Flow{id, arrival_ms, bounded_size});
    }
    return flows;
}

void retire_completed(Path& path, double now_ms) {
    while (!path.pending.empty() && path.pending.front().finish_ms <= now_ms) {
        path.pending_bytes -= path.pending.front().size_bytes;
        path.pending.pop_front();
    }
}

std::size_t choose_path(const std::string& policy, const Flow& flow, std::vector<Path>& paths,
                        std::size_t& round_robin_cursor) {
    if (policy == "round_robin") {
        const auto selected = round_robin_cursor;
        round_robin_cursor = (round_robin_cursor + 1) % paths.size();
        return selected;
    }

    std::size_t selected = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const auto& path = paths[i];
        double score = 0.0;
        if (policy == "shortest_queue") {
            score = static_cast<double>(path.pending.size());
        } else {
            const double start_ms = std::max(flow.arrival_ms, path.available_at_ms);
            const double service_ms = static_cast<double>(flow.size_bytes) / path.capacity_bytes_per_ms;
            score = (start_ms - flow.arrival_ms) + service_ms;
        }
        if (score < best_score) {
            best_score = score;
            selected = i;
        }
    }
    return selected;
}

std::vector<Result> simulate(const Options& options, const std::vector<Flow>& flows) {
    std::vector<Path> paths;
    for (const auto capacity_mb_per_second : kPathCapacitiesMbPerSecond) {
        // 1 MB/s is 1000 bytes/ms when MB is decimal.
        paths.push_back(Path{capacity_mb_per_second * 1000.0, 0.0, {}, 0});
    }

    std::vector<Result> results;
    results.reserve(flows.size());
    std::size_t round_robin_cursor = 0;
    for (const auto& flow : flows) {
        for (auto& path : paths) {
            retire_completed(path, flow.arrival_ms);
        }

        const auto path_index = choose_path(options.policy, flow, paths, round_robin_cursor);
        auto& path = paths[path_index];
        const double queued_ms = std::max(0.0, path.available_at_ms - flow.arrival_ms);
        const auto queued_bytes = static_cast<std::uint64_t>(queued_ms * path.capacity_bytes_per_ms);
        const bool exceeds_buffer = queued_bytes + flow.size_bytes > options.queue_limit_bytes;

        Result result;
        result.flow = flow;
        result.path_id = static_cast<int>(path_index);
        result.dropped = exceeds_buffer;
        if (!exceeds_buffer) {
            const double start_ms = std::max(flow.arrival_ms, path.available_at_ms);
            result.queue_delay_ms = start_ms - flow.arrival_ms;
            result.service_ms = static_cast<double>(flow.size_bytes) / path.capacity_bytes_per_ms;
            result.finish_ms = start_ms + result.service_ms;

            path.available_at_ms = result.finish_ms;
            path.pending.push_back(Completion{result.finish_ms, flow.size_bytes});
            path.pending_bytes += flow.size_bytes;
        }
        results.push_back(result);
    }
    return results;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    const auto index = static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

void write_csv(const std::string& output_path, const std::string& policy,
               const std::vector<Result>& results) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("could not open output file: " + output_path);
    }
    output << "flow_id,arrival_ms,size_bytes,policy,path_id,dropped,queue_delay_ms,service_ms,finish_ms,latency_ms\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        const double latency_ms = result.dropped ? 0.0 : result.queue_delay_ms + result.service_ms;
        output << result.flow.id << ',' << result.flow.arrival_ms << ',' << result.flow.size_bytes << ','
               << policy << ',' << result.path_id << ','
               << (result.dropped ? 1 : 0) << ',' << result.queue_delay_ms << ',' << result.service_ms << ','
               << result.finish_ms << ',' << latency_ms << '\n';
    }
}

void print_summary(const Options& options, const std::vector<Result>& results) {
    std::vector<double> latencies;
    std::uint64_t admitted = 0;
    std::uint64_t dropped = 0;
    std::uint64_t admitted_bytes = 0;
    const double start_ms = results.empty() ? 0.0 : results.front().flow.arrival_ms;
    double end_ms = 0.0;
    for (const auto& result : results) {
        if (result.dropped) {
            ++dropped;
            continue;
        }
        ++admitted;
        admitted_bytes += result.flow.size_bytes;
        end_ms = std::max(end_ms, result.finish_ms);
        latencies.push_back(result.queue_delay_ms + result.service_ms);
    }
    const double mean_latency = latencies.empty()
                                    ? 0.0
                                    : std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                                          static_cast<double>(latencies.size());
    const double duration_ms = end_ms - start_ms;
    const double throughput_mbps = duration_ms > 0.0
                                       ? (static_cast<double>(admitted_bytes) / (duration_ms / 1000.0)) /
                                             kBytesPerMegabyte
                                       : 0.0;

    std::cout << std::fixed << std::setprecision(3)
              << "policy: " << options.policy << '\n'
              << "flows_offered: " << results.size() << '\n'
              << "flows_admitted: " << admitted << '\n'
              << "flows_dropped: " << dropped << '\n'
              << "drop_rate_percent: "
              << (results.empty() ? 0.0 : 100.0 * static_cast<double>(dropped) / results.size()) << '\n'
              << "mean_latency_ms: " << mean_latency << '\n'
              << "p95_latency_ms: " << percentile(latencies, 0.95) << '\n'
              << "throughput_MB_per_s: " << throughput_mbps << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto flows = generate_workload(options);
        const auto results = simulate(options, flows);
        write_csv(options.output, options.policy, results);
        print_summary(options, results);
        std::cout << "csv: " << options.output << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << "Run with --help to see available options.\n";
        return 1;
    }
    return 0;
}
