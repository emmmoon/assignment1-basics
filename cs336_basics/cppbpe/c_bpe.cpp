#include <nanobind/nanobind.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/tuple.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <set>

namespace nb = nanobind;

inline uint64_t pair_hash(int a, int b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

inline std::pair<int, int> pair_unhash(uint64_t pair) {
    int a = static_cast<int>(pair >> 32);
    int b = static_cast<int>(pair & 0xFFFFFFFF);
    return {a, b};
}

std::tuple<std::unordered_map<int, std::vector<uint8_t>>, std::vector<std::pair<int, int>>>
train_bpe_cpp(
    std::unordered_map<int, int> wordid_count,
    std::unordered_map<int, std::vector<int>> wordid_encoding,
    std::unordered_map<int, std::vector<uint8_t>> vocabulary,
    int size,
    int vocab_size
) {
    std::vector<std::pair<int, int>> merges;
    std::unordered_map<uint64_t, int> pair_counts;
    std::unordered_map<uint64_t, std::set<int>> pair_to_wordid;

    for (const auto& [wordid, count] : wordid_count) {
        const auto &encoding = wordid_encoding[wordid];
        for(int i = 0; i < encoding.size() - 1; ++i) {
            uint64_t pair = pair_hash(encoding[i], encoding[i + 1]);
            pair_counts[pair] += count;
            pair_to_wordid[pair].insert(wordid);
        }
    }

    while(size < vocab_size) {
        if (pair_counts.empty()) break;
        uint64_t best_pair = 0;
        int best_count = -1;
        int best_a = -1, best_b = -1;
        for(const auto& [pair, count]: pair_counts) {
            int a, b;
            std::tie(a, b) = pair_unhash(pair);
            if(count > best_count) {
                best_count = count;
                best_pair = pair;
                best_a = a, best_b = b;
            } else if(count == best_count) {
                if(vocabulary[a] > vocabulary[best_a] || 
                (vocabulary[a] == vocabulary[best_a] &&
                 vocabulary[b] > vocabulary[best_b]))
                {
                    best_pair = pair;
                    best_a = a, best_b = b;
                }
            }
        }
        merges.emplace_back(best_a, best_b);
        int new_id = size++;
        vocabulary[new_id] = vocabulary[best_a];
        vocabulary[new_id].insert(vocabulary[new_id].end(), vocabulary[best_b].begin(), vocabulary[best_b].end());

        auto affected_wordids = pair_to_wordid[best_pair];
        for(int wordid: affected_wordids) {
            auto &encoding = wordid_encoding[wordid];
            int count = wordid_count[wordid];
            std::vector<int> new_encoding;
            for(int i = 0; i < encoding.size() - 1; ++i) {
                uint64_t pair = pair_hash(encoding[i], encoding[i + 1]);
                pair_counts[pair] -= count;
                if(pair_counts[pair] <= 0) {
                    pair_counts.erase(pair);
                } else {
                    pair_to_wordid[pair].erase(wordid);
                    if(pair_to_wordid[pair].empty()) {
                        pair_to_wordid.erase(pair);
                    }
                }
            }
            int i = 0;
            while(i < encoding.size()) {
                if(i < encoding.size() - 1 && encoding[i] == best_a && encoding[i + 1] == best_b) {
                    new_encoding.push_back(new_id);
                    i += 2;
                } else {
                    new_encoding.push_back(encoding[i]);
                    i += 1;
                }
            }

            encoding = std::move(new_encoding);
            for(int i = 0; i < encoding.size() - 1; ++i) {
                uint64_t pair = pair_hash(encoding[i], encoding[i + 1]);
                pair_counts[pair] += count;
                pair_to_wordid[pair].insert(wordid);
            }            
        }
    }

    return std::make_tuple(vocabulary, merges);
}

NB_MODULE(c_bpe, m) {
    m.def("train_bpe_cpp", &train_bpe_cpp);
}