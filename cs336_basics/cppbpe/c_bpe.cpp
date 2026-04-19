#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/tuple.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace nb = nanobind;

inline uint64_t pair_hash(int a, int b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

inline std::pair<int, int> pair_unhash(uint64_t pair) {
    int a = static_cast<int>(pair >> 32);
    int b = static_cast<int>(pair & 0xFFFFFFFF);
    return {a, b};
}

struct PQElement {
    int count;
    uint64_t pair_key;
};

class comparePQ {
    const std::unordered_map<int, std::vector<uint8_t>>* vocabulary;

public:
    comparePQ(const std::unordered_map<int, std::vector<uint8_t>>* vocab) : vocabulary(vocab) {}

    bool operator()(const PQElement& lhs, const PQElement& rhs) const {
        if (lhs.count != rhs.count) {
            return lhs.count < rhs.count; 
        } else {
            auto [lhs_a, lhs_b] = pair_unhash(lhs.pair_key);
            auto [rhs_a, rhs_b] = pair_unhash(rhs.pair_key);
            
            const auto& lhs_vocab_a = (*vocabulary).at(lhs_a);
            const auto& rhs_vocab_a = (*vocabulary).at(rhs_a);

            if (lhs_vocab_a != rhs_vocab_a) {
                return lhs_vocab_a < rhs_vocab_a; 
            } 
            
            const auto& lhs_vocab_b = (*vocabulary).at(lhs_b);
            const auto& rhs_vocab_b = (*vocabulary).at(rhs_b);
            return lhs_vocab_b < rhs_vocab_b;
        }
    }
};

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
    std::unordered_map<uint64_t, std::unordered_set<int>> pair_to_wordid;

    for (const auto& [wordid, count] : wordid_count) {
        const auto &encoding = wordid_encoding[wordid];
        for(int i = 0; i < encoding.size() - 1; ++i) {
            uint64_t pair = pair_hash(encoding[i], encoding[i + 1]);
            pair_counts[pair] += count;
            pair_to_wordid[pair].insert(wordid);
        }
    }
    
    std::priority_queue<PQElement, std::vector<PQElement>, comparePQ> pq((comparePQ(&vocabulary)));
    for (const auto & [pair, count] : pair_counts) {
        pq.push(PQElement{count, pair});
    }

    while(size < vocab_size) {
        if (pair_counts.empty()) break;

        if (size % 500 == 0) {
            std::cout << "[C++ Engine] Vocab size reached: " << size 
                      << " / " << vocab_size 
                      << " | Remaining active pairs: " << pair_counts.size() 
                      << std::endl;
        }

        uint64_t best_pair = 0;
        int best_count = -1;
        int best_a = -1, best_b = -1;
        bool found = false;

        while(!pq.empty()) {
            auto [count, pair_key] = pq.top();
            pq.pop();
            auto it = pair_counts.find(pair_key);
            if(it != pair_counts.end() && it->second == count) {
                best_pair = pair_key;
                best_count = count;
                std::tie(best_a, best_b) = pair_unhash(best_pair);     
                found = true;     
                break;     
            }
        }

        if(!found) {
            break;
        }

        merges.emplace_back(best_a, best_b);
        int new_id = size++;
        vocabulary[new_id] = vocabulary[best_a];
        vocabulary[new_id].insert(vocabulary[new_id].end(), vocabulary[best_b].begin(), vocabulary[best_b].end());

        auto affected_wordids = std::move(pair_to_wordid[best_pair]);
        pair_counts.erase(best_pair);
        pair_to_wordid.erase(best_pair);

        std::unordered_set<uint64_t> changed_pairs;

        for(int wordid: affected_wordids) {
            auto &encoding = wordid_encoding[wordid];
            int count = wordid_count[wordid];

            std::vector<int> new_encoding;
            new_encoding.reserve(encoding.size());

            bool last_merged = false;
            int i = 0;
            while(i < encoding.size()) {
                if(i < encoding.size() - 1 && encoding[i] == best_a && encoding[i + 1] == best_b) {
                    if(i > 0 && !last_merged) {
                        uint64_t left = pair_hash(encoding[i - 1], best_a);
                        if(left != best_pair) {
                            pair_counts[left] -= count;
                            if(pair_counts[left] <= 0) {
                                pair_counts.erase(left);
                                pair_to_wordid.erase(left);
                            } else {
                                changed_pairs.insert(left);
                            }
                        }
                    }

                    if(i < static_cast<int>(encoding.size() - 2)) {
                        uint64_t right = pair_hash(best_b, encoding[i + 2]);
                        if(right != best_pair) {
                            pair_counts[right] -= count;
                            if(pair_counts[right] <= 0) {
                                pair_counts.erase(right);
                                pair_to_wordid.erase(right);
                            } else {
                                changed_pairs.insert(right);
                            }
                        }
                    }

                    new_encoding.push_back(new_id);
                    i += 2;
                    last_merged = true;
                } else {
                    new_encoding.push_back(encoding[i]);
                    last_merged = false;
                    i += 1;
                }
            }

            encoding = std::move(new_encoding);

            for(int j = 0; j < encoding.size() - 1; ++j) {
                if(encoding[j] == new_id || encoding[j + 1] == new_id) {
                    uint64_t pair = pair_hash(encoding[j], encoding[j + 1]);
                    pair_counts[pair] += count;
                    pair_to_wordid[pair].insert(wordid);
                    changed_pairs.insert(pair);
                }
            }
        }

        for(uint64_t pair : changed_pairs) {
            auto it = pair_counts.find(pair);
            if(it != pair_counts.end()) {
                pq.push(PQElement{it->second, pair});
            }
        }
    }

    return std::make_tuple(vocabulary, merges);
}

NB_MODULE(c_bpe, m) {
    m.def("train_bpe_cpp", &train_bpe_cpp);
}