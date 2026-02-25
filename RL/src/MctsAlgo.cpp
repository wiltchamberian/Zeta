#include "MctsAlgo.h"

#include "reluLayer.h"
#include "tanhLayer.h"
#include <algorithm>
#include <random>

namespace mcts{
    //P(s; a) = (1 - epsilon ) * pa + epsilon * eta_a
    std::vector<float> Node::getPolicyDistribution(float temperature) {
        //9 actions;
        std::vector<float> distribution(9, 0);
        constexpr float alpha = 0.03f;  //eta ~ Dir(alpha)
        constexpr float epsilon = 0.25f;
        if (temperature > 0) {
            float total = 0;
            for (int i = 0; i < edges.size(); ++i) {
                float v = std::pow(edges[i]->visit_count, 1 / temperature);
                distribution[edges[i]->action] = v;
                total += v;
            }
            if (total > 0) {
                for (int i = 0; i < distribution.size(); ++i) {
                    distribution[i] /= total;
                }
            }
            return distribution;
        }
        else {
            //// temperature = 0 时，加入 Dirichlet 噪声
            //int K = edges.size();
            //std::vector<float> dir_noise(K, 0);
            //std::gamma_distribution<float> gamma_dist(alpha, 1.0f);
            //float sum = 0;
            //for (int i = 0; i < K; ++i) {
            //    dir_noise[i] = gamma_dist(gen);
            //    sum += dir_noise[i];
            //}
            //for (int i = 0; i < K; ++i) dir_noise[i] /= sum; // 归一化

            float total = 0;
            float max = -10000;
            int id = 0;
            for (int i = 0; i < edges.size(); ++i) {
                float v = edges[i]->visit_count;
                if (v > max) {
                    max = v;
                    id = i;
                }
            }
            distribution[edges[id]->action] = 1;
            float noise = 1.0f / edges.size();
            for (int i = 0; i < edges.size(); ++i) {
                distribution[edges[i]->action] = (1 - epsilon) * distribution[edges[i]->action] + epsilon * noise;
            }
            return distribution;
        }

    }

    std::vector<Entry> ReplayBuffer::sample(size_t batch_size)
    {
        std::vector<Entry> batch;
        batch.reserve(batch_size);

        std::uniform_int_distribution<size_t> dist(0, entries.size() - 1);
        std::mt19937 rng(std::random_device{}());

        for (size_t i = 0; i < batch_size; ++i)
        {
            batch.push_back(entries[dist(rng)]);
        }
        return batch;
    }


    void Mcts::backTrace(Node* node, float value) {
        while (node->parent != nullptr) {
            value = -value;
            node->parentEdge->W += value;
            node->parentEdge->visit_count += 1;

            node->parent->subTreeDepth = std::max(node->parent->subTreeDepth, node->subTreeDepth + 1);
            node = node->parent;
        }
    }


    void Mcts::simulate(Node* root) {
        Node* cur = root;
        while (true) {
            if (cur->state->is_terminal()) {
                float value = cur->state->terminal_value();
                backTrace(cur, value);

                return;
            }
            if (!cur->expanded) {
                CuHead head = proxy->predict(cur->state.get());
                auto actions = cur->state->legalActions();
                if (actions.empty()) {
                    assert(false);
                }

                cur->expanded = true;

                cur->edges.reserve(actions.size());
                cur->children.resize(actions.size());

                for (size_t i = 0; i < actions.size(); ++i)
                {
                    std::unique_ptr<Edge> edge = std::make_unique<Edge>(actions[i], head.policy[i]);
                    cur->edges.push_back(std::move(edge));
                }

                backTrace(cur, head.value);

                return;
            }

            int total = 0;
            {
                for (auto& e : cur->edges)
                    total += e->visit_count;
            }
            float best_score = -1e9f;
            int best = 0;
            Edge* selectedEdge = nullptr;
            for (size_t i = 0; i < cur->edges.size(); ++i) {
                Edge* edge = cur->edges[i].get();

                //PUCT equation 
                float q = edge->Q();
                float u = setting.c_puct * edge->prior * sqrt(total) / (1 + edge->visit_count);
                float score = q + u;
                if (total == 0) {
                    score = edge->prior;
                }

                if (score > best_score) {
                    best_score = score;
                    best = i;
                    selectedEdge = edge;
                }
            }
            assert(selectedEdge != nullptr);

            if (cur->children[best] == nullptr) {
                cur->children[best] = std::make_unique<Node>();
                cur->children[best]->parent = cur;
                cur->children[best]->parentEdge = selectedEdge;
            }
            Node* child = cur->children[best].get();
            child->state = cur->state->next_state(selectedEdge->action);

            cur = child;
        }

    }

    void Mcts::search() {

    }

    void Mcts::selfPlay(ReplayBuffer& replay) {
        std::vector<Tensor> labels;
        std::vector<std::shared_ptr<State>> states;

        std::unique_ptr<Node> cur = std::make_unique<Node>();
        cur->state = proxy->createState();

        int chessCount = 0;
        while (!cur->state->is_terminal()) {

            for (int i = 0; i < setting.simulationCount; ++i) {
                simulate(cur.get());
            }
            //std::cout << "Depth: " << cur->state.depth << " visits: ";
            //for (auto& e : cur->edges)
            //    std::cout << e->visit_count << " ";
            //std::cout << std::endl;

            //float temperature = cur->state->depth < 10 ? 1 : 0;
            float temperature = 0.1;

            //action distribution
            std::vector<float> policy_dis = cur->getPolicyDistribution(temperature);

            std::discrete_distribution<> dist(policy_dis.begin(), policy_dis.end());
            int selectedAction = dist(gen);

            //record state
            Tensor policy(9);
            policy.setData(policy_dis);
            labels.push_back(policy);
            states.push_back(cur->state);

            bool bingo = false;
            for (int i = 0; i < cur->edges.size(); ++i) {
                if (cur->edges[i]->action == selectedAction) {
                    auto child = std::move(cur->children[i]);
                    cur->children[i] = nullptr;
                    cur->children.clear();
                    cur->edges.clear();
                    cur = std::move(child);
                    cur->parent = nullptr;
                    cur->parentEdge = nullptr;
                    bingo = true;
                    break;
                }
            }
            if (!bingo) {
                assert(false);
            }
            //state = state.next_state(selectedAction);

            chessCount += 1;
            if (chessCount >= setting.maxChessLength) {
                break;
            }
        }
        int winner = (chessCount >= setting.maxChessLength) ? 0 : (-cur->state->player);
        for (int i = states.size() - 1; i >= 0/*states.size()-3*/; --i) {
            Entry entry;
            entry.label = labels[i];
            entry.state = states[i]->Encode();
            if (winner == 0) {
                entry.value = 0;
            }
            else {
                entry.value = (winner == states[i]->player) ? 1 : -1;
            }

            replay.entries.push_back(entry);
        }
    }

    void Mcts::train() {
        //alpha-go-zero:
        //minibatch: 2048
        //checkpoint: 1000 iteration
        InitRandom();

        ReplayBuffer buffer;
        for (int episode = 0; episode < setting.num_episodes; ++episode) {

            selfPlay(buffer);

            if (buffer.entries.size() >= setting.batchSize) {
                std::uniform_int_distribution<size_t> dist(
                    0, buffer.entries.size() - 1);

                //print last 32 buffers
                //int t = 0;
                //for (int k = buffer.entries.size() - 1; k >= 0; --k) {
                //    TicTac state = TicTac::FromTensor(buffer.entries[k].state);
                //    state.printState();
                //    std::cout << "value:" << buffer.entries[k].value << std::endl;
                //    t++;
                //    if (t == 32) {
                //        break;
                //    }
                //}

                std::vector<Entry> miniBatch;
                miniBatch.reserve(setting.miniBatchSize);

                for (size_t i = 0; i < setting.miniBatchSize; ++i) {
                    size_t idx = dist(gen);
                    miniBatch.push_back(buffer.entries[idx]);
                }

                for (int k = 0; k < setting.trainStepsPerEpisode; ++k) {
                    proxy->train(miniBatch);
                }

                buffer.entries.clear();
            }

            //if (buffer.entries.size() > batchSize) {
            //    // 丢掉前面的旧样本
            //    buffer.entries.erase(buffer.entries.begin(), buffer.entries.end() - batchSize);
            //}

            //if (!buffer.entries.empty()) {
            //    for (int k = 0; k < trainStepsPerEpisode; ++k) {
            //        proxy->train(buffer.entries); // 直接用 buffer.entries（现在就是最新 batchSize 个）
            //    }
            //}
        }
        return;
    }

    std::unique_ptr<State> Mcts::play(const std::unique_ptr<State>& state) const {
        CuHead head = proxy->predict(state.get());
        auto actions = state->legalActions();
        int id = 0;
        float best = -1000;
        for (int i = 0; i < head.policy.size(); ++i) {
            if (head.policy[i] > best) {
                best = head.policy[i];
                id = i;
            }
        }
        Index action = actions[id];
        auto result = state->next_state(action);
        return result;
    }

    std::unique_ptr<State> Mcts::randomPlay(const std::unique_ptr<State>& state) const {
        auto legalActions = state->legalActions();
        std::uniform_int_distribution<size_t> dist(0, legalActions.size() - 1);
        //int act = dis(gen);
        std::mt19937 rng(std::random_device{}());
        return state->next_state(legalActions[dist(rng)]);
    }

    void Mcts::InitRandom() {
        std::random_device rd;
        gen.seed(rd());
    }

    void Mcts::InitRandom(uint32_t seed) {
        gen.seed(seed);
    }

}