#include "MctsAlgo.h"

#include "reluLayer.h"
#include "tanhLayer.h"
#include <algorithm>
#include <random>

namespace mcts{
    std::vector<float> Node::getPolicyDistribution(float temperature, int totalActionCount) {
        std::vector<float> distribution(totalActionCount, 0);
        constexpr float alpha = 0.03f;  //eta ~ Dir(alpha)
        constexpr float epsilon = 0.25f;
        if (temperature == 1) {
            float total = 0;
            for (int i = 0; i < edges.size(); ++i) {
                float v = edges[i]->visit_count;
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
        else if (temperature == 0) {
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
            return distribution;
        }
        else {
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


    int Mcts::backTrace(Node* node, float value) {
        while (node->parent != nullptr) {
            value = -value;
            node->parentEdge->W += value;
            node->parentEdge->visit_count += 1;

            node->parent->subTreeDepth = std::max(node->parent->subTreeDepth, node->subTreeDepth + 1);
            node = node->parent;
        }
        return node->subTreeDepth;
    }

    void Mcts::expand(Node* cur, Proxy* proxy) {
        CuHead head = proxy->predict(cur->state.get());
        auto actions = cur->state->legalActions();
        if (actions.empty()) {
            assert(false);
        }

        cur->expanded = true;

        cur->edges.reserve(actions.size());
        cur->children.resize(actions.size(), nullptr);

        for (size_t i = 0; i < actions.size(); ++i)
        {
            std::unique_ptr<Edge> edge = std::make_unique<Edge>(actions[i], head.policy[i]);
            cur->edges.push_back(std::move(edge));
        }

        backTrace(cur, head.value);
    }

    void Mcts::simulate(Node* root, Proxy* proxy) {
        Node* cur = root;
        while (true) {
            if (cur->state->is_terminal()) {
                float value = cur->state->terminal_value();
                int depth = backTrace(cur, value);
                return;
            }
            if (!cur->expanded) {
                expand(cur, proxy);

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
                cur->children[best] = pool.Alloc();
                cur->children[best]->parent = cur;
                cur->children[best]->parentEdge = selectedEdge;
            }
            Node* child = cur->children[best];
            child->state = cur->state->next_state(selectedEdge->action);

            cur = child;
        }

    }

    std::vector<float> sample_dirichlet(int n, float alpha)
    {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::gamma_distribution<float> gamma(alpha, 1.0f);

        std::vector<float> samples(n);
        float sum = 0.0f;

        for (int i = 0; i < n; ++i) {
            samples[i] = gamma(gen);
            sum += samples[i];
        }

        // 归一化
        for (int i = 0; i < n; ++i) {
            samples[i] /= sum;
        }

        return samples;
    }

    void Mcts::addDirichletNoise(Node* cur) {
        int n = cur->edges.size();
        std::vector<float> noise = sample_dirichlet(n, setting.dirichletNoise);

        for (int i = 0; i < n; ++i) {
            cur->edges[i]->prior =
                (1 - setting.epsilon) * cur->edges[i]->prior
                + setting.epsilon * noise[i];
        }
    }

    int Mcts::selfPlay(ReplayBuffer& replay, std::shared_ptr<Proxy> proxy) {
        std::vector<Tensor> labels;
        std::vector<std::shared_ptr<State>> states;

        Node* cur = pool.Alloc();

        cur->state = proxy->createState();

        int chessCount = 0;
        while (!cur->state->is_terminal()) {
            //as paper, add dirichlet noise here

            if (setting.useDirichletNoise) {
                if (!cur->expanded) {
                    expand(cur, proxy.get());
                    addDirichletNoise(cur);
                }
            }
            
            for (int i = 0; i < setting.simulationCount; ++i) {
                simulate(cur, proxy.get());
            }
            //std::cout << "Depth: " << cur->state.depth << " visits: ";
            //for (auto& e : cur->edges)
            //    std::cout << e->visit_count << " ";
            //std::cout << std::endl;

            //float temperature = cur->state->depth < 10 ? 1 : 0;
            float temperature = chessCount < setting.explorationCount ? 1 : setting.targetTemperature;

            //action distribution
            std::vector<float> policy_dis = cur->getPolicyDistribution(temperature , proxy->totalActionCount);

            std::discrete_distribution<> dist(policy_dis.begin(), policy_dis.end());
            int selectedAction = dist(gen);

            //record state
            Tensor policy(proxy->totalActionCount);
            policy.setData(policy_dis);
            labels.push_back(policy);
            states.push_back(cur->state);

            bool bingo = false;
            mcts::Node* child = nullptr;
            for (int i = 0; i < cur->edges.size(); ++i) {
                if (cur->edges[i]->action == selectedAction) {
                    child = cur->children[i];
                    cur->children[i] = nullptr;
                    child->parent = nullptr;
                    child->parentEdge = nullptr;
                    bingo = true;
                    break;
                }
            }
            if (!bingo) {
                assert(false);
            }
            pool.FreeTree(cur);
            cur = child;
            //state = state.next_state(selectedAction);

            chessCount += 1;
            if (chessCount >= setting.maxChessLength) {
                break;
            }
        }
        int winner = (chessCount >= setting.maxChessLength) ? 0 : cur->state->winner();
        //float terminalValue = cur->state->terminal_value();
        replay.lock();
        for (int i = states.size() - 1; i >= 0/*states.size()-3*/; --i) {
            Entry entry;
            entry.label = labels[i];
            entry.state = states[i]->Encode();
            if (winner == 0) {
                entry.value = 0;
            }
            else {
                //entry.value = cur->state->terminal_value();
                entry.value = (winner == states[i]->player) ? 1 : -1;
            }
            replay.entries.push_back(entry);
        }
        replay.unlock();
        pool.FreeTree(cur);

        return chessCount;
    }

    void Mcts::train() {
        int step = 0;
        int bufferCount = 0;
        while (true) {
            int doubleStop = 0;
            if (stop) {
                doubleStop += 1;
            }
            ReplayBuffer buf = bufferQueue.pop_s();
            if (!buf.entries.empty()) {
                for (int i = 0; i < setting.trainStepsPerEpisode; ++i) {
                    trainProxy->train(buf.entries);
                    step += 1;
                }
                bufferCount += 1;
                auto proxy = std::shared_ptr<Proxy>(trainProxy->Clone());
                trainProxy->version += 1;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    std::atomic_store(&mctsProxy, proxy);
                    globalVersion++;
                }
                cv.notify_all();
            }
            else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            if (stop) {
                doubleStop += 1;
            }
            if (doubleStop) {
                break;
            }
        }
        std::cout << "step:" << step << std::endl;
        std::cout << "bufferCount:" << bufferCount << std::endl;
    }

    void Mcts::run() {
        InitRandom();

        int localVersion = globalVersion.load();
        std::shared_ptr<Proxy> proxy = mctsProxy;
        std::thread th([this]() {
            train();
            });

        ReplayBuffer tmpBuffer;
        int maxLength = 0;
        for (int episode = 0; episode < setting.num_episodes; ++episode) {
            int length = selfPlay(tmpBuffer, proxy);
            maxLength = std::max<int>(maxLength, length);
            if (tmpBuffer.entries.size() > setting.batchSize) {

                bufferQueue.append_s(std::move(tmpBuffer));

                //wait for a new trained proxy
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] {
                    return globalVersion.load() != localVersion;
                    });
                localVersion = globalVersion.load();

                proxy = std::atomic_load(&mctsProxy);

                std::cout << "proxy_version:" << proxy->version << std::endl;
                std::cout << "max_length:" << maxLength << std::endl;
            }
            //if ((episode+1) % setting.sample_episodes == 0) {
            //    //wait for a new trained proxy
            //    std::unique_lock<std::mutex> lock(mtx);
            //    cv.wait(lock, [&] {
            //        return globalVersion.load() != localVersion;
            //        });
            //    localVersion = globalVersion.load();
            //}
        }

        stop = true;
        th.join();

        return;
    }

    std::shared_ptr<State> Mcts::play(const std::shared_ptr<State>& state) const {
        auto proxy = std::atomic_load(&mctsProxy);
        
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

    std::shared_ptr<State> Mcts::randomPlay(const std::shared_ptr<State>& state) const {
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