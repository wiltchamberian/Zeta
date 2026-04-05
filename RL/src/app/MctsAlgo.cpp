#include "MctsAlgo.h"

#include "reluLayer.h"
#include "tanhLayer.h"
#include <algorithm>
#include <random>
#include <array>

namespace mcts{
    
    int Mcts::backTrace(Node* node, float value) const {
        float mul = 1;
        int loopTime = 0;
        while (node->parent != nullptr) {
            value = -value;
            
            
            float v = loopTime > setting.door ? 0 : ( value * mul);

            node->parentEdge->W += v;


            node->parentEdge->visit_count += 1;

            node->parent->subTreeDepth = std::max(node->parent->subTreeDepth, node->subTreeDepth + 1);
            node = node->parent;

            mul = mul * setting.gamma;

            loopTime += 1;
        }
        return node->subTreeDepth;
    }

    void Mcts::expand(Node* cur, Proxy* proxy) {
        auto actions = cur->state->legalActions();
        if (actions.empty()) {
            assert(false);
        }

        CuHead head = proxy->predict(cur->state.get());

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

    void Mcts::simulate(Node* root, Proxy* proxy, NodePool<Node>* pool) {
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
                cur->children[best] = pool->Alloc();
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

    int Mcts::selfPlay(ReplayBuffer& replay, std::shared_ptr<Proxy> proxy, NodePool<Node>* pool) {
        std::vector<Tensor> labels;
        std::vector<std::shared_ptr<State>> states;
      
        Node* cur = pool->Alloc();

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
                simulate(cur, proxy.get(), pool);
            }
            //std::cout << "Depth: " << cur->state.depth << " visits: ";
            //for (auto& e : cur->edges)
            //    std::cout << e->visit_count << " ";
            //std::cout << std::endl;

            //float temperature = cur->state->depth < 10 ? 1 : 0;
            float temperature = computeTemperature(chessCount);

            //action distribution
            std::vector<double> policy_dis = cur->getPolicyDistribution(temperature , proxy->totalActionCount);

            std::discrete_distribution<> dist(policy_dis.begin(), policy_dis.end());
            int selectedAction = dist(gen);

            //record state
            
            std::vector<double> policy_real = cur->getPolicyDistribution(1, proxy->totalActionCount);
            

            std::vector<std::vector<double>> policies;
            std::vector<std::shared_ptr<State>> newStates = cur->state->permuteStates(policy_real, policies);
            if (!newStates.empty()) {
                for (int i = 0; i < newStates.size(); ++i) {
                    states.push_back(newStates[i]);
                    Tensor policy(proxy->totalActionCount);
                    policy.setData(policies[i]);
                    labels.push_back(policy);
                }
            }
            else {
                Tensor policy(proxy->totalActionCount);
                policy.setData(policy_real);
                labels.push_back(policy);
                states.push_back(cur->state);
            }
            

            bool bingo = false;
            Node* child = nullptr;
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
            pool->FreeTree(cur);
            cur = child;

            chessCount += 1;
            if (chessCount >= setting.maxChessLength) {
                break;
            }
        }
        int winner = (chessCount >= setting.maxChessLength) ? 0 : cur->state->winner();
        //float terminalValue = cur->state->terminal_value();
        //replay.lock();
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
        //replay.unlock();
        pool->FreeTree(cur);

        return chessCount;
    }

    void Mcts::train_proxy() {
        int step = 0;
        int bufferCount = 0;
        while (!endTrain) {
            //ReplayBuffer buf = bufferQueue.pop_s();
            ReplayBuffer buf = replayBuffer.clone_s();
            if (!buf.entries.empty()) {
                if (trainLoopStop == true) {
                    
                    bufferCount += 1;
                    auto proxy = std::shared_ptr<Proxy>(trainProxy->Clone());
                    trainProxy->version += 1;
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        std::atomic_store(&mctsProxy, proxy);
                        globalVersion++;
                    }
                    cv.notify_all();
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] {
                        return trainLoopStop == false;
                        });
      
                }
                //auto [state, action, reward] = ReplayBuffer::EntriesToTensors(buf.entries);
                buf.shuffle();
                while ((!trainLoopStop) && (!endTrain)) {
                    auto entries = buf.sample(setting.miniBatchSize);
                    auto [state, action, reward] = ReplayBuffer::EntriesToTensors(entries);
                    trainProxy->train(state, action, reward);
                    step += 1;
                    if (step % 100 == 0) {
                        std::cout << episode << std::endl;
                        trainProxy->PrintLoss();
                        std::cout << std::endl;
                    }
                }
                
                
            }
            else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        std::cout << "step:" << step << std::endl;
        std::cout << "bufferCount:" << bufferCount << std::endl;
    }

    void Mcts::train() {
#define THREAD_NUM 6
        InitRandom();

        int localVersion = globalVersion.load();
        std::shared_ptr<Proxy> proxy = mctsProxy;
        std::thread th([this]() {
            train_proxy();
            });

        ReplayBuffer tmpBuffer;
        int maxLength = 0;
        ReplayBuffer tempBuffers[THREAD_NUM];
        //run four episode at once...
        NodePool<Node> pool[THREAD_NUM];
        std::array<std::shared_ptr<Proxy>, THREAD_NUM> proxies;
        proxies[0] = proxy;
        for (int i = 1; i < THREAD_NUM; ++i) {
            proxies[i] = std::shared_ptr<Proxy>(proxy->Clone());
        }
        for (int episode = 0; episode < setting.num_episodes; ++episode) {
            this->episode = episode;

            std::vector<std::thread> threads;
            for (int i = 0; i < THREAD_NUM; i++) {
                threads.emplace_back([&tempBuffers, &pool, this, &proxies, i] {
                    selfPlay(tempBuffers[i], proxies[i], &pool[i]);
                    });
            }
            for (auto& t : threads) {
                t.join();
            }

            //int length = selfPlay(tmpBuffer, proxy, &pool);
            //maxLength = std::max<int>(maxLength, length);
            
            //int totalSize = tmpBuffer.entries.size();
            int totalSize = tempBuffers[0].entries.size() + tempBuffers[1].entries.size() + tempBuffers[2].entries.size() + tempBuffers[3].entries.size();
            if (totalSize > setting.batchSize) {

                //bufferQueue.append_s(std::move(tmpBuffer));
                /*replayBuffer.append_s(tmpBuffer.entries);
                tmpBuffer.entries.clear();*/

                for (int i = 0; i < THREAD_NUM; ++i) {
                    replayBuffer.append_s(tempBuffers[i].entries);
                    tempBuffers[i].entries.clear();
                }

                //wait for a new trained proxy
                trainLoopStop.store(true);
                
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] {
                    return globalVersion.load() != localVersion;
                    });
                localVersion = globalVersion.load();

                proxy = std::atomic_load(&mctsProxy);
                proxies[0] = proxy;
                for (int k = 1; k < THREAD_NUM; ++k) {
                    proxies[k].reset(proxy->Clone());
                }

                trainLoopStop = false;
                
                cv.notify_all();
                std::cout << "proxy_version:" << proxy->version << std::endl;
                std::cout << "max_length:" << maxLength << std::endl;
            }
        }


        endTrain.store(true);
        
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

    float Mcts::computeTemperature(int chessCount) const{
        if (chessCount >= setting.explorationCount)
            return setting.targetTemperature;

        float t = (float)chessCount / setting.explorationCount;

        return setting.startTemperature *
            pow(setting.targetTemperature / setting.startTemperature, t);
    }

    void Mcts::MinMax(Node* root) {

        visits.clear();
        values.clear();

        std::vector<Node*> states;
        states.push_back(root);
        while (!states.empty()) {
            Node* t = states.back();
            uint64_t key = t->state->Hash();
            if (t->state->is_terminal()) {
                t->expanded = true;
                visits[key] = VISITIED;
                values[key] = -t->state->player;
                states.pop_back();
                continue;
            }
            else if (t->expanded == false) {
                auto actions = t->state->legalActions();
                t->expanded = true;
                visits[key] = VISITIED;
                for (int k = actions.size() - 1; k >= 0; --k) {
                    Node* nd = pool.Alloc();
                    nd->parent = t;
                    nd->subTreeDepth = t->subTreeDepth + 1;
                    nd->state = t->state->next_state(actions[k]);
                    states.push_back(nd);
                }
            }
            else {
                auto actions = t->state->legalActions();
                for (int k = 0; k < actions.size(); ++k) {
                    auto state = t->state->next_state(actions[k]);
                    auto hash = state->Hash();
                    int v = values[hash];
                    if (t->state->player == 1) {
                        values[key] = std::max(v, values[key]);
                    }
                    else {
                        values[key] = std::min(v, values[key]);
                    }
                }
                states.pop_back();
            }

        }

        for (auto it : visits) {
            int v = values[it.first];
            auto st = root->state->next_state(0);
            st->UnHash(it.first);
            st->printState();
            std::cout << "value:" << v << std::endl;

        }


    }
}

