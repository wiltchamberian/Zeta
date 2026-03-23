#include "QLearning.h"
#include <climits>
#include <algorithm>

namespace zeta {

    void QLearning::UpdateState(Node* nd) {
        for (int i = 0; i < nd->edges.size(); ++i) {
            Node* child = nd->children[i];
            if (child != nullptr /*&& (!child->edges.empty())*/) {
                int index = 0;
                float q_max = -std::numeric_limits<float>::infinity();
                for (int j = 0; j < child->edges.size(); ++j) {
                    if (child->edges[j]->q > q_max) {
                        index = j;
                        q_max = child->edges[j]->q;
                    }
                }
                if (child->edges.empty()) {
                    q_max = 0;
                }
                
                nd->edges[i]->q = (1 - setting.alpha)* nd->edges[i]->q + setting.alpha * (nd->edges[i]->reward + setting.gamma * q_max);
                
                //if (q_max > -std::numeric_limits<float>::infinity()) {
                //    std::cout << "1\n";
                //}
                //if (nd->edges[i]->q < 0 || nd->edges[i]->q > 0) {
                //    std::cout << "1\n";
                //}
                //if (nd->edges[i]->q != 0) {
                //    std::cout << "1\n";
                //}
            }
                
        }
        

    }

    int QLearning::singlePlay(NodePool<Node>* pool, std::unordered_map<uint64_t, Node*>& table) {
        Node* cur = nullptr;
        std::shared_ptr<State> initialState = proxy->createState();
        uint64_t h = initialState->Hash();
        if (table.find(h) != table.end()) {
            cur = table[h];
        }
        else {
            cur = pool->Alloc();
            cur->state = initialState;
        }
        std::vector<Node*> vec;
        vec.push_back(cur);
        std::vector<Edge*> edges;
        int terminal_count = 0;

        //float epsilon_init = 0.1f;
        //float epsilon_final = 0.05f;
        //float decay_rate = 0.001f; 

        for (int i = 0; i < setting.max_play_length; ++i) {

            float  epsilon = std::max(setting.epsilon_final, setting.epsilon_init - i * setting.decay_rate);


            if (cur->state->is_terminal()) {
                terminal_count += 1;
                break;
            }
            if (!cur->expanded) {
                expand(cur);
            }
            //UpdateState(cur);

            // 计算总访问次数
            int best = chooseActionIndex(cur, epsilon);
            auto actions = cur->state->legalActions();
            int action = actions[best];

            std::shared_ptr<State> nextState = cur->state->next_state(action);
            uint64_t h = nextState->Hash();
            Node* next = nullptr;
            auto iter = table.find(h);
            if (iter != table.end()) {
               next = iter->second;
            }
            else {
                next = pool->Alloc();
                next->state = nextState;
                uint64_t h = next->state->Hash();
                table.insert(std::make_pair(h, next));
            }
            if (cur->children[best] == nullptr) {
                cur->children[best] = next;
            }
            else {
                assert(cur->children[best] == next);
            }
            edges.push_back(cur->edges[best].get());
            
            cur = next;
            vec.push_back(cur);
        }
        if (!cur->expanded) {
            expand(cur);
        }
 
        //backup
        float v = 0;
        if (cur->state->is_terminal()) {
            int win = cur->state->winner();
            assert(win == -cur->state->player
            || win == 0);
            if (win == 0) {
                v = 0;
            }
            else {
                v = 1;
            }
        }
        for (int i = edges.size() - 1; i >= 0; --i) {
            edges[i]->visit_count += 1;
            edges[i]->W += v;
            v = -v;
        }
        for (int i = vec.size() - 1; i >= 0; --i) {
            UpdateState(vec[i]);
        }

        return terminal_count;
    }

    int QLearning::chooseActionIndex(Node* cur, float epsilon) {
        int best = 0;

        // 计算总访问次数
        int total = 0;
        for (auto& e : cur->edges)
            total += e->visit_count;

        //if (total != 0) {
        //    std::cout << "1\n";
        //}

        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        float r = prob_dist(gen);

        if (r < epsilon) {
            // 随机选择动作
            std::uniform_int_distribution<size_t> dist(0, cur->edges.size() - 1);
            best = dist(gen);
        }
        else {
            // PUCT选择
            float best_score = -1e9f;
            for (int i = 0; i < cur->edges.size(); ++i) {
                Edge* edge = cur->edges[i].get();
                float q = edge->Q();
                float u = setting.c_puct * edge->prior * sqrt(total) / (1 + edge->visit_count);
                float score = (total == 0) ? edge->prior : q + u;

                if (score > best_score) {
                    best_score = score;
                    best = i;
                }
            }
        }
        return best;
    }

    void QLearning::expand(Node* cur) {
        auto actions = cur->state->legalActions();
        if (actions.empty()) {
            //assert(false);
            return;
        }

        cur->expanded = true;

        cur->edges.reserve(actions.size());
        cur->children.resize(actions.size(), nullptr);

        for (size_t i = 0; i < actions.size(); ++i)
        {
            std::unique_ptr<Edge> edge = std::make_unique<Edge>(actions[i], 0);
            
            auto newState = cur->state->next_state(actions[i]);
            if (newState->is_terminal()) {
                int win = newState->winner();
                if (win == cur->state->player) {
                    edge->reward = 10;
                }
                else {
                    edge->reward = -10;
                }
            }
            
            cur->edges.push_back(std::move(edge));


        }
    }

    int QLearning::train() {
        int n = 0;
        for (int i = 0; i < setting.episode_num; ++i) {
            n += singlePlay(&pool, table);
        }
        return n;
    }

    std::shared_ptr<State> QLearning::play(const std::shared_ptr<State>& state) const {
        uint64_t h = state->Hash();
        auto iter = table.find(h);
        if (iter != table.end()) {
            auto st = iter->second->state;
            Node* nd = iter->second;
            float q_max = - std::numeric_limits<float>::infinity();
            int chosen = 0;

            std::vector<int> legals = nd->state->legalActions();
            assert(legals.size() == nd->edges.size());
            std::vector<float> probs(legals.size(),0);
            for (int i = 0; i < nd->edges.size(); ++i) {
                if (nd->edges[i]->q > q_max) {
                    q_max = nd->edges[i]->q;
                    chosen = i;
                }
            }
            //print
            for (int i = 0; i < probs.size(); ++i) {
                std::cout << legals[i] << ":" << nd->edges[i]->q <<" ";
            }
            std::cout << std::endl;
            
            int action = nd->edges[chosen]->action;
            return st->next_state(action);
        }
        else {
            std::cout << "unSeen state!\n";
            
            std::vector<int> actions = state->legalActions();
            return state->next_state(actions[0]);
            //std::uniform_int_distribution<int> dist(0, static_cast<int>(actions.size() - 1));
            //int index = dist(gen);
            //int action = actions[index];
            //int action = actions[index];
            //return state->next_state(action);
        }
        return nullptr;
    }
}


