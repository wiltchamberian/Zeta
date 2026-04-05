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
        std::shared_ptr<State> initialState = proxy->createState();
        Node* cur = createNodeIfNotExist(initialState);

        std::vector<Node*> vec;
        vec.push_back(cur);
        std::vector<Edge*> edges;
        int terminal_count = 0;
        bool is_terminal = false;
        for (int i = 0; i < setting.max_play_length; ++i) {

            float  epsilon = std::max(setting.epsilon_final, setting.epsilon_init - i * setting.decay_rate);
            if (cur->state->is_terminal()) {
                is_terminal = true;
                terminal_count += 1;
                break;
            }
            if (!cur->expanded) {
                expand(cur);
            }
            //UpdateState(cur);

            // 计算总访问次数
            int best = chooseActionIndexWithPuct(cur, epsilon);
            auto actions = cur->state->legalActions();
            int action = actions[best];

            std::shared_ptr<State> nextState = cur->state->next_state(action);
            Node* next = createNodeIfNotExist(nextState);
           
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
        if ((!is_terminal) && (!cur->expanded)) {
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


        //for (int i = vec.size() - 1; i >= 0; --i) {
        //    UpdateState(vec[i]);
        //}

        return terminal_count;
    }

    int QLearning::chooseActionIndexWithPuct(Node* cur, float epsilon) {
        int best = 0;

        // 计算总访问次数
        int total = 0;
        for (auto& e : cur->edges)
            total += e->visit_count;

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

    int QLearning::chooseActionIndexByEpsilonPolicy(Node* cur, float epsilon) {
        
        int maxIndex = 0;
        float qmax = -10e6;
        float qmin = 10e6;
        for (int i = 0; i < cur->edges.size(); ++i) {
            if (qmax < cur->edges[i]->Q()) {
                qmax = cur->edges[i]->Q();
                maxIndex = i;
            }
            if (qmin > cur->edges[i]->Q()) {
                qmin = cur->edges[i]->Q();
            }
        }

        int chosenIndex = 0;
        //nearly the same, use 
        if (qmax - qmin < 10e-6) {
            std::uniform_int_distribution<int> dist(0, cur->edges.size() - 1);
            chosenIndex = dist(gen);
            return chosenIndex;
        }
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        float r = prob_dist(gen);
        
        if (r < epsilon) {
            std::uniform_int_distribution<int> dist(0, cur->edges.size()-1);
            chosenIndex = dist(gen);
        }
        else {
            chosenIndex = maxIndex;
        }
        return chosenIndex;
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
            std::unique_ptr<Edge> edge = std::make_unique<Edge>(actions[i], 1.0/actions.size());
            
            auto newState = cur->state->next_state(actions[i]);
            if (newState->is_terminal()) {
                int win = newState->winner();
                if (win == cur->state->player) {
                    edge->reward = 1;
                }
                else {
                    edge->reward = -1;
                }
            }
            
            cur->edges.push_back(std::move(edge));


        }
    }

    Node* QLearning::createNodeIfNotExist(std::shared_ptr<State> st) {
        uint64_t h = st->Hash();
        Node* next = nullptr;
        auto iter = table.find(h);
        if (iter != table.end()) {
            next = iter->second;
        }
        else {
            next = pool.Alloc();
            next->state = st;
            table.insert(std::make_pair(h, next));
            expand(next);
        }
        return next;
    }

    void QLearning::Backup(Node* nd) {
        int totalVisitCount = 1;
        float sum_childq = 0;
        for (int i = 0; i < nd->edges.size(); ++i) {
            totalVisitCount += nd->edges[i]->visit_count;
            sum_childq += nd->children[i]->q * nd->edges[i]->visit_count;
        }
        nd->q = (nd->u + sum_childq) * 1.0 / totalVisitCount;
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

    int QLearning::train_mcts() {
        int n = 0;
        for (int i = 0; i < setting.episode_num; ++i) {
            n += mctsPlay();
        }
        return n;
    }

    int QLearning::mctsPlay() {
        
        std::shared_ptr<State> initialState = proxy->createState();
        Node* cur = createNodeIfNotExist(initialState);
        bool is_terminal = false;
        uint64_t h = initialState->Hash();
        std::unordered_map<uint64_t, std::pair<std::shared_ptr<State>, int>> buf;

        int winner = 0;
        for (int i = 0; i < setting.max_play_length; ++i) {

            float  epsilon = std::max(setting.epsilon_final, setting.epsilon_init - i * setting.decay_rate);
            if (cur->state->is_terminal()) {
                winner = cur->state->winner();
                break;
            }

            int best = chooseActionIndexByEpsilonPolicy(cur, epsilon);
            auto actions = cur->state->legalActions();
            int action = actions[best];
            uint64_t h = cur->state->Hash();
            if (buf.find(h) == buf.end()) {
                buf.insert(std::make_pair(h, std::pair(cur->state, best)));
            }

            std::shared_ptr<State> nextState = cur->state->next_state(action);
            Node* next = createNodeIfNotExist(nextState);
            uint64_t hNext = nextState->Hash();
            
            cur = next;
        }

        //backup
        for (auto iter : buf) {
            std::shared_ptr<State> st = iter.second.first;
            Node* nd = table[iter.first];
            Edge* edge = nd->edges[iter.second.second].get();
            if (winner == 0) {
                edge->W += 0;
                edge->visit_count += 1;
                edge->q = edge->W / edge->visit_count;
            }
            else {
                if (st->player != winner) {
                    edge->W += -1;
                    edge->visit_count += 1;
                }
                else {
                    edge->W += 1;
                    edge->visit_count += 1;
                }
            }
            
        }

        return 0;
    }

    ///////////////////////////////////////////////////////////////

   
}


