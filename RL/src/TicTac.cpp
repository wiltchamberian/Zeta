#include "TicTac.h"
#include <random>

TicTac TicTac::next_state(int action) const
{
    TicTac s = *this;

    //action
    int playerId = action / 3;
    int targetId = action % 3;
    int playerRover = 0;
    int targetRover = 0;
    int count = 0;
    for (int i = 0; i < 9; ++i)
    {
        if (s.board[i] == player) {
            if (playerRover == playerId) {
                s.board[i] = 0;
                count += 1;
            }
            playerRover++;
        }
        if (s.board[i] == 0) {
            if (targetRover == targetId) {
                s.board[i] = playerId;
                count += 1;
            }
            targetRover++;
        }
        if (count == 2) {
            break;
        }
    }
    
    s.player = -player;
    s.depth++;
    return s;
}

bool TicTac::legal(int i, int j) const {
    if (i == 4 || j == 4) {
        return true;
    }
    else if (i == 0) {
        if (j == 1 || j == 3) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 1) {
        if (j == 0 || j == 2 ) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 2) {
        if (j == 1 || j == 5) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 3) {
        if (j == 0  || j == 6) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 5) {
        if (j == 2  || j == 8) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 6) {
        if (j == 3 || j == 7) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (i == 7) {
        if ( j == 6 || j == 8) {
            return true;
        }
        else {
            return false;
        }
    }
    else if (j == 8) {
        if (j == 5 || j == 7) {
            return true;
        }
        else {
            return false;
        }
    }
    return false;
}

bool TicTac::legalAction(int action) const {
    int playerId = action / 3;
    int targetId = action % 3;

    int playerRover = 0;
    int targetRover = 0;
    int count = 0;
    int playerPos = 0;
    int targetPos = 0;
    for (int i = 0; i < 9; ++i)
    {
        if (board[i] == playerId) {
            if (playerRover == playerId) {
                playerPos = i;
                count += 1;
            }
            playerRover++;
        }
        if (board[i] == 0) {
            if (targetRover == targetId) {
                targetPos = i;
                count += 1;
            }
            targetRover++;
        }
        if (count == 2) {
            break;
        }
    }

    return legal(playerPos, targetPos);
}

std::vector<int> TicTac::legalActions() const{
    std::vector<int> actions;
    int m = 0;
    int n = 0;
    int startPos = 0;
    int endPos = 0;
    std::vector<int> startPositions;
    std::vector<int> endPositions;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == player) {
            startPositions.push_back(i);
        }
        else if (board[i] == 0) {
            endPositions.push_back(i);
        }
    }
    for (int i = 0; i < 9; ++i) {
        int s = i / 3;
        int t = i % 3;
        if (legal(startPositions[s], endPositions[t])) {
            actions.push_back(i);
        }
    }
    
    return actions;
}

std::vector<float> TicTacNode::getPolicyDistribution(float temperature) {
    std::vector<float> distribution(9, 0);
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

CuHead TicTacNNProxy::predict(const TicTac& state) {
    CuHead head;
    Tensor input = state.Encode();

    Tensor result = cunn->ForwardAndFetchPredY(input);

    std::vector<int> legalActions = state.legalActions();
    
    for (int i = 0; i < legalActions.size(); ++i) {
        head.policy.push_back(result(0,legalActions[i]));
    }

    //re-softmax
    float total = 0.0f;
    for (int j = 0; j < head.policy.size(); ++j) {
        total += head.policy[j];
    }
    if (total > 0) {
        for (int j = 0; j < head.policy.size(); ++j) {
            head.policy[j] = head.policy[j] / total;
        }
    }
    return head;

}

void TicTacMcts::backTrace(TicTacNode* node, float value) {
    do {
        value = -value;
        node->parentEdge->W += value;
        node = node->parent;
    } while (node->parent != nullptr);
    //
}


float TicTacMcts::simulate(TicTacNode* root) {

    float c_puct = 0.5;

    if (root->state.is_terminal())
        return root->state.terminal_value();

    TicTacNode* cur = root;
    while (true) {
        if (!cur->expanded){
            CuHead head = proxy->predict(cur->state);
            auto actions = cur->state.legalActions();

            cur->expanded = true;

            cur->edges.reserve(actions.size());
            cur->children.resize(actions.size());

            for (size_t i = 0; i < actions.size(); ++i)
            {
                std::unique_ptr<TicTacEdge> edge = std::make_unique<TicTacEdge>(actions[i], head.policy[i]);
                
                cur->children[i] = std::make_unique<TicTacNode>();
                cur->children[i]->parent = cur;
                cur->children[i]->parentEdge = edge.get();

                cur->edges.push_back(std::move(std::make_unique<TicTacEdge>(actions[i], head.policy[i])));
            }

            backTrace(cur, head.value);
            
            return head.value;
        }

        int total = 0;
        {
            for (auto& e : cur->edges)
                total += e->visit_count;
        }
        float best_score = -1e9f;
        int best = 0;
        TicTacEdge* selectedEdge = nullptr;
        for (size_t i = 0; i < cur->edges.size(); ++i) {
            TicTacEdge* edge = cur->edges[i].get();

            //PUCT equation 
            float q = edge->Q();
            float u = c_puct * edge->prior * sqrt(total) / (1 + edge->visit_count);
            float score = q + u;

            if (score > best_score) {
                best_score = score;
                best = i;
                selectedEdge = edge;
            }
        }

        selectedEdge->visit_count += 1;

        TicTacNode* child = cur->children[best].get();
        child->state = cur->state.next_state(selectedEdge->action);

        cur = child;
    }

}

void TicTacMcts::search() {

}

void TicTacMcts::selfPlay(TicTacReplayBuffer& replay) {
    std::random_device rd;
    std::mt19937 gen(rd());
    TicTac state = TicTac::initState();
    
    int simulationCount = 100;
    std::vector<Tensor> labels;
    std::vector<TicTac> states;
    while (!state.is_terminal()) {
        
        std::unique_ptr<TicTacNode> root = std::make_unique<TicTacNode>();
        root->state = state;

        for (int i = 0; i < simulationCount; ++i) {
            simulate(root.get());
        }

        float temperature = state.depth < 10 ? 1 : 0.01;

        std::vector<float> policy_dis = root->getPolicyDistribution(temperature);

        std::discrete_distribution<> dist(policy_dis.begin(), policy_dis.end());
        int selection = dist(gen);

        Tensor tensor(9);
        tensor.setData(policy_dis);
        labels.push_back(tensor);
        states.push_back(state);

        state = state.next_state(selection);

    }
    int winner = -state.player;
    for (int i = 0; i < states.size(); ++i) {
        TicTacEntry entry;
        entry.label = labels[i];
        entry.state = states[i].Encode();
        entry.value = (winner == states[i].player) ? 1 : -1;
        replay.entries.push_back(entry);
    }
}

void TicTacMcts::train() {
    const int num_episodes = 1000;
    TicTacReplayBuffer buffer;
    int batch = 10;
    for (int episode = 0; episode < num_episodes; ++episode) {

        selfPlay(buffer);

        if (buffer.entries.size() > batch) {
            //sampe and train
        }
    }
    return;
}
