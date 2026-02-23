#include "TicTac.h"
#include "reluLayer.h"
#include "tanhLayer.h"
#include <algorithm>

TicTac TicTac::next_state(int action) const
{
    TicTac s = *this;

    //action
    int playerId = action / 3;
    int targetId = action % 3;
    int playerRover = 0;
    int targetRover = 0;
    int count = 0;

    int startPos = 0;
    int endPos = 0;
    for (int i = 0; i < 9; ++i)
    {
        if (s.board[i] == player) {
            if (playerRover == playerId) {
                startPos = i;
                count += 1;
            }
            playerRover++;
        }
        if (s.board[i] == 0) {
            if (targetRover == targetId) {
                endPos = i;
                count += 1;
            }
            targetRover++;
        }
        if (count == 2) {
            break;
        }
    }

    s.board[startPos] = 0;
    s.board[endPos] = player;
    
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
    else if (i == 8) {
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
        if (board[i] == player) {
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

//P(s; a) = (1 - epsilon ) * pa + epsilon * eta_a
std::vector<float> TicTacNode::getPolicyDistribution(float temperature) {
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

CuHead TicTacNNProxy::predict(const TicTac& state) {
    CuHead head;
    Tensor input = state.Encode();

    nn->Forward(input);
    policyHead->FetchActivationToCpu();
    Tensor& result = policyHead->distribution;
    valueHead->FetchPredYToCpu();
    Tensor& value = valueHead->predY;

    std::vector<int> legalActions = state.legalActions();
    
    head.value = value(0,0);
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

void TicTacNNProxy::setLearningRate(float rate) {
    nn->SetLearningRate(rate);
}

//build from scratch
void TicTacNNProxy::createNetwork(float learningRate) {

    nn = std::make_unique<CuNN>();
    nn->SetLearningRate(learningRate);

    //layer
    auto c1 = nn->CreateLayer<CuConvolutionLayer>(16, 2, 3, 3);
    c1->alpha = 0.0;
    c1->RandomParameters();
    root = c1;

    auto c2 = nn->CreateLayer<CuConvolutionLayer>(16, 16, 3, 3);
    c2->alpha = 0.0;
    c2->RandomParameters();
    c1->AddLayer(c2);

    //1d conv
    auto c3 = nn->CreateLayer<CuConvolutionLayer>(1, 16, 1, 1);
    c3->padH = 0;
    c3->padW = 0;
    c3->RandomParameters();
    c2->AddLayer(c3);

    auto relu = nn->CreateLayer<CuReluLayer>();
    c3->AddLayer(relu);

    auto fully1 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    fully1->RandomParameters();
    auto relu1 = nn->CreateLayer<CuReluLayer>();
    fully1->AddLayer(relu1);

    auto cross = nn->CreateLayer<CuSoftmaxCrossEntropyLayer>();
    relu1->AddLayer(cross);

    auto fully2 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    fully2->RandomParameters();
    auto relu2 = nn->CreateLayer<CuReluLayer>();
    fully2->AddLayer(relu2);
   

    auto fully2_1 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 1);
    fully2_1->RandomParameters();

    relu2->AddLayer(fully2_1);

    auto tanh = nn->CreateLayer<CuTanhLayer>();
    fully2_1->AddLayer(tanh);

    auto mse = nn->CreateLayer<CuMseLayer>();

    tanh->AddLayer(mse);

    relu->AddLayer(fully1);
    relu->AddLayer(fully2);

    auto tail = nn->CreateLayer<CuAddLayer>();
    cross->AddLayer(tail);
    mse->AddLayer(tail);

    policyHead = cross;
    valueHead = mse;

    nn->AllocDeviceMemory();
}

void TicTacNNProxy::train(const std::vector<TicTacEntry>& entries) {
    if (entries.empty()) {
        return;
    }
    Tensor label(entries.size(), entries[0].label.shape[0]);
    Tensor values(entries.size());
    Tensor states(entries.size(), entries[0].state.shape[1], entries[0].state.shape[2], entries[0].state.shape[3]);
    for (int i = 0; i < entries.size(); ++i) {
        label[i].copy(entries[i].label);
        values(i) = entries[i].value;
        states[i].copy(entries[i].state);
    }
    policyHead->label = label;
    valueHead->label = values;
    
    nn->Forward(states);

    policyHead->BindLabelToDevice();
    valueHead->BindLabelToDevice();


    float crossLoss = policyHead->FetchLoss();
    float mseLoss = valueHead->FetchLoss();
    float loss = crossLoss + mseLoss;
    std::cout << "loss:" << loss << " mse:" << mseLoss << " cross:" << crossLoss << std::endl;

    

    nn->Backward();
    nn->Step();
}


std::vector<TicTacEntry> TicTacReplayBuffer::sample(size_t batch_size)
{
    std::vector<TicTacEntry> batch;
    batch.reserve(batch_size);

    std::uniform_int_distribution<size_t> dist(0, entries.size() - 1);
    std::mt19937 rng(std::random_device{}());

    for (size_t i = 0; i < batch_size; ++i)
    {
        batch.push_back(entries[dist(rng)]);
    }
    return batch;
}


void TicTacMcts::backTrace(TicTacNode* node, float value) {
    while (node->parent != nullptr) {
        value = -value;
        node->parentEdge->W += value;
        node->parentEdge->visit_count += 1;

        node->parent->subTreeDepth = std::max(node->parent->subTreeDepth, node->subTreeDepth + 1);
        node = node->parent;
    }
}


void TicTacMcts::simulate(TicTacNode* root) {
    TicTacNode* cur = root;
    while (true) {
        if (cur->state.is_terminal()) {
            float value = cur->state.terminal_value();
            backTrace(cur, value);
          
            return ;
        }
        if (!cur->expanded){
            CuHead head = proxy->predict(cur->state);
            auto actions = cur->state.legalActions();
            if (actions.empty()) {
                assert(false);
            }

            cur->expanded = true;

            cur->edges.reserve(actions.size());
            cur->children.resize(actions.size());

            for (size_t i = 0; i < actions.size(); ++i)
            {
                std::unique_ptr<TicTacEdge> edge = std::make_unique<TicTacEdge>(actions[i], head.policy[i]);
                
                cur->children[i] = std::make_unique<TicTacNode>();
                cur->children[i]->parent = cur;
                cur->children[i]->parentEdge = edge.get();

                cur->edges.push_back(std::move(edge));
            }

            backTrace(cur, head.value);
            
            return ;
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
            float u = setting.c_puct * edge->prior * sqrt(total) / (1 + edge->visit_count);
            float score = q + u;

            if (score > best_score) {
                best_score = score;
                best = i;
                selectedEdge = edge;
            }
        }
        assert(selectedEdge != nullptr);
        TicTacNode* child = cur->children[best].get();
        child->state = cur->state.next_state(selectedEdge->action);

        cur = child;
    }

}

void TicTacMcts::search() {

}

void TicTacMcts::selfPlay(TicTacReplayBuffer& replay) {
    std::vector<Tensor> labels;
    std::vector<TicTac> states;

    std::unique_ptr<TicTacNode> cur = std::make_unique<TicTacNode>();
    cur->state = TicTac::initState();

    while (!cur->state.is_terminal()) {
        
        for (int i = 0; i < setting.simulationCount; ++i) {
            simulate(cur.get());
        }
        //std::cout << "Depth: " << cur->state.depth << " visits: ";
        //for (auto& e : cur->edges)
        //    std::cout << e->visit_count << " ";
        //std::cout << std::endl;

        float temperature = cur->state.depth < 10 ? 1 : 0;
        //float temperature = 1;

        //action distribution
        std::vector<float> policy_dis = cur->getPolicyDistribution(temperature);

        std::discrete_distribution<> dist(policy_dis.begin(), policy_dis.end());
        int selectedAction = dist(gen);

        //record state
        Tensor policy(9);
        policy.setData(policy_dis);
        labels.push_back(policy);
        states.push_back(cur->state);

        
        for (int i = 0; i < cur->edges.size(); ++i) {
            if (cur->edges[i]->action == selectedAction) {
                auto child = std::move(cur->children[i]);
                cur = std::move(child);
                cur->parent = nullptr;
                cur->parentEdge = nullptr;
                break;
            }
        }
        //state = state.next_state(selectedAction);

    }
    int winner = - cur->state.player;
    for (int i = states.size()-1 ; i >= states.size()-3; --i) {
        TicTacEntry entry;
        entry.label = labels[i];
        entry.state = states[i].Encode();
        entry.value = (winner == states[i].player) ? 1 : -1;
        replay.entries.push_back(entry);
    }
}

void TicTacMcts::train() {
    //alpha-go-zero:
    //minibatch: 2048
    //checkpoint: 1000 iteration
    InitRandom();

    TicTacReplayBuffer buffer;
    for (int episode = 0; episode < setting.num_episodes; ++episode) {

        selfPlay(buffer);

        if (buffer.entries.size() >= setting.batchSize) {
            std::uniform_int_distribution<size_t> dist(
                0, buffer.entries.size() - 1);

            std::vector<TicTacEntry> miniBatch;
            miniBatch.reserve(setting.miniBatchSize);

            for (size_t i = 0; i < setting.miniBatchSize; ++i) {
                size_t idx = dist(gen);
                miniBatch.push_back(buffer.entries[idx]);
            }

            for (int k = 0; k < setting.trainStepsPerEpisode; ++k) {
                proxy->train(miniBatch);
            }
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

TicTac TicTacMcts::play(const TicTac& state) const {
    CuHead head = proxy->predict(state);
    auto actions = state.legalActions();
    int id = 0;
    float best = -1000;
    for (int i = 0; i < head.policy.size(); ++i) {
        if (head.policy[i] > best) {
            best = head.policy[i];
            id = i;
        }
    }
    Index action = actions[id];
    TicTac result = state.next_state(action);
    return result;
}

void TicTacMcts::InitRandom() {
    std::random_device rd;
    gen.seed(rd());
}

void TicTacMcts::InitRandom(uint32_t seed) {
    gen.seed(seed);
}
