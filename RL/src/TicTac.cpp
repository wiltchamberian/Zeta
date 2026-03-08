#include "TicTac.h"
#include "reluLayer.h"
#include "tanhLayer.h"
#include "DnnHead.h"
#include <algorithm>
#include <random>
#include <queue>

std::shared_ptr<mcts::State> TicTac::next_state(int action) const
{
    std::shared_ptr<TicTac> s = std::make_shared<TicTac>();
    *s = *this;

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
        if (s->board[i] == player) {
            if (playerRover == playerId) {
                startPos = i;
                count += 1;
            }
            playerRover++;
        }
        if (s->board[i] == 0) {
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

    s->board[startPos] = 0;
    s->board[endPos] = player;
    
    s->player = -player;
    s->depth = depth+1;
    return s;
}

TicTac TicTac::NextState(int action) const {
    TicTac s;
    for (int i = 0; i < 9; ++i) {
        s.board[i] = board[i];
    }
    s.expanded = false;

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
    s.depth = depth + 1;
    return s;
}

TicTac* TicTac::NextState(int action, mcts::NodePool<TicTac>& pool) const {
    TicTac* s = pool.Alloc();
    for (int i = 0; i < 9; ++i) {
        s->board[i] = board[i];
    }
    s->expanded = false;

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
        if (s->board[i] == player) {
            if (playerRover == playerId) {
                startPos = i;
                count += 1;
            }
            playerRover++;
        }
        if (s->board[i] == 0) {
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

    s->board[startPos] = 0;
    s->board[endPos] = player;

    s->player = -player;
    s->depth = depth + 1;


    return s;
}

uint64_t TicTac::Hash() const {
    uint64_t h = 0;
    for (int i = 0; i < 9; ++i)
    {
        h = h * 3 + (board[i] + 1);
    }

    h = h * 2 + (player == 1 ? 1 : 0);
    return h;
}

void TicTac::UnHash(uint64_t h)  {
    int playerBit = h % 2;
    player = (playerBit == 1 ? 1 : -1);

    h /= 2;

    for (int i = 8; i >= 0; --i)
    {
        int cellEncoded = h % 3;
        h /= 3;

        board[i] = cellEncoded - 1;
    }

    depth = 0;
    expanded = false;
    return;
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


std::shared_ptr<mcts::State> TicTacProxy::createState() {
    std::shared_ptr<mcts::State> st = std::make_shared<TicTac>();
    st->Init();
    return st;
}

CuHead TicTacProxy::predict(const mcts::State* state) {
    CuHead head;
    Tensor input = state->Encode();

    nn->Forward(input);
    policyHead->FetchActivationToCpu();
    Tensor& result = policyHead->distribution;
    valueHead->FetchPredYToCpu();
    Tensor& value = valueHead->predY;

    std::vector<int> legalActions = state->legalActions();

    head.value = value(0, 0);
    for (int i = 0; i < legalActions.size(); ++i) {
        head.policy.push_back(result(0, legalActions[i]));
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

void TicTacProxy::setLearningRate(float rate) {
    nn->SetLearningRate(rate);
}

//build from scratch
void TicTacProxy::createNetwork(float learningRate) {
    totalActionCount = 9;

    nn = std::make_unique<CuNN>();
    nn->SetLearningRate(learningRate);

    //layer
    auto c1 = nn->CreateLayer<Conv2d>(16, 2, 3, 3, Size2D{ 1,1 });
    c1->alpha = 0.0;
    root = c1;

    auto c2 = nn->CreateLayer<Conv2d>(16, 16, 3, 3, Size2D{ 1,1 });
    c2->alpha = 0.0;
    c1->AddLayer(c2);

    //1d conv
    auto c3 = nn->CreateLayer<Conv2d>(1, 16, 1, 1);
    c2->AddLayer(c3);

    auto relu = nn->CreateLayer<CuReluLayer>();
    c3->AddLayer(relu);

    auto fully1 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 9);

    auto cross = nn->CreateLayer<CuSoftmaxCrossEntropyLayer>();
    fully1->AddLayer(cross);

    auto fully2 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 9);
    auto relu2 = nn->CreateLayer<CuReluLayer>();
    fully2->AddLayer(relu2);


    auto fully2_1 = nn->CreateLayer<CuLinearLeakyReluLayer>(9, 1);

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

void TicTacProxy::createNNnetwork(float learningRate,OptimizerType optType) {
    totalActionCount = 9;

    auto dnn = std::make_shared<DNN>();
    dnn->SetLearningRate(learningRate);
    dnn->optimizerType = optType;
    nn = dnn;

    auto c1 = dnn->CreateDnnLayer<DnnConv>(16, 2, 3, 3, Size2D{ 1,1 });
    c1->SetName("c1");
    auto relu1 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu1->SetName("relu1");
    auto c2 = dnn->CreateDnnLayer<DnnConv>(16, 16, 3, 3, Size2D{ 1,1 });
    c2->SetName("c2");
    auto relu2 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu2->SetName("relu2");
    auto c3 = dnn->CreateDnnLayer<DnnConv>(1, 16, 1, 1);
    c3->SetName("c3");
    auto relu = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu->SetName("relu");
    auto fully1 = dnn->CreateDnnLayer<DnnLinear>(9, 9);
    fully1->SetName("fully1");
    auto cross = dnn->CreateDnnLayer<DnnSoftmax>();
    cross->SetName("cross");
    auto fully2 = dnn->CreateDnnLayer<DnnLinear>(9, 9);
    fully2->SetName("fully2");
    auto relu3 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu3->SetName("relu3");
    auto fully2_1 = dnn->CreateDnnLayer<DnnLinear>(9, 1);
    fully2_1->SetName("fully2_1");
    auto tanh = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Tanh);
    tanh->SetName("tanh");
    auto mse = dnn->CreateLayer<CuMseLayer>();
    mse->SetName("mse");
    auto tail = dnn->CreateLayer<CuAddLayer>();
    tail->SetName("tail");

    c1->Add(relu1)->Add(c2)->Add(relu2)->Add(c3)->Add(relu);
    relu->Add(fully1);
    relu->Add(fully2);
    fully1->Add(cross)->Add(tail);
    fully2->Add(relu3)->Add(fully2_1)->Add(tanh)->Add(mse)->Add(tail);

    root = c1;
    policyHead = cross;
    valueHead = mse;

    dnn->AllocDeviceMemory();
}

void TicTacProxy::train(const std::vector<mcts::Entry>& entries) {
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


    Tensor crossLoss = policyHead->FetchLoss();
    Tensor mseLoss = valueHead->FetchLoss();
    Tensor loss = crossLoss + mseLoss;
    loss.print_torch_style("loss:");
    mseLoss.print_torch_style("mse:");
    crossLoss.print_torch_style("cross:");


    nn->Backward();
    nn->Step();
}

float TicTacProxy::train(const Tensor & states, const Tensor & actions, const Tensor & values) {

    policyHead->label = actions;
    valueHead->label = values;

    nn->Forward(states);

    policyHead->BindLabelToDevice();
    valueHead->BindLabelToDevice();

    Tensor crossLoss = policyHead->FetchLoss();
    Tensor mseLoss = valueHead->FetchLoss();
    Tensor loss = crossLoss + mseLoss;

    std::cout << "loss:" << loss(0) << "mse:" << mseLoss(0) << "cross:" << crossLoss(0);
    std::cout << "lr:" << nn->learningRate << std::endl;

    nn->Backward();
    nn->Step();

    return loss(0);
}

mcts::Proxy* TicTacProxy::Clone() const {
    TicTacProxy* proxy = new TicTacProxy();
    proxy->nn = std::unique_ptr<CuNN>(this->nn->Clone());
    proxy->root = this->root->ref;
    proxy->policyHead = dynamic_cast<CuSoftmaxCrossEntropyLayer*>(this->policyHead->ref);
    proxy->valueHead = dynamic_cast<CuMseLayer*>(this->valueHead->ref);
    this->nn->CleanRefs();
    proxy->totalActionCount = totalActionCount;
    return proxy;
}

uint64_t Hash(const TicTac& s)
{
    uint64_t h = 0;
    for (int i = 0; i < 9; ++i)
    {
        h = h * 3 + (s.board[i] + 1);
        // -1 -> 0
        //  0 -> 1
        //  1 -> 2
    }

    // 再加上当前轮到谁
    h = h * 2 + (s.player == 1 ? 1 : 0);
    return h;
}

TicTac UnHash(uint64_t h)
{
    TicTac s;

    int playerBit = h % 2;
    s.player = (playerBit == 1 ? 1 : -1);

    h /= 2;  

    for (int i = 8; i >= 0; --i)
    {
        int cellEncoded = h % 3;
        h /= 3;

        s.board[i] = cellEncoded - 1;
    }

    return s;
}

void TicTacProxy::MinMax(TicTac root) {
    visits.clear();
    values.clear();

    std::vector<TicTac> states;
    states.push_back(root);
    while (!states.empty()) {
        TicTac& t = states.back();
        uint64_t key = Hash(t);
        if (t.is_terminal()) {
            t.expanded = true;
            visits[key] = mcts::VISITIED;
            values[key] = -t.player;
            states.pop_back();
            continue;
        }
        else if (t.expanded == false) {
            if( visits[key] == mcts::VISITIED) {
                //
                states.pop_back();
                continue;
            }
            else {
                auto actions = t.legalActions();
                t.expanded = true;
                visits[key] = mcts::VISITIED;
                TicTac tmp = t;
                for (int k = actions.size() - 1; k >= 0; --k) {
                    TicTac state = tmp.NextState(actions[k]);
                    states.push_back(state);//may make t invalid!
                }
            }
        }
        else {
            auto actions = t.legalActions();
            for (int k = 0; k < actions.size(); ++k) {
                TicTac state = t.NextState(actions[k]);
                auto hash = Hash(state);
                int v = values[hash];
                if (t.player == 1) {
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
        TicTac state = UnHash(it.first);
        state.printState();
        std::cout << "value:" << v << std::endl;

    }

}

std::vector<TicTac> TicTacProxy::ComputeAllStates() {
    std::vector<TicTac> res;
    visits.clear();
    std::queue<TicTac> st;
    TicTac root;
    st.push(root);
    while (!st.empty()) {
        TicTac t = st.front();
        uint64_t h = t.Hash();
        if (visits.find(h) != visits.end()) {
            st.pop();
            continue;
        }
        visits[h] = mcts::VISITIED;
        t.value = 0;
        t.children.clear();
        t.depth = 0;
        res.push_back(t);
        auto actions = t.legalActions();
        for (int i = 0; i < actions.size(); ++i) {
            TicTac state = t.NextState(actions[i]);
            st.push(state);
        }
        st.pop();
    }
    visits.clear();
    return res;
}

//该实现的问题在于共享hash,但是因为存在重复状态，同一个hash
//可能对应不统计节点，修改其他节点的hash会导致出错
int TicTacProxy::Discover(const TicTac& t, int depth) {
    //test board
    TicTac t0 = t;
    int h0 = t.Hash();

    TicTac t1; 
    t1.board = { 1,0,-1,1,1,0,-1,-1,0};
    t1.player = -1;
    int h1 = t1.Hash();

    TicTac t2;
    t2.board = { 1,1,-1,0,1,0,-1,-1,0 };
    t2.player = -1;
    int h2 = t2.Hash();

    TicTac t3;
    t3.board = { 1,1,-1,-1,1,0,0,-1,0 };
    t3.player = 1;
    int h3 = t3.Hash();

    TicTac t4;
    t4.board = { 1,1,-1,-1,0,1,0,-1,0 };
    t4.player = -1;
    int h4 = t4.Hash();

    TicTac t5;
    t5.board = { 1,1,-1,-1,-1,1,0,0,0 };
    t5.player = 1;
    int h5 = t5.Hash();

    TicTac t6;
    t6.board = { 1,1,-1,-1,-1,0,0,0,1 };
    t6.player = -1;
    int h6 = t6.Hash();

    TicTac t7;
    t7.board = { 1,1,-1,-1,-1,-1,0,0,0 };
    t7.player = 1;
    int h7 = t7.Hash();

    visits.clear();
    values.clear();
    std::stack<TicTac> stack;
    stack.push(t);
    while (!stack.empty()) {
        TicTac& t = stack.top();
        uint64_t h = t.Hash();

        //if (values.find(h) != values.end()) {
        //    stack.pop();
        //    continue;
        //}

        if (t.is_terminal()) {
            values[h] = -t.player;
            stack.pop();
            continue;
        }
        if (t.depth == depth) {
            values[h] = 0;
            stack.pop();
            continue;
        }
        
        if (!t.expanded) {
            t.expanded = true;
            auto actions = t.legalActions();
            TicTac tmp = t;
            for (int i = 0; i < actions.size(); ++i) {
                TicTac state  = tmp.NextState(actions[i]);
                state.depth = tmp.depth + 1;
                stack.push(state);
            }
        }
        else if (t.expanded) {

            auto actions = t.legalActions();
            int vs = 0;
            if (t.player == 1) {
                vs = -1;
            }
            else {
                vs = 1;
            }
            if (h == t1.Hash()) {
                std::cout << "abc\n";
            }
            else if (h == t2.Hash()) {
                std::cout << "abc\n";
            }
            else if (h == t3.Hash()) {
                std::cout << "abc\n";
            }
            else if (h == t4.Hash()) {
                std::cout << "abc\n";
            }
            else if (h == t5.Hash()) {
                std::cout << "abc\n";
            }
            else if (h == t6.Hash()) {
                std::cout << "abc\n";
            }
            else if (h == t7.Hash()) {
                std::cout << "abc\n";
            }
            for (int i = 0; i < actions.size(); ++i) {
                TicTac state = t.NextState(actions[i]);
                uint64_t hash = state.Hash();
                if (t.player == 1) {
                    vs = std::max<int>(vs, values[hash]);
                }
                else {
                    vs = std::min<int>(vs, values[hash]);
                }
            }
            values[h] = vs;

            
           
            stack.pop();
        }
    }
    uint64_t hash = t.Hash();
    return values[hash];
}

bool TicTacProxy::UpdateValue(TicTac* t,int h) {
    auto iter = mapping.find(h);
    if (iter != mapping.end()) {
        if (iter->second->depth <= t->depth) {
            t->value = iter->second->value;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

int TicTacProxy::DiscoverAlphaBeta(TicTac* t, int depth) {
    std::stack<TicTac*> stack;
    stack.push(t);
    //mapping.clear();
    while (!stack.empty()) {
        TicTac* t = stack.top();
        int h = t->Hash();
        if (t->is_terminal()) {
            t->value = -t->player;
            stack.pop();

            //mapping[h] = t;

            UpdateAlphaBeta(t, -t->player);
            continue;
        }
        if (t->depth == depth) {
            t->value = 0;
            stack.pop();
            //mapping[h] = t;

            UpdateAlphaBeta(t, 0);
            continue;
        }
        if (!t->expanded) {
            //bool ok = UpdateValue(t, h);
            //if (ok) {
            //    stack.pop();
            //    continue;
            //}

            if (t->parent != nullptr) {
                //pruning
                if (t->parent->alpha >= t->parent->beta) {
                    stack.pop();
                    continue;
                }
                //deliver alpha, beta
                t->alpha = t->parent->alpha;
                t->beta = t->parent->beta;
            }
            

            t->expanded = true;
            auto actions = t->legalActions();
            for (int i = 0; i < actions.size(); ++i) {
                TicTac* state = t->NextState(actions[i], pool);
                t->children.push_back(state);
                state->depth = t->depth + 1;
                state->parent = t;
                stack.push(state);
            }
        }
        else if (t->expanded) {
            //mapping[h] = t;
            auto actions = t->legalActions();
            if (t->player == 1) {
                t->value = -1;
            }
            else {
                t->value = 1;
            }
            for (int i = 0; i < actions.size(); ++i) {
                TicTac* state = t->children[i];
                if (t->player == 1) {
                    t->value = std::max<int>(t->value, state->value);
                }
                else {
                    t->value = std::min<int>(t->value, state->value);
                }
            }
            stack.pop();

            UpdateAlphaBeta(t, t->value);
 
        }
    }
    return t->value;

}

void TicTacProxy::UpdateAlphaBeta(TicTac* t, int v) {
    if (t->parent != nullptr) {
        if (t->player == 1) {
            //parent is min node
            t->parent->beta = std::min<int>(v, t->parent->beta);
        }
        else if (t->player == -1) {
            t->parent->alpha = std::max<int>(t->parent->alpha, v);
        }
        else {
            assert(false);
        }
    }
}





