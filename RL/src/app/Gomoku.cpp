#include "reluLayer.h"
#include "tanhLayer.h"
#include "Gomoku.h"
#include "DnnHead.h"

void Gomoku::Init() {
    for (int i = 0; i < GOMOKU_DIM; ++i) {
        board[i] = 0;
    }
}

Tensor Gomoku::Encode() const {
    Tensor result(1, 2, GOMOKU_W, GOMOKU_W);
    float* d = result.data();
    for (int i = 0; i < GOMOKU_DIM; ++i) {
        if (board[i] == player) {
            d[i] = 1;
        }
        else if (board[i] == -player) {
            d[GOMOKU_DIM + i] = 1;
        }
    }
    return result;
}

void Gomoku::FromTensor(const Tensor& result) {
    player = 1;
    memset(board, 0, sizeof(char) * GOMOKU_DIM);
    const float* d = result.data();
    for (int i = 0; i < GOMOKU_DIM; ++i) {
        if (d[i] == 1) {
            board[i] = 1;
        }
        if (d[i + GOMOKU_DIM] == 1) {
            board[i] = -1;
        }
    }
}

std::vector<int> Gomoku::legalActions() const {
    std::vector<int> actions;
    for (int i = 0; i < GOMOKU_DIM; ++i) {
        if (board[i] == 0) {
            actions.push_back(i);
        }
    }
    return actions;
}

std::shared_ptr < mcts::State > Gomoku::next_state(int action) const {
    std::shared_ptr <Gomoku> st = std::make_shared<Gomoku>();
    *st = *this;
    st->board[action] = player;
    st->player = -st->player;
    st->depth += 1;
    st->lastAction = action;
    return st;
}

bool Gomoku::check_win(int x, int y) const
{
    int p = board[ y * GOMOKU_W + x]; // 当前落子玩家
    if (p == 0) return false;

    const int directions[4][2] = {
        {1, 0},   // 横
        {0, 1},   // 竖
        {1, 1},   // 主对角线
        {1, -1}   // 副对角线
    };

    for (int d = 0; d < 4; ++d)
    {
        int dx = directions[d][0];
        int dy = directions[d][1];

        int count = 1;

        // 正方向
        for (int step = 1; step < 5; ++step)
        {
            int nx = x + dx * step;
            int ny = y + dy * step;

            if (nx < 0 || ny < 0 || nx >= GOMOKU_W || ny >= GOMOKU_W) break;
            if (board[nx + ny * GOMOKU_W] != p) break;

            count++;
        }

        // 反方向
        for (int step = 1; step < 5; ++step)
        {
            int nx = x - dx * step;
            int ny = y - dy * step;

            if (nx < 0 || ny < 0 || nx >= GOMOKU_W || ny >= GOMOKU_W) break;
            if (board[nx + ny * GOMOKU_W] != p) break;

            count++;
        }

        if (count >= 5)
            return true;
    }

    return false;
}

bool Gomoku::is_terminal() const
{
    // 判断是否有人赢
    if (check_win(lastAction%GOMOKU_W, lastAction/GOMOKU_W))
        return true;

    // 判断是否棋盘满
    for (int i = 0; i < GOMOKU_W; ++i)
        for (int j = 0; j < GOMOKU_W; ++j)
            if (board[i + j * GOMOKU_W] == 0)
                return false;

    return true; // 平局
}

float Gomoku::terminal_value() const
{
    int x = lastAction % GOMOKU_W;
    int y = lastAction / GOMOKU_W;

    if (check_win(x, y))
    {
        return -1.0;
    }

    return 0.0f; // 平局
}

int Gomoku::winner() const
{
    int x = lastAction % GOMOKU_W;
    int y = lastAction / GOMOKU_W;

    if (check_win(x, y))
    {
        return -player; // 上一步的人赢
    }

    return 0; // 没人赢
}

char Gomoku::character(int p) const {
    static char chs[3] = { 'O','.' ,'X' };
    return chs[p + 1];
}

void Gomoku::printState() const
{
    for (int y = GOMOKU_W - 1; y >= 0; --y)
    {
        for (int x = 0; x < GOMOKU_W; ++x)
        {
            std::cout << character(board[x + y * GOMOKU_W]) << " ";
        }
        std::cout << std::endl;
    }

    if (player == 1)
        std::cout << "player: human" << std::endl;
    else
        std::cout << "player: AI" << std::endl;
}


std::shared_ptr<mcts::State> GomokuProxy::createState() {
    auto state = std::make_shared<Gomoku>();
    state->Init();
    return state;
}

void GomokuProxy::createNetwork(float learningRate)
{
    totalActionCount = GOMOKU_DIM; // 225

    // 创建 DNN
    auto dnn = std::make_shared<DNN>();
    dnn->SetLearningRate(learningRate);
    nn = dnn;

    // ===== 输入卷积层 =====
    auto c1 = dnn->CreateDnnLayer<DnnConv>(32, 2, 3, 3, Size2D{ 1,1 }); // 输入2通道
    c1->SetName("c1");
    auto relu1 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu1->SetName("relu1");

    auto c2 = dnn->CreateDnnLayer<DnnConv>(64, 32, 3, 3, Size2D{ 1,1 });
    c2->SetName("c2");
    auto relu2 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu2->SetName("relu2");

    auto c3 = dnn->CreateDnnLayer<DnnConv>(64, 64, 3, 3, Size2D{ 1,1 });
    c3->SetName("c3");
    auto relu3 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu3->SetName("relu3");

    auto c4 = dnn->CreateDnnLayer<DnnConv>(64, 64, 3, 3, Size2D{ 1,1 });
    c4->SetName("c4");
    auto relu4 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    relu4->SetName("relu4");

    // ===== policy head =====
    auto p_conv = dnn->CreateDnnLayer<DnnConv>(1, 64, 1, 1); // 输出1通道
    p_conv->SetName("p_conv");
    auto p_relu = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    p_relu->SetName("p_relu");
    auto p_fc = dnn->CreateDnnLayer<DnnLinear>(GOMOKU_DIM, GOMOKU_DIM);
    p_fc->SetName("p_fc");
    auto p_softmax = dnn->CreateDnnLayer<DnnSoftmax>();
    p_softmax->SetName("policy");

    // ===== value head =====
    auto v_conv = dnn->CreateDnnLayer<DnnConv>(1, 64, 1, 1); // 输出1通道
    v_conv->SetName("v_conv");
    auto v_relu = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    v_relu->SetName("v_relu");
    auto v_fc1 = dnn->CreateDnnLayer<DnnLinear>(GOMOKU_DIM, 64);
    v_fc1->SetName("v_fc1");
    auto v_relu2 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    v_relu2->SetName("v_relu2");
    auto v_fc2 = dnn->CreateDnnLayer<DnnLinear>(64, 1);
    v_fc2->SetName("v_fc2");
    auto v_tanh = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Tanh);
    v_tanh->SetName("v_tanh");
    auto v_mse = dnn->CreateLayer<CuMseLayer>();
    v_mse->SetName("value");

    // ===== tail layer（loss融合）=====
    auto tail = dnn->CreateLayer<CuAddLayer>();
    tail->SetName("tail");

    // ===== 连接网络 =====
    c1->Add(relu1)->Add(c2)->Add(relu2)->Add(c3)->Add(relu3)->Add(c4)->Add(relu4);

    // policy 分支
    relu4->Add(p_conv)->Add(p_relu)->Add(p_fc)->Add(p_softmax)->Add(tail);

    // value 分支
    relu4->Add(v_conv)->Add(v_relu)->Add(v_fc1)->Add(v_relu2)->Add(v_fc2)->Add(v_tanh)->Add(v_mse)->Add(tail);

    // ===== 保存 root 和头 =====
    root = c1;
    policyHead = p_softmax;
    valueHead = v_mse;

    // 分配显存
    dnn->AllocDeviceMemory();

    nn = dnn;
}

void GomokuProxy::createNNnetwork(float l, OptimizerType optType) {
    createNetwork(l);
    nn->optimizerType = optType;
}

mcts::Proxy* GomokuProxy::Clone() const {
    GomokuProxy* proxy = new GomokuProxy();
    proxy->version = version;
    proxy->nn = std::unique_ptr<CuNN>(this->nn->Clone());
    proxy->root = this->root->ref;
    proxy->policyHead = dynamic_cast<CuSoftmaxCrossEntropyLayer*>(this->policyHead->ref);
    proxy->valueHead = dynamic_cast<CuMseLayer*>(this->valueHead->ref);
    proxy->totalActionCount = totalActionCount;
    this->nn->CleanRefs();
    return proxy;
}


