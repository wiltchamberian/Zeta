#include"Hex.h"
#include "DNN.h"
#include "DnnHead.h"

using namespace zeta;


static uint64_t zobristHex[HEX_DIM][2];
static bool zobristHex_initialized = false;
void InitHexZobrist() {
    if (zobristHex_initialized) return;

    std::mt19937_64 rng(123456); // 固定种子（保证可复现）

    for (int i = 0; i < HEX_DIM; i++) {
        for (int j = 0; j < 2; j++) {
            zobristHex[i][j] = rng();
        }
    }

    zobristHex_initialized = true;
}


HexState::HexState() {
    Init();
}

void HexState::Init() {
    std::memset(board, 0, sizeof(board));
    player = 1;
    depth = 0;
    InitHexZobrist();
    currentHash = 0;
}

int HexState::idx(int x, int y) const {
    return x + y * HEX_WIDTH;
}

bool HexState::inBoard(int x, int y) const {
    return x >= 0 && x < HEX_WIDTH && y >= 0 && y < HEX_WIDTH;
}

Tensor HexState::Encode() const {
    Tensor result(1, 2, HEX_WIDTH, HEX_WIDTH);
    float* d = result.data();
    for (int i = 0; i < HEX_DIM; ++i) {
        if (board[i] == player) {
            d[i] = 1;
        }
        else {
            d[i] = 0;
        }
    }
    for (int i = 0; i < HEX_DIM; ++i) {
        if (board[i] == -player) {
            d[HEX_DIM + i] = 1;
        }
        else {
            d[HEX_DIM + i] = 0;
        }
    }
    return result;
}

void HexState::FromTensor(const Tensor& result) {
    player = 1;
    const float* d = result.data();
    for (int i = 0; i < HEX_DIM; ++i) {
        if (d[i] == 1) {
            board[i] = 1;
        }
        else {
            board[i] = 0;
        }
    }
    for (int i = 0; i < HEX_DIM; ++i) {
        if (d[HEX_DIM + i] == 1) {
            board[i] = -1;
        }
    }

}

// 六方向
const std::vector<std::pair<int, int>> dirs = {
    {-1,0}, {1,0}, {0,-1}, {0,1},
    {-1,-1}, {1,1}
};

std::vector<int> HexState::legalActions() const {
    std::vector<int> actions;
    for (int i = 0; i < HEX_DIM; i++) {
        if (board[i] == 0)
            actions.push_back(i);
    }
    return actions;
}

std::shared_ptr<State> HexState::next_state(int action) const {
    auto s = std::make_shared<HexState>(*this);
    s->board[action] = player;
    s->player = -player;
    s->depth++;

    int piece = (player == 1) ? 0 : 1;
    s->currentHash = this->currentHash;
    s->currentHash ^= zobristHex[action][piece];

    return s;
}

bool HexState::is_terminal() const {
    return winner() != 0 || legalActions().empty();
}

float HexState::terminal_value() const {
    int w = winner();
    if (w == 0) return 0;
    return (w == player ? -1.0f : 1.0f);
}

int HexState::winner() const {
    if (checkWin(1)) return 1;
    if (checkWin(-1)) return -1;
    return 0;
}

uint64_t HexState::Hash() const {
    return currentHash;
}

void HexState::UnHash(uint64_t hash) {
    assert(false);
}

// =========================
// ⭐ 核心：DFS判断连通
// =========================
bool HexState::checkWin(int p) const {
    bool visited[HEX_DIM] = { false };
    std::stack<std::pair<int, int>> st;

    if (p == 1) {
        // bottom -> up
        for (int i = 0; i < HEX_WIDTH; i++) {
            if (board[idx(i, 0)] == 1) {
                st.push({ i, 0 });
                visited[idx(i, 0)] = true;
            }
        }

        while (!st.empty()) {
            auto [x, y] = st.top();
            st.pop();

            if (y == (HEX_WIDTH - 1)) return true; //to up

            for (auto [dx, dy] : dirs) {
                int nx = x + dx, ny = y + dy;
                if (inBoard(nx, ny)) {
                    int id = idx(nx, ny);
                    if (!visited[id] && board[id] == 1) {
                        visited[id] = true;
                        st.push({ nx, ny });
                    }
                }
            }
        }
    }
    else {
        // left -> right
        for (int j = 0; j < HEX_WIDTH; j++) {
            if (board[idx(0, j)] == -1) {
                st.push({ 0, j });
                visited[idx(0, j)] = true;
            }
        }

        while (!st.empty()) {
            auto [x, y] = st.top();
            st.pop();

            if (x == (HEX_WIDTH-1)) return true; //arrive right

            for (auto [dx, dy] : dirs) {
                int nx = x + dx, ny = y + dy;
                if (inBoard(nx, ny)) {
                    int id = idx(nx, ny);
                    if (!visited[id] && board[id] == -1) {
                        visited[id] = true;
                        st.push({ nx, ny });
                    }
                }
            }
        }
    }

    return false;
}

void HexState::printState() const {
    for (int i = 0; i < HEX_WIDTH * 2 - 1; i++) {
        if (i % 2 == 0) {
            for (int j = 0; j < HEX_WIDTH * 2 - 1; ++j) {
                if (j % 2 == 0) {
                    char c = board[idx( j/2, HEX_WIDTH - i / 2 - 1)];
                    if (c == 1) {
                        std::cout << "X";
                    }
                    else if (c == -1) {
                        std::cout << "O";
                    }
                    else {
                        std::cout << "*";
                    }
                }
                else {
                    std::cout << "--";
                }
            }
            std::cout << std::endl;
        }
        else {
            for (int j = 0; j < HEX_WIDTH * 2 - 1; ++j) {
                if (j % 2 == 0) {
                    std::cout << "|";
                }
                else {
                    std::cout << "／";
                }
            }
            std::cout << std::endl;
        }
    }
    std::cout << "\n";
}

std::shared_ptr<State> HexProxy::createState() const {
    std::shared_ptr<State> v = std::make_shared<HexState>();
    return v;
}

CuHead HexProxy::predict(const State* state) {
    CuHead head;
    Tensor input = state->Encode();

    nn->Forward(input);

    valueHead->FetchPredYToCpu();
    Tensor& value = valueHead->predY;

    std::vector<int> legalActions = state->legalActions();

    for (int i = 0; i < legalActions.size(); ++i) {
        head.policy.push_back(value(0, legalActions[i]));
    }

    return head;

}

void HexProxy::train(const Tensor& states, const Tensor& actions) {
    valueHead->label = actions;

    nn->Forward(states);

    valueHead->BindLabelToDevice();

    nn->Backward();
    nn->Step();

    return;
}

void HexProxy::createNetwork(float learningRate) {
    totalActionCount = HEX_DIM; 

    // 创建 DNN
    auto dnn = std::make_shared<DNN>();
    dnn->SetLearningRate(learningRate);
    nn = dnn;

    auto c1 = dnn->CreateDnnLayer<DnnConv>(4, 2, 3, 3, Size2D{ 1,1 });
    auto relu1 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    auto c2 = dnn->CreateDnnLayer<DnnConv>(4, 4, 3, 3, Size2D{ 1,1 });
    auto relu2 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    auto l1 = dnn->CreateDnnLayer<DnnLinear>(4 * HEX_DIM * 2, HEX_DIM * 2);
    auto relu3 = dnn->CreateDnnLayer<DnnAct>(LayerType::Act_Relu);
    auto l2 = dnn->CreateDnnLayer<DnnLinear>(2 * HEX_DIM, HEX_DIM);
    auto mse = dnn->CreateLayer<CuMseLayer>();
    auto output = dnn->CreateLayer<OutputLayer>();

    c1->Add(relu1)->Add(c2)->Add(relu2)->Add(l1)->Add(relu3)->Add(l2)->Add(output);

    valueHead = mse;
    root = c1;

    dnn->AllocDeviceMemory();
}
