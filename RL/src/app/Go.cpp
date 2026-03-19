#include "Go.h"
#include "reluLayer.h"
#include "tanhLayer.h"
#include <queue>
#include <set>

void Go::Init() {
    memset(board, 0, BOARD_SIZE);
    player = 1;
    depth = 0;
}

Tensor Go::Encode() const {
    Tensor tensor(1, 2, BOARD_WIDTH, BOARD_WIDTH);
    for (int i = 0; i < BOARD_WIDTH; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            int idx = i * BOARD_WIDTH + j;
            if (board[idx] == player) {
                tensor(0, 0, i, j) = 1;
            }
            else if (board[idx] == -player) {
                tensor(0, 1, i, j) = 1;
            }
        }

    }
    return tensor;
}

void Go::FromTensor(const Tensor& result) {
    memset(board, 0, BOARD_SIZE);
    for (int i = 0; i < BOARD_WIDTH; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            int idx = i * BOARD_WIDTH + j;
            if (result(0, 0, i, j) == 1) {
                board[idx] = 1;
            }
            if (result(0, 1, i, j) == 1) {
                board[idx] = -1;
            }
        }

    }
}

std::vector<int> Go::legalActions() const {
    std::vector<int> res;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (board[i] == 0)
            res.push_back(i);
    }
    res.push_back(BOARD_SIZE); // pass
    return res;
}

std::shared_ptr<mcts::State> Go::next_state(int action) const {

    auto st = std::make_shared<Go>(*this); // 复制当前状态
    if (action == BOARD_SIZE) {
        st->continuous_pass += 1;
        st->player = -player;
        st->depth += 1;
        return st;
    }
    else {
        st->continuous_pass = 0;
    }

    int i = action / BOARD_WIDTH;
    int j = action % BOARD_WIDTH;
    int idx = i * BOARD_WIDTH + j;

    st->board[idx] = (char)player;

    // 四个方向
    int dx[4] = { -1, 0, 1, 0 };
    int dy[4] = { 0, -1, 0, 1 };

    for (int k = 0; k < 4; ++k) {
        int ni = i + dx[k];
        int nj = j + dy[k];
        if (ni >= 0 && ni < BOARD_WIDTH && nj >= 0 && nj < BOARD_WIDTH) {
            int nidx = ni * BOARD_WIDTH + nj;
            if (st->board[nidx] == -player) {
                // 检查敌方气数，如果为0 → 提子
                if (st->countLiberties(ni, nj) == 0)
                    st->removeGroup(ni, nj);
            }
        }
    }

    // 检查自杀
    if (st->countLiberties(i, j) == 0) {
        st->removeGroup(i, j); // 如果落子后无气，禁止自杀或特殊处理
    }

    st->player = -player;
    st->depth = depth + 1;
    return st;
}

int Go::countLiberties(int i, int j) const {
    int color = board[i * BOARD_WIDTH + j];
    if (color == 0) return 0;

    std::vector<std::vector<bool>> visited(BOARD_WIDTH, std::vector<bool>(BOARD_WIDTH, false));
    std::stack<std::pair<int, int>> stk;
    stk.push({ i,j });
    visited[i][j] = true;

    int liberties = 0;

    int dx[4] = { -1, 0, 1, 0 };
    int dy[4] = { 0, -1, 0, 1 };

    while (!stk.empty()) {
        auto [x, y] = stk.top(); stk.pop();

        for (int k = 0; k < 4; ++k) {
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_WIDTH)
                continue;

            int idx = nx * BOARD_WIDTH + ny;
            if (board[idx] == 0) {
                liberties++;
            }
            else if (board[idx] == color && !visited[nx][ny]) {
                stk.push({ nx, ny });
                visited[nx][ny] = true;
            }
        }
    }

    return liberties;
}

void Go::removeGroup(int i, int j) {
    int color = board[i * BOARD_WIDTH + j];
    if (color == 0) return;

    std::vector<std::vector<bool>> visited(BOARD_WIDTH, std::vector<bool>(BOARD_WIDTH, false));
    std::stack<std::pair<int, int>> stk;
    stk.push({ i,j });
    visited[i][j] = true;

    int dx[4] = { -1,0,1,0 };
    int dy[4] = { 0,-1,0,1 };

    while (!stk.empty()) {
        auto [x, y] = stk.top(); stk.pop();
        board[x * BOARD_WIDTH + y] = 0; // 清空

        for (int k = 0; k < 4; ++k) {
            int nx = x + dx[k];
            int ny = y + dy[k];
            if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_WIDTH)
                continue;

            int idx = nx * BOARD_WIDTH + ny;
            if (board[idx] == color && !visited[nx][ny]) {
                stk.push({ nx, ny });
                visited[nx][ny] = true;
            }
        }
    }
}

bool Go::is_terminal() const {
    if (continuous_pass == 2) {
        return true;
    }
    if (legalActions().empty()) {
        return true;
    }
    return false;
}

float Go::terminal_value() const {
    int black_score = 0, white_score = 0;
    std::vector<std::vector<bool>> visited(BOARD_WIDTH, std::vector<bool>(BOARD_WIDTH, false));

    int dx[4] = { -1,0,1,0 };
    int dy[4] = { 0,-1,0,1 };

    for (int i = 0; i < BOARD_WIDTH; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            int idx = i * BOARD_WIDTH + j;
            if (board[idx] == 1) black_score++;
            else if (board[idx] == -1) white_score++;
            else if (board[idx] == 0 && !visited[i][j]) {
                // BFS 找连通空点
                std::queue<std::pair<int, int>> q;
                q.push({ i,j });
                visited[i][j] = true;
                std::set<int> neighbors; // 1=黑, -1=白

                int empty_count = 1;

                while (!q.empty()) {
                    auto [x, y] = q.front(); q.pop();

                    for (int k = 0; k < 4; k++) {
                        int nx = x + dx[k], ny = y + dy[k];
                        if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_WIDTH) continue;
                        int nidx = nx * BOARD_WIDTH + ny;
                        if (board[nidx] == 0 && !visited[nx][ny]) {
                            visited[nx][ny] = true;
                            q.push({ nx,ny });
                            empty_count += 1;
                        }
                        else if (board[nidx] == 1) neighbors.insert(1);
                        else if (board[nidx] == -1) neighbors.insert(-1);
                    }
                }

                // 判断空格归属
                if (neighbors.size() == 1) {
                    if (*neighbors.begin() == 1) black_score += empty_count;
                    else white_score += empty_count;
                }
                // 如果同时被黑白包围，可以算中立，不加分
            }
        }
    }

    const float komi = 6.5f;
    float score = black_score - white_score - komi;

    if (score > 0) {
        return (player == 1) ? 1.0 : -1;
    }
    else {
        return (player == -1) ? 1.0 : -1;
    }
}

int Go::winner() const {
    int black_score = 0, white_score = 0;
    std::vector<std::vector<bool>> visited(BOARD_WIDTH, std::vector<bool>(BOARD_WIDTH, false));

    int dx[4] = { -1,0,1,0 };
    int dy[4] = { 0,-1,0,1 };

    for (int i = 0; i < BOARD_WIDTH; ++i) {
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            int idx = i * BOARD_WIDTH + j;
            if (board[idx] == 1) black_score++;
            else if (board[idx] == -1) white_score++;
            else if (board[idx] == 0 && !visited[i][j]) {
                // BFS 找连通空点
                std::queue<std::pair<int, int>> q;
                q.push({ i,j });
                visited[i][j] = true;
                std::set<int> neighbors; // 1=黑, -1=白

                int empty_count = 1;

                while (!q.empty()) {
                    auto [x, y] = q.front(); q.pop();

                    for (int k = 0; k < 4; k++) {
                        int nx = x + dx[k], ny = y + dy[k];
                        if (nx < 0 || nx >= BOARD_WIDTH || ny < 0 || ny >= BOARD_WIDTH) continue;
                        int nidx = nx * BOARD_WIDTH + ny;
                        if (board[nidx] == 0 && !visited[nx][ny]) {
                            visited[nx][ny] = true;
                            q.push({ nx,ny });
                            empty_count += 1;
                        }
                        else if (board[nidx] == 1) neighbors.insert(1);
                        else if (board[nidx] == -1) neighbors.insert(-1);
                    }
                }

                // 判断空格归属
                if (neighbors.size() == 1) {
                    if (*neighbors.begin() == 1) black_score += empty_count;
                    else white_score += empty_count;
                }
                // 如果同时被黑白包围，可以算中立，不加分
            }
        }
    }

    const float komi = 0.0f;
    float score = black_score - white_score - komi;

    if (score > 0) {
        return 1;
    }
    else {
        return -1;
    }
}

char Go::character(int p) const {
    static char chs[3] = { 'O', '.', 'X' };
    return chs[p + 1];  // -1 -> O, 0 -> ., 1 -> X
}

void Go::printState() const {
    std::cout << "  ";
    for (int j = 0; j < BOARD_WIDTH; ++j)
        std::cout << j << " ";
    std::cout << "\n";

    for (int i = BOARD_WIDTH - 1; i >= 0; --i) {
        std::cout << i << " ";
        for (int j = 0; j < BOARD_WIDTH; ++j) {
            int idx = i * BOARD_WIDTH + j;
            std::cout << character(board[idx]) << " ";
        }
        std::cout << "\n";
    }

    if (player == 1)
        std::cout << "player: Black (X)\n";
    else
        std::cout << "player: White (O)\n";

    std::cout << "depth: " << depth << "\n";
}



std::shared_ptr<mcts::State> GoProxy::createState() {
    std::shared_ptr<Go> res = std::make_shared<Go>();
    res->Init();
    return res;
}

void GoProxy::createNetwork(float learningRate) {
    totalActionCount = Go::BOARD_SIZE + 1;

    nn = std::make_unique<CuNN>();
    nn->SetLearningRate(learningRate);

    //layer
    auto c1 = nn->CreateLayer<Conv2d>(32, 2, 3, 3, Size2D{ 1,1 });
    c1->alpha = 0.0;
    root = c1;

    auto r1 = nn->CreateLayer<CuReluLayer>();

    auto c2 = nn->CreateLayer<Conv2d>(32, 32, 3, 3, Size2D{ 1,1 });
    c2->alpha = 0.0;
    auto r2 = nn->CreateLayer<CuReluLayer>();

    auto c3 = nn->CreateLayer<Conv2d>(32, 32, 3, 3, Size2D{ 1,1 });
    c3->alpha = 0.0;
    auto r3 = nn->CreateLayer<CuReluLayer>();

    auto c4 = nn->CreateLayer<Conv2d>(32, 32, 3, 3, Size2D{ 1,1 });
    c4->alpha = 0.0;
    auto r4 = nn->CreateLayer<CuReluLayer>();

    //conv
    auto c_policy = nn->CreateLayer<Conv2d>(2, 32, 1, 1);
    c_policy->alpha = 0.0;

    auto relu = nn->CreateLayer<CuReluLayer>();

    auto fully1 = nn->CreateLayer<Linear>(50, 26);

    auto cross = nn->CreateLayer<CuSoftmaxCrossEntropyLayer>();


    auto c_value = nn->CreateLayer<Conv2d>(1, 32, 1, 1);
    c_value->alpha = 0;

    auto fully2 = nn->CreateLayer<Linear>(25, 64);
    auto relu2 = nn->CreateLayer<CuReluLayer>();

    auto fully2_1 = nn->CreateLayer<Linear>(64, 1);


    auto tanh = nn->CreateLayer<CuTanhLayer>();

    auto mse = nn->CreateLayer<CuMseLayer>();

    auto tail = nn->CreateLayer<CuAddLayer>();

    policyHead = cross;
    valueHead = mse;

    c1->AddLayer(r1)->AddLayer(c2)->AddLayer(r2)->AddLayer(c3)->AddLayer(r3)->AddLayer(c4)->AddLayer(r4);
    r4->AddLayer(c_policy)->AddLayer(relu)->AddLayer(fully1)->AddLayer(cross)->AddLayer(tail);
    r4->AddLayer(c_value)->AddLayer(relu2)->AddLayer(fully2)->AddLayer(tanh)->AddLayer(mse)->AddLayer(tail);


    nn->AllocDeviceMemory();
}

mcts::Proxy* GoProxy::Clone() const {
    GoProxy* proxy = new GoProxy();
    proxy->version = version;
    proxy->nn = std::unique_ptr<CuNN>(this->nn->Clone());
    proxy->root = this->root->ref;
    proxy->policyHead = dynamic_cast<CuSoftmaxCrossEntropyLayer*>(this->policyHead->ref);
    proxy->valueHead = dynamic_cast<CuMseLayer*>(this->valueHead->ref);
    proxy->totalActionCount = totalActionCount;
    this->nn->CleanRefs();
    return proxy;
}
