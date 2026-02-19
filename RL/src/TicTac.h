#pragma once
#include <memory>
#include <vector>
#include "CuNN.h"

struct Action {
    Action() {}
    Action(char a, char b):startPos(a),endPos(b) {
    }
    char startPos = 0;
    char endPos = 0;
};


class TicTac {
public:
    char board[9];
    char player = 1;//black, -1:white
    int depth = 0;
    static TicTac initState() {
        TicTac res;
        memset(res.board, 0, 9 *sizeof(char));
        res.board[0] = 1;
        res.board[1] = 1;
        res.board[2] = 1;
        res.board[6] = 1;
        res.board[7] = 1;
        res.board[8] = 1;
        return res;
    }
    Tensor Encode() const {
        Tensor result(1, 2, 3, 3);
        float* d = result.data();
        for (int i = 0; i < 9; ++i) {
            if (board[i] == player) {
                d[i] = 1;
            }
            else {
                d[i] = 0;
            }
        }
        for (int i = 0; i < 9; ++i) {
            if (board[i] == -player) {
                d[9 + i] = 1;
            }
            else {
                d[9 + i] = 0;
            }
        }
        return result;
    }
    bool legal(int i, int j) const;
    bool legalAction(int action) const;
    std::vector<int> legalActions() const;
    TicTac next_state(int action) const;
    bool is_terminal() const
    {
        if (board[4] == (-player)) {
            for (int i = 0; i < 4; ++i) {
                if (board[i] == (-player) && board[8 - i] == (-player)) {
                    return true;
                }
            }
        }
        return false;
    }

    float terminal_value() const
    {
        return 0.0f;
    }
};

//each edge save a prior probaliby P(s,a)
//visit count N(s,a) and action-value Q(s,a)
struct TicTacEdge {
    TicTacEdge(int ac, float policy)
    :action(ac)
    ,prior(policy){

    }
    int action = 0;

    float prior = 0.0f;
    int visit_count = 0;
    float W = 0.0f;

    float Q() {
        if (visit_count == 0) {
            return 0;
        }
        else {
            return W / visit_count;
        }
    }
};

class TicTacNode {
public:
    TicTacEdge* parentEdge = nullptr;
    TicTacNode* parent = nullptr;
    TicTac state;
    bool expanded = false;
    std::vector<std::unique_ptr<TicTacEdge>> edges;
    std::vector<std::unique_ptr<TicTacNode>> children;

    std::vector<float> getPolicyDistribution(float);
};

class TicTacNNProxy {
public:
    CuNN* cunn = nullptr;

    CuHead predict(const TicTac& state);
};

struct TicTacEntry {
    Tensor label;
    Tensor state;
    float value;
};

class TicTacReplayBuffer {
public:
    std::vector<TicTacEntry> entries;
};

class TicTacMcts {
public:
    void backTrace(TicTacNode* n, float value);
    float simulate(TicTacNode* n);
    //一次搜索包含多次mcts simulation
    void search();
    //一次完整对弈，每一步棋包含一次搜索
    void selfPlay(TicTacReplayBuffer& buffer);
    //一次训练
    void train();

    TicTacNNProxy* proxy = nullptr;
};