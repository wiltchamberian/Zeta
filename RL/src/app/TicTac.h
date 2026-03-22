#pragma once
#include <memory>
#include <vector>
#include <random>
#include <map>
#include <array>
#include "MctsAlgo.h"
#include "binary.h"

using namespace zeta;

struct Action {
    Action() {}
    Action(char a, char b):startPos(a),endPos(b) {
    }
    char startPos = 0;
    char endPos = 0;
};

class TicTac: public mcts::State {
public:
    TicTac() {
        for (int i = 0; i < 9; ++i) {
            board[i] = 0;
        }
        board[0] = 1;
        board[1] = 1;
        board[2] = 1;
        board[6] = -1;
        board[7] = -1;
        board[8] = -1;
    }
    TicTac(const TicTac& t) {
        for (int i = 0; i < 9; ++i) {
            board[i] = t.board[i];
            player = t.player;
            depth = t.depth;
            expanded = t.expanded;
        }
    }
    std::array<char, 9> board;
    bool expanded = false;
    //alpha beta pruning
    int alpha = -1;
    int beta = 1;
    TicTac* parent = nullptr;
    int value = 0;
    std::vector<TicTac*> children;
    void Init() override {
        for (int i = 0; i < 9; ++i) {
            board[i] = 0;
        }
        board[0] = 1;
        board[1] = 1;
        board[2] = 1;
        board[6] = -1;
        board[7] = -1;
        board[8] = -1;
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
    void FromTensor(const Tensor& result) {
        player = 1;
        const float* d = result.data();
        for (int i = 0; i < 9; ++i) {
            if (d[i] == 1) {
                board[i] = 1;
            }
            else {
                board[i] = 0;
            }
        }
        for (int i = 0; i < 9; ++i) {
            if (d[9 + i] == 1) {
                board[i] = -1;
            }
        }
    }

    bool legal(int i, int j) const;
    bool legalAction(int action) const;
    std::vector<int> legalActions() const override;
    std::shared_ptr<mcts::State> next_state(int action) const;
    TicTac NextState(int action) const;
    TicTac* NextState(int action, mcts::NodePool<TicTac>& pool) const;
    uint64_t Hash() const override;
    void UnHash(uint64_t hash) override;
    std::vector<std::shared_ptr<mcts::State>> permuteStates(const std::vector<double>& policy, std::vector<std::vector<double>>& policies);

    void mirrorBoard(std::array<char, 9>&);
    void rotateBoard(std::array<char, 9>&, int rot);
    std::vector<double> rotatePolicy(const std::vector<double>& policy, int rot);
    std::vector<double> mirrorPolicy(const std::vector<double>& policy);
    static int rotate(int pos, int rot);
    static int mirror(int pos);
    std::pair<int, int> actionToPos(int action);
    int posToAction(int posStart, int posEnd);
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
    bool near_terminal() const {
        return false;
    }

    char character(int p) const {
        static char chs[3] = { 'O','.' ,'X' };
        return chs[p + 1];
    }

    void printState() const {

        std::cout << character(board[6]) << " | ";
        std::cout << character(board[7]) << " | ";
        std::cout << character(board[8]) << std::endl;

        std::cout << character(board[3]) << " | ";
        std::cout << character(board[4]) << " | ";
        std::cout << character(board[5]) << std::endl;

        std::cout << character(board[0]) << " | ";
        std::cout << character(board[1]) << " | ";
        std::cout << character(board[2]) << std::endl;

        if (player == 1) {
            std::cout << "player:1,value:" << value << std::endl;
        }
        else {
            std::cout << "player:-1,value:" << value << std::endl;
        }
    }

    void WriteBinary(BinaryStream& stream) const {
        for (int i = 0; i < board.size(); ++i) {
            stream.write(board[i]);
        }
        stream.write(player);
        stream.write(value);
    }

    void ReadFromBinary(BinaryStream& stream) {
        for (int i = 0; i < board.size(); ++i) {
            board[i] = stream.read<char>();
        }
        player = stream.read<int>();
        value = stream.read<int>();
        value = (player == 1) ? value : (-value);
    }

    static void WriteBinary(std::vector<TicTac>& tictacs, BinaryStream& stream) {
        stream.write<int>(tictacs.size());
        for (int i = 0; i < tictacs.size(); ++i) {
            tictacs[i].WriteBinary(stream);
        }
    }

    static std::vector<TicTac> ReadBinary(BinaryStream& stream) {
        int siz = stream.read<int>();
        std::vector<TicTac> vec(siz);
        for (int i = 0; i < siz; ++i) {
            vec[i].ReadFromBinary(stream);
        }
        return vec;
    }

    float terminal_value() const
    {
        return -1.0f;
    }

    int winner() const override {
        return -player;
    }
};


class TicTacProxy : public mcts::Proxy {
public:
    virtual ~TicTacProxy();
    using StateType = TicTac;
    std::shared_ptr<CuNN> nn = nullptr;

    virtual std::shared_ptr<mcts::State> createState() const override;
    CuHead predict(const mcts::State* state);
    CuHead predict_s(const mcts::State* state) override;
    void setLearningRate(float rate);
    void createNetwork(float learningRate);
    void createNNnetwork(float learningRate, OptimizerType type);
    void train(const std::vector<mcts::Entry>& entries);
    void train(const Tensor& states, const Tensor& actions, const Tensor& values);
    virtual Proxy* Clone() const override;
    virtual void PrintLoss() const override;
    virtual void Save(const std::string& path) const override;
    virtual void Save(BinaryStream& stream) const override;
    virtual void Load(BinaryStream& stream) override;
    CuLayer* root = nullptr;
    CuSoftmaxCrossEntropyLayer* policyHead = nullptr;
    CuMseLayer* valueHead = nullptr;

    //minmax 算法不灵，换一种思路，先求出所有局面
    //然后对每一个局面判断是否是必输走法，方法就是指定固定深度
    std::vector<TicTac> ComputeAllStates();
    int Discover(const TicTac& t, int depth);
    int DiscoverAlphaBeta(TicTac* t, int depth);
    bool UpdateValue(TicTac* t, int h);
    void UpdateAlphaBeta(TicTac* t, int v);
    void MinMax(TicTac state);
    
    std::map<uint64_t, mcts::VisitState> visits;
    std::map<uint64_t, int> values;
    std::unordered_map<uint64_t, TicTac*> mapping;

    mcts::NodePool<TicTac> pool;

    std::mutex mutex;
};

