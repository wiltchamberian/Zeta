//export module core.mctsalgo;

#include <stack>
#include <random>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <condition_variable>
#include <climits>

#include "CuNN.h"
#include "tensor.h"
#include "Agent.h"
#include "Help.h"

using namespace zeta;

namespace mcts {

    enum VisitState {
        UNVISITED = 0,
        VISITIED,
    };


    

    struct Setting {
        int simulationCount = 0;
        int num_episodes = 100;
        int sample_episodes = 20;
        int trainStepsPerEpisode = 20;
        int batchSize = 100;
        int miniBatchSize = 32;
        float c_puct = 1.0f;
        int maxChessLength = 20;
        float dirichletNoise = 0.03f;
        float epsilon = 0.25f;
        float gamma = 1.0f;
        int door = INT_MAX;
        
        int checkpointCount = 50;
        bool useDirichletNoise = true;

        float startTemperature = 100.f;
        float targetTemperature = 0.1f;
        int explorationCount = 10;
    };

    class Mcts {
    public:
        int backTrace(Node* n, float value) const;
        void simulate(Node* n, Proxy* proxy, NodePool<Node>* pool);
        //一次搜索包含多次mcts simulation
        void addDirichletNoise(Node* cur);
        //一次完整对弈，每一步棋包含一次搜索
        int selfPlay(ReplayBuffer& buffer, std::shared_ptr<Proxy> proxy, NodePool<Node>* pool);
        //train_thread
        void train_proxy();
        //一次训练
        void train();
        //实战下棋
        std::shared_ptr<State> play(const std::shared_ptr<State>& state) const;
        //随机下棋
        std::shared_ptr<State> randomPlay(const std::shared_ptr<State>& state) const;
        void InitRandom();
        void InitRandom(uint32_t seed);
        void expand(Node* root, Proxy* proxy);

        void MinMax(Node* state);
        float computeTemperature(int chessCount) const;

        int episode = 0;

        std::unordered_map<int64_t, Node*> transpositionTable;

        std::shared_ptr<Proxy> mctsProxy = nullptr; //for mcts simulation
        Proxy* trainProxy = nullptr;
        std::mt19937 gen;
        BufferQueue bufferQueue;
        ReplayBuffer replayBuffer;
        Setting setting;
        std::atomic<bool> stop; //for training thread to stop
        std::atomic<bool> trainLoopStop = false; //for training loop to stop
        NodePool<Node> pool;

        //for sychronize between traing thread and mcts thread
        std::atomic<int> globalVersion;
        std::mutex mtx;
        std::condition_variable cv;
        std::atomic<bool> endTrain = false;

        //for min max
        std::unordered_map<uint64_t, mcts::VisitState> visits;
        std::unordered_map<uint64_t, int> values;
    };

}



