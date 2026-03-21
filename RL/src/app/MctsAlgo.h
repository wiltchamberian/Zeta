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

using namespace zeta;

namespace mcts {

    enum VisitState {
        UNVISITED = 0,
        VISITIED,
    };


    struct Entry {
        Tensor label;
        Tensor state;
        float value;
    };

    class State {
    public:
        // 默认构造
        State() = default;

        // 拷贝构造
        State(const State& other)
            : player(other.player),
            depth(other.depth)
        {
        }

        // 移动构造
        State(State&& other) noexcept
            : player(other.player),
            depth(other.depth)
        {
        }

        // 拷贝赋值
        State& operator=(const State& other)
        {
            if (this != &other)
            {
                player = other.player;
                depth = other.depth;
            }
            return *this;
        }

        // 移动赋值
        State& operator=(State&& other) noexcept
        {
            if (this != &other)
            {
                player = other.player;
                depth = other.depth;
            }
            return *this;
        }

        // 必须是 virtual 析构函数（关键）
        virtual ~State() = default;
        
        virtual void Init() {}
        virtual Tensor Encode() const { return Tensor(1);  }
        virtual void FromTensor(const Tensor& result) {}
        virtual std::vector<int> legalActions() const { return {};  }
        virtual std::shared_ptr<State> next_state(int action) const { return nullptr; }
        virtual bool is_terminal() const { return false; }
        virtual float terminal_value() const { return 0; }
        virtual void printState() const {}
        virtual int winner() const {
            return 0;
        }
        virtual uint64_t Hash() const { return 0;  }
        virtual void UnHash(uint64_t hash) { return;  }
        virtual std::vector<std::shared_ptr<mcts::State>> permuteStates(const std::vector<double>& policy, std::vector<std::vector<double>>& policies) { return {}; }

        int player = 1;
        int depth = 0;
    };

    class Proxy {
    public:
        virtual ~Proxy() {}
        virtual std::shared_ptr<mcts::State> createState() const { return nullptr; }
        virtual void Save(const std::string& path) const {}
        virtual CuHead predict(const mcts::State* state) { return CuHead(); }
        virtual CuHead predict_s(const mcts::State* state) { return CuHead(); }
        virtual void setLearningRate(float rate) {}
        virtual void createNetwork(float learningRate) {}
        virtual void train(const std::vector<mcts::Entry>& entries) {}
        virtual void train(const Tensor& states, const Tensor& actions, const Tensor& values) {}
        virtual Proxy* Clone() const = 0;
        virtual void PrintLoss() const {}
        int totalActionCount = 0;
        int version = 0;
    };

    //each edge save a prior probaliby P(s,a)
    //visit count N(s,a) and action-value Q(s,a)
    struct Edge {
        Edge(int ac, float policy)
            :action(ac)
            , prior(policy) {

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


    template<class _T>
    class NodePool {
    public:
        using Ele = _T;
        ~NodePool() {
            Clear();

        }

        void Clear() {
            for (auto& chunk : chunks) {
                free(chunk.data); // 释放 malloc 内存
            }
            chunks.clear();
        }

        struct Chunk {
            int siz = 0; //capacity
            Ele* data = nullptr;
            std::vector<Ele*> free_slots;
        };

        NodePool(int siz = 1000) : chunk_size(siz) {

        }

        void resize(int siz) {
            Clear();
            chunk_size = siz;
        }

        void Free(_T* t) {
            for (int i = 0; i < chunks.size(); ++i) {
                if (t >= chunks[i].data && t < chunks[i].data + chunks[i].siz - chunks[i].free_slots.size()) {
                    chunks[i].free_slots.push_back(t);
                    t->~_T();
                    break;
                }
            }
        }

        void FreeTree(_T* root) {
            std::stack<_T*> stk;
            stk.push(root);

            while (!stk.empty()) {
                _T* node = stk.top();
                stk.pop();

                for (auto child : node->children) {
                    if (child != nullptr)
                        stk.push(child);
                }
                node->children.clear(); 
                Free(node);
            }
        }

        _T* Alloc() {
            for (int i = 0; i < chunks.size(); ++i) {
                if (!chunks[i].free_slots.empty()) {
                    _T* res = chunks[i].free_slots.back();
                    new(res) _T();
                    chunks[i].free_slots.pop_back();
                    return res;
                }
            }
            Chunk newChunk;
            newChunk.siz = chunk_size;
            newChunk.data = (_T*)malloc(sizeof(_T) * chunk_size);
            newChunk.free_slots.resize(chunk_size);
            for (int i = 0; i < chunk_size; ++i) {
                newChunk.free_slots[i] = newChunk.data + chunk_size - i - 1;
            }
            chunks.push_back(newChunk);

            _T* res = chunks.back().free_slots.back();
            new(res) _T();
            chunks.back().free_slots.pop_back();
            return res;
        }

        std::vector<Chunk> chunks;
        int chunk_size = 0;
    };

    //alloc on memory pool
    class Node {
    public:
        friend class NodePool<Node>;
        ~Node() {
            for (int i = 0; i < children.size(); ++i) {
                if (children[i] != nullptr) {
                    delete children[i];
                }
            }
        }
    private:
        Node() {

        }
        
    public:
        Edge* parentEdge = nullptr;
        Node* parent = nullptr;
        std::shared_ptr<State> state;
        bool expanded = false;
        std::vector<std::unique_ptr<Edge>> edges;
        std::vector<Node*> children;

        std::vector<double> getPolicyDistribution(double, int totalActionCount);
        int subTreeDepth = 0;
    };

    class ReplayBuffer {
    public:
        ReplayBuffer() {
            std::random_device rd;
            gen.seed(rd());
        }
        ReplayBuffer(const ReplayBuffer& buf) {
            entries = buf.entries;
        }
        ReplayBuffer(ReplayBuffer&& buf) {
            entries = std::move(buf.entries);
        }
        ReplayBuffer& operator = (ReplayBuffer&& buf) {
            entries = std::move(buf.entries);
            return *this;
        }
        ReplayBuffer clone_s() {
            std::lock_guard<std::mutex> lock(mutex);
            ReplayBuffer output;
            output.entries = entries;
            return output;
        }
        
        void lock() {
            mutex.lock();
        }
        void unlock() {
            mutex.unlock();
        }
        void append_s(const std::vector<Entry>& entry) {
            std::lock_guard<std::mutex> lock(mutex);
            entries.insert(entries.end(), entry.begin(), entry.end());
            if (entries.size() > maxSize) {
                size_t remove_count = entries.size() - maxSize;
                entries.erase(entries.begin(), entries.begin() + remove_count);
            }
        }
        std::vector<Entry> uniform_sample(int count, int sample_count) {
            std::vector<Entry> res;
            mutex.lock();
            if (entries.size() < sample_count) {
                mutex.unlock();
                return {};
            }
            std::uniform_int_distribution<size_t> dist(
                0, sample_count - 1);
            for (int i = 0; i < count; ++i) {
                res.push_back(entries[dist(gen)]);
            }
            mutex.unlock();
            return res;
        }
        static std::tuple<Tensor, Tensor, Tensor> EntriesToTensors(const std::vector<Entry>& entryVec) {
            Tensor label(entryVec.size(), entryVec[0].label.shape[0]);
            Tensor values(entryVec.size());
            Tensor states(entryVec.size(), entryVec[0].state.shape[1], entryVec[0].state.shape[2], entryVec[0].state.shape[3]);
            for (int i = 0; i < entryVec.size(); ++i) {
                label[i].copy(entryVec[i].label);
                values(i) = entryVec[i].value;
                states[i].copy(entryVec[i].state);
            }
            return { states, label, values };
        }
        int maxSize = INT_MAX;
        std::vector<Entry> entries;
        std::mt19937 gen;
        std::mutex mutex;
        void setMaxSize(int siz) { maxSize = siz;  }
        std::vector<Entry> sample(size_t batch_size);
        void shuffle();
    };

    class BufferQueue {
    public:
        void append_s(const ReplayBuffer& buf) {
            mutex.lock();
            buffers.push_back(buf);
            mutex.unlock();
        }
        void append_s(ReplayBuffer&& buf) {
            mutex.lock();
            buffers.push_back(buf);
            mutex.unlock();
            buf.entries.clear();
        }
        ReplayBuffer pop_s() {
            ReplayBuffer res;
            mutex.lock();
            if (buffers.size() > 0) {
                res = std::move(buffers.back());
                buffers.pop_back();
            }
            mutex.unlock();
            return res;
        }
        std::mutex mutex;
        std::vector<ReplayBuffer> buffers;
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
        
        int checkpointCount = 50;
        bool useDirichletNoise = true;

        float startTemperature = 100.f;
        float targetTemperature = 0.1f;
        int explorationCount = 10;
    };

    class Mcts {
    public:
        int backTrace(Node* n, float value) const;
        void simulate(Node* n, Proxy* proxy, NodePool<mcts::Node>* pool);
        //一次搜索包含多次mcts simulation
        void addDirichletNoise(Node* cur);
        //一次完整对弈，每一步棋包含一次搜索
        int selfPlay(ReplayBuffer& buffer, std::shared_ptr<Proxy> proxy, NodePool<mcts::Node>* pool);
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



