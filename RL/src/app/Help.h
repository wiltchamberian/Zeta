#pragma once
#include <mutex>
#include <random>
#include <stack>
#include "tensor.h"
#include <tuple>

namespace zeta {

    struct Entry {
        Tensor label;
        Tensor state;
        float value;
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
                free(chunk.data); // ÊÍ·Å malloc ÄÚ´æ
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
        static std::tuple<Tensor, Tensor, Tensor> EntriesToTensors(const std::vector<Entry>& entryVec);

        int maxSize = INT_MAX;
        std::vector<Entry> entries;
        std::mt19937 gen;
        std::mutex mutex;
        void setMaxSize(int siz) { maxSize = siz; }
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
}