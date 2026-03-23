#include "Help.h"

namespace zeta {
    std::tuple<Tensor, Tensor, Tensor> ReplayBuffer::EntriesToTensors(const std::vector<Entry>& entryVec) {
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

    std::vector<Entry> ReplayBuffer::sample(size_t batch_size)
    {
        std::vector<Entry> batch;
        batch.reserve(batch_size);

        std::uniform_int_distribution<size_t> dist(0, entries.size() - 1);

        for (size_t i = 0; i < batch_size; ++i)
        {
            batch.push_back(entries[dist(gen)]);
        }
        return batch;
    }

    void ReplayBuffer::shuffle() {
        std::shuffle(entries.begin(), entries.end(), gen);
    }
}