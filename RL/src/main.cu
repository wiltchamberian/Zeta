#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "NeuralNetwork.h"
#include <thread>
#include <chrono>

//#include <cudnn.h>
#include "test.h"
#include "cu_tool.h"
#include "cnn_test.h"
#include "ThreeTac.h"
#include "mnist.h"
#include "LeNet.h"
#include "Go.h"

void GenAllTicTacSamples() {
    TicTacProxy tp;
    TicTac b;
    b.board[0] = 0;
    b.board[1] = 1;
    b.board[2] = 1;
    b.board[3] = -1;
    b.board[4] = 1;
    b.board[5] = 0;
    b.board[6] = -1;
    b.board[7] = 0;
    b.board[8] = -1;
    b.player = 1;

    auto start = std::chrono::high_resolution_clock::now();
    int c = tp.DiscoverAlphaBeta(&b, 10);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << " ms\n";

    std::vector<TicTac> states = tp.ComputeAllStates();
    std::vector<int> values(states.size(), 0);
    for (int i = 0; i < states.size(); ++i) {
        int value = tp.DiscoverAlphaBeta(&states[i], 10);
        tp.pool.Clear();
        //states[i].printState();
    }
    //save to file
    BinaryStream bs;
    TicTac::WriteBinary(states, bs);
    bs.saveToFile("tictacs.bin");
}


int main()
{
    CudaInit();

    CudaGetDeviceProps();

    //test_cnn_linear();
    //test_dnn_linear();
    //test_cnn_conv();
    //test_dnn_conv();
    //test_cnn_tictac();
    //test_blaslt();
    mnist_test();

    BinaryStream bs;
    bs.loadFromFile("tictacs.bin");
    std::vector<TicTac> res = TicTac::ReadBinary(bs);
    for (int i = 0; i < res.size(); ++i) {
        res[i].printState();
    }

    


    mcts::Setting setting;
    setting.simulationCount = 100;
    setting.batchSize = 128;
    setting.miniBatchSize = 128;
    setting.trainStepsPerEpisode = 100;
    setting.num_episodes = 200;
    setting.sample_episodes = 20;
    setting.maxChessLength = 50;
    setting.checkpointCount = 1;
    setting.useDirichletNoise = true;
    
    setting.targetTemperature = 0.1;
    setting.explorationCount = 5; // minus means not use

    std::shared_ptr<ThreeTacProxy> proxy = std::make_shared<ThreeTacProxy>();
    proxy->createNetwork(0.01);
    proxy->nn->c = 0.0001;

    setting.dirichletNoise = 10.0f / proxy->totalActionCount;

    mcts::Mcts mcts;
    mcts.mctsProxy = proxy;
    mcts.trainProxy = proxy->Clone();
    mcts.setting = setting;
    mcts.run();

    //save

    int d[64];
    bool human = true;
    
    std::shared_ptr<mcts::State> state = std::make_shared<ThreeTacState>();
    state->Init();
    std::cout << "human first? (1:human 0:AI)\n";
    std::cin >> d[0];
    if (d[0] > 0) {
        std::cout << "huamn first!\n";
        human = true;
    }
    else {
        std::cout << "Ai first!\n";
        human = false;
        state->player = -1;
    }
    state->printState();
    //start play game
    while (true) {
        if (state->is_terminal()) {
            if (state->player == 1) {
                std::cout << "winner is AI!\n";
            }
            else {
                std::cout << "winner is human!\n";
            }
            std::cout << std::endl;
            std::cout << "new game!" << std::endl;
            state->Init();

            std::cout << "human first? (1:human 0:AI)\n";
            std::cin >> d[0];
            if (d[0] > 0) {
                std::cout << "huamn first!\n";
                human = true;
            }
            else {
                std::cout << "Ai first!\n";
                human = false;
                state->player = -1;
            }
            state->printState();
        }
        if (human) {
            std::cout << "please choose your action!\n";
            auto legals = state->legalActions();
            while (true) {
                std::cin >> d[0];
                bool bingo = false;
                for (int k = 0; k < legals.size(); ++k) {
                    if (d[0] == legals[k]) {
                        bingo = true;
                        break;
                    }
                }
                if (bingo) {
                    break;
                }
            }
            
            state = state->next_state(d[0]);
            state->printState();
            human = !human;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else {
            state = mcts.play(state);
            state->printState();
            human = !human;
        }

    }
    std::cin >> d[0];

    return 0;
}

void HumanSupervised() {

}

