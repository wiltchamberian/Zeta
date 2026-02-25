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



int main()
{
    CudaInit();

    CudaGetDeviceProps();

    //test_cnn_linear();
    //test_cnn_conv();
    //test_cnn_tictac();
    //mnist_test();

    mcts::Setting setting;
    setting.simulationCount = 100;
    setting.batchSize = 128;
    setting.miniBatchSize = 128;
    setting.trainStepsPerEpisode = 500;
    setting.num_episodes = 20;
    setting.maxChessLength = 50;

    std::unique_ptr<TicTacProxy> proxy = std::make_unique<TicTacProxy>();
    proxy->createNetwork(0.01);
    proxy->nn->c = 0.0001;

    mcts::Mcts mcts;
    mcts.proxy = proxy.get();
    mcts.setting = setting;
    mcts.train();

    //save

    int d[64];
    bool human = true;
    
    std::unique_ptr<mcts::State> state = std::make_unique<TicTac>();
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
            break;
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

