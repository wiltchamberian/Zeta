#include "Agent.h"

namespace zeta {

    std::vector<double> Node::getPolicyDistribution(double temperature, int totalActionCount) {
        std::vector<double> distribution(totalActionCount, 0);
        constexpr double alpha = 0.03f;  //eta ~ Dir(alpha)
        constexpr double epsilon = 0.25f;
        if (temperature == 1) {
            double total = 0;
            for (int i = 0; i < edges.size(); ++i) {
                double v = edges[i]->visit_count;
                distribution[edges[i]->action] = v;
                total += v;
            }
            if (total > 0) {
                for (int i = 0; i < distribution.size(); ++i) {
                    distribution[i] /= total;
                }
            }
            return distribution;
        }
        else if (temperature == 0) {
            double total = 0;
            double max = -10000;
            int id = 0;
            for (int i = 0; i < edges.size(); ++i) {
                double v = edges[i]->visit_count;
                if (v > max) {
                    max = v;
                    id = i;
                }
            }
            distribution[edges[id]->action] = 1;
            return distribution;
        }
        else {
            double total = 0;
            for (int i = 0; i < edges.size(); ++i) {
                double v = std::pow(edges[i]->visit_count, 1 / temperature);
                distribution[edges[i]->action] = v;
                total += v;
            }
            if (total > 0) {
                for (int i = 0; i < distribution.size(); ++i) {
                    distribution[i] /= total;
                }
            }
            return distribution;
        }
    }

}