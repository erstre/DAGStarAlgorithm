package org.dbanelas.cost;

import org.dbanelas.Configuration;

public interface CostModel {
    double processingCost(String operator, Configuration config);
    double communicationCost(Configuration configA, Configuration configB);
}
