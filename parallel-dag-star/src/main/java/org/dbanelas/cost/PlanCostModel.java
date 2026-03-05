package org.dbanelas.cost;

import org.dbanelas.PartialGraph;

public interface PlanCostModel {
    double getTotalCost(PartialGraph plan);
    double getPerformanceCost(PartialGraph plan);
    double getMigrationCost(PartialGraph plan);
}
