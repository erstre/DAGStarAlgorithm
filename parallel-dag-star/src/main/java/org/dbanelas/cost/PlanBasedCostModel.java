package org.dbanelas.cost;


import org.dbanelas.Configuration;
import org.dbanelas.PartialGraph;

import java.util.Random;

public class PlanBasedCostModel implements PlanCostModel {
    private final PartialGraph initialPlan;
    private final Random random = new Random(7);

    public PlanBasedCostModel(PartialGraph initialPlan) {
        this.initialPlan = initialPlan;
    }

    @Override
    public double getTotalCost(PartialGraph plan) {
        return getMigrationCost(plan) + getPerformanceCost(plan);
    }

    @Override
    public double getPerformanceCost(PartialGraph plan) {
        return random.nextDouble();
    }

    @Override
    public double getMigrationCost(PartialGraph plan) {
        return random.nextDouble();
    }
}
