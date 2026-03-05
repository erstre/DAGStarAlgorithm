package org.dbanelas.esc;

import org.dbanelas.*;
import org.dbanelas.cost.PlanCostModel;

import java.util.concurrent.Callable;

public class ESCWorker implements Callable<Double> {

    private final long startId;
    private final long endId;
    private final long base; // number of configurations
    private final PlanCostModel costModel;
    // Local results
    private final PartialGraph initialPlan;
    private final SignatureAssigner signatureAssigner;
    private double localMinCost = Double.MAX_VALUE;
    private int threadId;

    public ESCWorker(int threadId,
                     long startId,
                     long endId,
                     PartialGraph initialPlan,
                     PlanCostModel costModel,
                     SignatureAssigner signatureAssigner,
                     long base) {
        this.startId = startId;
        this.endId = endId;
        this.costModel = costModel;
        this.signatureAssigner = signatureAssigner;
        this.initialPlan = new PartialGraph(initialPlan);
        this.base = base;
        this.threadId = threadId;
    }

    @Override
    public Double call() {

        PartialGraph candidate = new PartialGraph(initialPlan);
        try {
            // Iterate through assigned Plan IDs
            for (long planId = startId; planId < endId; planId++) {

                long tempId = planId;

                for (String opName : candidate.getAssignments().keySet()) {
                    int configIndex = (int) (tempId % base);
                    Configuration config = signatureAssigner.getConfigurationFromSymbol((char) configIndex);
                    candidate.assignOperatorToConfig(opName, config);
                    tempId /= base;
                }

                // 2. Evaluate Cost
                double cost = costModel.getTotalCost(candidate);

                // 3. Local Update
                if (cost < localMinCost) {
                    localMinCost = cost;
                }
            }
        } finally {
            // Signal completion to the monitor loop
            System.out.println("Worker completed range [" + startId + ", " + endId + ")");
            System.out.println("Total of plans evaluated: " + (endId - startId));
        }
        return this.localMinCost;
    }
}