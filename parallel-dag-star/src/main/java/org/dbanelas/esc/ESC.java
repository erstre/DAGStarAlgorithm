package org.dbanelas.esc;

import org.dbanelas.*;
import org.dbanelas.cost.PlanBasedCostModel;
import org.dbanelas.cost.PlanCostModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

public class ESC implements OptimalSearchAlgorithm {

    private final Workflow workflow;
    private final int numThreads;
    private final PartialGraph initialPlan;

    public ESC(Workflow workflow, int numThreads) {
        this.workflow = workflow;
        this.numThreads = numThreads;
        Map<String, Configuration> initialAss = new HashMap<>();
        for (String opName : workflow.getNodes().keySet()) {
            initialAss.put(opName, workflow.getNodes().get(opName).validConfigurations().get(0));
        }
        this.initialPlan = new PartialGraph(workflow, initialAss);
    }

    /**
     * Solves the optimal workflow execution problem using Exhaustive Search with Counting.
     * Follows the thread management pattern of HDDAGStar.
     */
    @Override
    public PartialGraph solve() {
        // In ESC, "active tasks" are simply the workers processing their ranges.
        AtomicLong activeWorkerCount = new AtomicLong(0);
        double[] localMinCosts = new double[numThreads];


        long configsPerOp = workflow.getNodes().get("annotation").validConfigurations().size();
        int numOperators = workflow.getNodes().size();
        // Calculate Total Search Space Size: N^M
        long totalSearchSpace = (long) Math.pow(configsPerOp, numOperators);

        // 4. Create Workers
        List<ESCWorker> workers = new ArrayList<>();
        long chunkSize = (totalSearchSpace + numThreads - 1) / numThreads;

        for (int i = 0; i < numThreads; i++) {
            PlanCostModel costModel = new PlanBasedCostModel(initialPlan);
            long startId = i * chunkSize;
            long endId = Math.min((i + 1) * chunkSize, totalSearchSpace);

            if (startId >= totalSearchSpace) break;

            // Workers are initialized with the activeWorkerCount to track completion
            workers.add(new ESCWorker(
                    i,
                    startId,
                    endId,
                    new PartialGraph(initialPlan),
                    costModel,
                    new SignatureAssigner(workflow),
                    configsPerOp
            ));
        }

        // 5. Start Threads
        System.out.println("Starting ESC Search with " + workers.size() + " active workers.");
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        // Increment active count before submitting
        activeWorkerCount.set(workers.size());
        try {
            // This blocks until all workers are finished

            List<Future<Double>> results = executor.invokeAll(workers);
            // Collect local minima
            double globalMinCost = Double.MAX_VALUE;
            for (Future<Double> result : results) {
                globalMinCost = Math.min(globalMinCost, result.get());
            }
            System.out.println("ESC Search completed. Global Minimum Cost: " + globalMinCost);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        } finally {
            executor.shutdown();
        }

        return null;
    }
}