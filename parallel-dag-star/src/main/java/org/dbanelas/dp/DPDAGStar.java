package org.dbanelas.dp;

import com.google.common.util.concurrent.AtomicDoubleArray;
import org.dbanelas.OptimalSearchAlgorithm;
import org.dbanelas.PartialGraph;
import org.dbanelas.SignatureAssigner;
import org.dbanelas.Workflow;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class DPDAGStar implements OptimalSearchAlgorithm {

    private final Workflow graph;
    private final int numThreads;
    private final double latencyThreshold;
    private final long pollIntervalMs = 100; // Frequency of termination checks

    public DPDAGStar(Workflow graph, int numThreads, double latencyThreshold) {
        this.graph = graph;
        this.numThreads = numThreads;
        this.latencyThreshold = latencyThreshold;
    }

    /**
     * Solves the optimal workflow execution problem.
     * @return The optimal PartialGraph (ANOC) or null if no solution exists.
     */
    public PartialGraph solve() {
        // Init global shared state
        AtomicReference<PartialGraph> globalBestSolution = new AtomicReference<>(null);
        AtomicBoolean terminateFlag = new AtomicBoolean(false);

        // Init termination mechamism
        AtomicLong activeTaskCount = new AtomicLong(0);
        AtomicDoubleArray workerMinF = new AtomicDoubleArray(numThreads);


        // In this version, we seed the threads with plans that fix the first operator
        PartialGraph rootState = new PartialGraph(graph);
        List<PartialGraph> seedStates = new ArrayList<>(rootState.expandAll(rootState.getCandidateNodes().get(0)));

        // Create workers with references
        List<DPDAGStarWorker> workers = new ArrayList<>();

        int totalSeeds = seedStates.size();
        int baseChunkSize = totalSeeds / numThreads;
        int remainder = totalSeeds % numThreads;
        int currentIndex = 0;

        for (int i = 0; i < numThreads; i++) {
            // If there is a remainder, give one extra plan to the first 'remainder' threads
            int count = baseChunkSize + (i < remainder ? 1 : 0);
            int endIndex = currentIndex + count;

            // Create a specific sublist for this worker (defensively copied to a new ArrayList)
            List<PartialGraph> workerSeeds = new ArrayList<>(seedStates.subList(currentIndex, endIndex));

            workers.add(new DPDAGStarWorker(i, latencyThreshold, workerSeeds, globalBestSolution,
                    terminateFlag, activeTaskCount, workerMinF));

            // Move the start pointer for the next iteration
            currentIndex = endIndex;
        }

        // Start Threads
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        workers.forEach(executor::submit);

        // Monitor Loop for termination and optimal plan
        PartialGraph finalResult = null;

        try {
            while (!terminateFlag.get()) {
                Thread.sleep(pollIntervalMs);
//                for (int i = 0; i < numThreads; i++) {
//                    System.out.print("| Worker " + i + ": " + workerMinF.get(i));
//                }
//                System.out.println();

                // Checks termination due to search space exhaustion (this probably will never happen)
                if (activeTaskCount.get() == 0L) {
//                    System.out.println("Search space exhausted.");
                    finalResult = globalBestSolution.get();
                    terminateFlag.set(true);
                    break;
                }

                // Checks termination due to optimality => Global best plan is better than the best possible
                // of each worker
                PartialGraph bestSoFar = globalBestSolution.get();
                if (bestSoFar != null) {
                    // Fast Read of Dashboard
                    double globalMinF = Double.MAX_VALUE;
                    for (int i = 0; i < numThreads; i++) {
                        globalMinF = Math.min(globalMinF, workerMinF.get(i));
                    }

                    if (bestSoFar.getFCost() <= globalMinF) {
//                        System.out.println("Optimal Solution Confirmed! Cost: " + bestSoFar.getFCost());
                        finalResult = bestSoFar;
                        terminateFlag.set(true);
                        break;
                    }
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            // Shutdown
            terminateFlag.set(true);
            executor.shutdownNow();
        }

        return finalResult;
    }
}