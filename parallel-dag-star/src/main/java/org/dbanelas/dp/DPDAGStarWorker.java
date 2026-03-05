package org.dbanelas.dp;
import com.google.common.util.concurrent.AtomicDoubleArray;
import org.dbanelas.Configuration;
import org.dbanelas.Operator;
import org.dbanelas.PartialGraph;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class DPDAGStarWorker implements Runnable {

    private final int threadId;

    private final PriorityQueue<PartialGraph> openSet;

    // Shared Global State for Termination
    private final AtomicReference<PartialGraph> globalBestSolution;
    private final AtomicBoolean terminateFlag;

    private long pruned = 0;

    private final double latencyThreshold;

    // Termination / Mattern controls
    private final AtomicLong activeTaskCount; // Counts total nodes in flight/processing
    private final AtomicDoubleArray workerMinF; // Dashboard for Main Thread

    public DPDAGStarWorker(int threadId,
                          double latencyThreshold,
                          List<PartialGraph> seedStates,
                          AtomicReference<PartialGraph> globalBestSolution,
                          AtomicBoolean terminateFlag,
                          AtomicLong activeTaskCount,
                          AtomicDoubleArray workerMinF) {
        this.threadId = threadId;
        this.latencyThreshold = latencyThreshold;
        this.globalBestSolution = globalBestSolution;
        this.terminateFlag = terminateFlag;
        this.workerMinF = workerMinF;
        this.activeTaskCount = activeTaskCount;

        this.openSet = new PriorityQueue<>(); // Uses PartialGraph.compareTo which compares f costs.
        this.openSet.addAll(seedStates);
    }

    /**
     * Main worker loop
     */
    @Override
    public void run() {
        System.out.println("Thread " + threadId + " starting with " + openSet.size() + " seed states.");
        try {
            while (!terminateFlag.get()) {

                if (openSet.isEmpty()) {
                    // No work currently, set the best possible cost from here to infinity
                    workerMinF.set(this.threadId, Double.MAX_VALUE);
                    Thread.sleep(1);
                    continue;
                } else {
                    // Update the best possible cost from this worker
                    workerMinF.set(this.threadId, openSet.peek().getFCost());
                }

                // Poll best candidate
                PartialGraph current = openSet.poll();

                // Pruning against Global Best Plan
                PartialGraph bestSoFar = globalBestSolution.get();
                if (bestSoFar != null && current.getFCost() >= bestSoFar.getFCost()){
                    // If the plan is pruned, we decrement the active task count
                    activeTaskCount.decrementAndGet();
                    continue;
                }

                // Check Solution
                assert current != null;
                if (current.isSolution()) {
                    updateGlobalSolution(current);
                    // Also need to decrement tasks, since there is no processing further after the solution
                    activeTaskCount.decrementAndGet();
                    continue;
                }

                // If not solution, expand
                expand(current);
            }
            System.out.println("Thread " + threadId + " terminating. Pruned states: " + pruned);
        } catch (InterruptedException e) {
            System.out.println("Thread " + threadId + " terminating. Pruned states: " + pruned);
            Thread.currentThread().interrupt();
        }
    }

    private void expand(PartialGraph current) {
        // Get candidate operators for expansion
        List<Operator> candidates = current.getCandidateNodes();
        PartialGraph currentBest = globalBestSolution.get();

        for (Operator node : candidates) {
            // Try every valid configuration (site-platform combination) for this operator
            for (Configuration config : node.validConfigurations()) {

                // Create the expanded partial graph
                PartialGraph child = current.expand(node, config);

                if (currentBest != null && child.getFCost() >= currentBest.getFCost()) {
                    // Prune against Global Best Plan
                    continue;
                }

                if (child.getFCost() > latencyThreshold) {
                    // Prune against Latency Threshold
                    pruned++;
                    continue;
                }

                activeTaskCount.incrementAndGet();

                openSet.add(child);
            }
        }
    }

    private void updateGlobalSolution(PartialGraph solution) {
        while (true) {
            PartialGraph currentBest = globalBestSolution.get();

            // If we found a worse solution than what exists, ignore.
            if (currentBest != null && solution.getFCost() >= currentBest.getFCost()) {
                return;
            }

            // CAS loop to update
            if (globalBestSolution.compareAndSet(currentBest, solution)) {
//                System.out.println("Thread " + threadId + " found new best solution: Cost " + solution.getFCost());
                return;
            }
        }
    }


}