package org.dbanelas.hda;

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

public class HDDAGStar implements OptimalSearchAlgorithm {

    private final Workflow graph;
    private final int numThreads;
    private final long pollIntervalMs = 100; // Frequency of termination checks

    public HDDAGStar(Workflow graph, int numThreads) {
        this.graph = graph;
        this.numThreads = numThreads;
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

        // Init communication inboxes
        List<BlockingQueue<PartialGraph>> allInboxes = new ArrayList<>(numThreads);
        for (int i = 0; i < numThreads; i++) {
            allInboxes.add(new LinkedBlockingQueue<>());
        }

        // Create workers with references
        SignatureAssigner signatureAssigner = new SignatureAssigner(graph);
        List<HDDAGStarWorker> workers = new ArrayList<>();
        for (int i = 0; i < numThreads; i++) {
            workers.add(new HDDAGStarWorker(i, numThreads, allInboxes, globalBestSolution,
                    terminateFlag, signatureAssigner, activeTaskCount, workerMinF));
        }

        // 5. Seed the Search
        // Create the initial partial graph (Just the START node)
        PartialGraph rootState = new PartialGraph(graph);

        // Determine the owner of the root state
        int rootHash = rootState.getStateHash();
        int ownerThread = Math.abs(rootHash % numThreads);

        System.out.println("Seeding search at Thread " + ownerThread);

        // Mark one active task
        activeTaskCount.incrementAndGet();
        allInboxes.get(ownerThread).offer(rootState);

        // Start Threads
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        workers.forEach(executor::submit);

        // Monitor Loop for termination and optimal plan
        PartialGraph finalResult = null;

        try {
            while (!terminateFlag.get()) {
                Thread.sleep(pollIntervalMs);

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