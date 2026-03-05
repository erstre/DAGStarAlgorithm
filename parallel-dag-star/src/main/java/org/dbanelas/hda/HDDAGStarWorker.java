package org.dbanelas.hda;

import com.google.common.util.concurrent.AtomicDoubleArray;
import org.dbanelas.Configuration;
import org.dbanelas.Operator;
import org.dbanelas.PartialGraph;
import org.dbanelas.SignatureAssigner;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class HDDAGStarWorker implements Runnable {

    private final int threadId;
    private final int totalThreads;

    // Communication channels
    private final BlockingQueue<PartialGraph> inbox;
    private final List<BlockingQueue<PartialGraph>> allInboxes;

    // Local Search Structures
    private final PriorityQueue<PartialGraph> openSet;

    // Set that stores the visited state signatures
    private final Set<String> visited;

    // Shared Global State for Termination
    private final AtomicReference<PartialGraph> globalBestSolution;
    private final AtomicBoolean terminateFlag;

    // Termination / Mattern controls
    private final AtomicLong activeTaskCount; // Counts total nodes in flight/processing
    private final AtomicDoubleArray workerMinF; // Dashboard for Main Thread

    private final SignatureAssigner signatureAssigner;

    public HDDAGStarWorker(int threadId,
                           int totalThreads,
                           List<BlockingQueue<PartialGraph>> allInboxes,
                           AtomicReference<PartialGraph> globalBestSolution,
                           AtomicBoolean terminateFlag,
                           SignatureAssigner signatureAssigner,
                           AtomicLong activeTaskCount,
                           AtomicDoubleArray workerMinF) {
        this.threadId = threadId;
        this.totalThreads = totalThreads;
        this.allInboxes = allInboxes;
        this.inbox = allInboxes.get(threadId);
        this.globalBestSolution = globalBestSolution;
        this.terminateFlag = terminateFlag;
        this.signatureAssigner = signatureAssigner;
        this.openSet = new PriorityQueue<>(); // Uses PartialGraph.compareTo which compares f costs.
        this.visited = new HashSet<>();

        this.activeTaskCount = activeTaskCount;
        this.workerMinF = workerMinF;
    }

    /**
     * Main worker loop
     */
    @Override
    public void run() {
        try {
            while (!terminateFlag.get()) {
                // Get all incoming states into the open set
                processInbox();

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

                // Prune against the closed set
                String planSignature = signatureAssigner.generateSignature(current.getAssignments());
                if (visited.contains(planSignature)) {
                    // Again, if the plan is pruned, we need to decrement the active task count
                    activeTaskCount.decrementAndGet();
                    continue;
                }

                // Check Solution
                if (current.isSolution()) {
                    updateGlobalSolution(current);
                    // Also need to decrement tasks, since there is no processing further after the solution
                    activeTaskCount.decrementAndGet();
                    continue;
                }

                // If not solution, expand
                expandAndRoute(current);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Method that transfers all states from the inbox to the local OpenSet
     * with an early Closed Set check
     */
    private void processInbox() {
        // Drain everything currently in the inbox to the OpenSet
        List<PartialGraph> incoming = new ArrayList<>();
        inbox.drainTo(incoming);

        for (PartialGraph state : incoming) {
            // If we already closed this state with a better G, there is no point adding it.
            if (!visited.contains(signatureAssigner.generateSignature(state.getAssignments())))
                openSet.add(state);
        }
    }

    private void expandAndRoute(PartialGraph current) {
        // Get candidate operators for expansion
        List<Operator> candidates = current.getCandidateNodes();

        for (Operator node : candidates) {
            // Try every valid configuration (site-platform combination) for this operator
            for (Configuration config : node.validConfigurations()) {

                // Create the expanded partial graph
                PartialGraph child = current.expand(node, config);

                // Get the target thread for this child state
                int hash = child.getStateHash();
                int targetThreadId = Math.abs(hash % totalThreads);

                activeTaskCount.incrementAndGet();

                // If this thread is the owner, add to local OpenSet with Closed Set check directly
                // No reason to send to inbox
                if (targetThreadId == this.threadId) {
                    if (!visited.contains(signatureAssigner.generateSignature(child.getAssignments())))
                        openSet.add(child);
                } else {

                    // Send to the target thread's inbox
                    allInboxes.get(targetThreadId).add(child);
                }
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

    public Double getBestOpenF() {
        PartialGraph best = openSet.peek();
        return (best != null) ? best.getFCost() : Double.POSITIVE_INFINITY;
    }

    public boolean isOpenSetEmpty() {
        return openSet.isEmpty();
    }
}