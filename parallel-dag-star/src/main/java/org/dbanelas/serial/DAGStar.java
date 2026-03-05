package org.dbanelas.serial;

import org.dbanelas.*;

import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

public class DAGStar implements OptimalSearchAlgorithm {

    public enum AggregationMethod {
        SUM,
        MAX
    }

    private final Workflow workflow;
//    private final SignatureAssigner signatureAssigner;
    public static AggregationMethod aggregationMethod = AggregationMethod.MAX;

    private final double latencyThreshold;

    public DAGStar(Workflow workflow, String aggr, double latencyThreshold) {
        this.workflow = workflow;
        this.latencyThreshold = latencyThreshold;
        aggregationMethod = aggr.equals("MAX") ? AggregationMethod.MAX : AggregationMethod.SUM;
    }

    public PartialGraph solve() {
        PriorityQueue<PartialGraph> openSet = new PriorityQueue<>();
        long pruned = 0L;
        long iterations = 0L;
        PartialGraph root = new PartialGraph(workflow);
        openSet.add(root);

        while (!openSet.isEmpty()) {
            // Dequeue the partial graph with the lowest estimated cost
            PartialGraph current = openSet.poll();
            iterations++;

            // Check if we have reached a complete solution
            if (current.isSolution()) {
                System.out.println("Total pruned states due to latency threshold: " + pruned);
                System.out.println("Total iterations: " + iterations);
                return current;
            }

            // Get candidate nodes for expansion
            List<Operator> candidates = current.getCandidateNodes();
            if (candidates.isEmpty()) {
                throw new IllegalStateException("No candidate nodes available for expansion, but solution not found.");
            }

            Operator newOperator = candidates.get(0);
            for (Configuration config : newOperator.validConfigurations()) {
                // Create the new partial graph
                PartialGraph nextPartialGraph = current.expand(newOperator, config);
                if (nextPartialGraph.getFCost() > latencyThreshold) {
                    pruned++;
                    continue; // Prune this partial graph
                }
                openSet.add(nextPartialGraph);
            }
        }

        // Return null if no solution found (death)
        return null;
    }
}