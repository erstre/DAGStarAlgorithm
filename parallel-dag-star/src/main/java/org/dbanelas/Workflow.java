package org.dbanelas;

import org.dbanelas.cost.CostModel;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Workflow {
    private final Map<String, Operator> nodes = new HashMap<>();
    private final CostModel costModel;
    public static final String START_NODE_ID = "START";
    public static final String END_NODE_ID = "END";

    public static final String START_END_DEVICE = "DUMMY_DEVICE";
    public static final String START_END_PLATFORM = "DUMMY_PLATFORM";

    public Workflow(CostModel costModel) {
        this.costModel = costModel;
        addNode(new Operator(START_NODE_ID, List.of(new Configuration(START_END_DEVICE, START_END_PLATFORM))));
        addNode(new Operator(END_NODE_ID, List.of(new Configuration(START_END_DEVICE, START_END_PLATFORM))));
    }

    public void addNode(Operator node) {
        nodes.put(node.id(), node);
    }

    public void addEdge(String fromId, String toId) {
        if (!nodes.containsKey(fromId) || !nodes.containsKey(toId)) {
            throw new IllegalArgumentException("Nodes must exist before adding edges.");
        }
        nodes.get(fromId).addSuccessor(toId);
        nodes.get(toId).addPredecessor(fromId);
    }

    public Operator getNode(String id) { return nodes.get(id); }
    public Map<String, Operator> getNodes() { return Collections.unmodifiableMap(nodes); }
    public CostModel getCostModel() { return costModel; }
    public String getStartNodeId() { return START_NODE_ID; }
    public String getEndNodeId() { return END_NODE_ID; }

    /**
     * Connects START to nodes with no predecessors and END to nodes with no successors.
     * Must be called after all standard nodes/edges are added.
     */
    public void finalizeGraph() {
        // Link START to root operators
        List<String> roots = nodes.values().stream()
                .filter(n -> n.predecessors().isEmpty() && !n.id().equals(START_NODE_ID) && !n.id().equals(END_NODE_ID))
                .map(Operator::id)
                .collect(Collectors.toList());

        for (String root : roots) addEdge(START_NODE_ID, root);

        // Link leaves to END
        List<String> leaves = nodes.values().stream()
                .filter(n -> n.successors().isEmpty() && !n.id().equals(END_NODE_ID) && !n.id().equals(START_NODE_ID))
                .map(Operator::id)
                .collect(Collectors.toList());

        for (String leaf : leaves) addEdge(leaf, END_NODE_ID);

        // Compute Heuristics immediately after finalizing
        computeHeuristics();
    }

    /**
     * Computes the admissible heuristic h(i) for all nodes.
     * Performed in reverse topological order
     */
    public void computeHeuristics() {
        // Reset heuristics
        nodes.values().forEach(n -> n.setHeuristicValue(-1));

        // The heuristic of END is 0
        nodes.get(END_NODE_ID).setHeuristicValue(0.0);

        computeH(nodes.get(START_NODE_ID));
    }

    private double computeH(Operator operator) {
        // Return if already computed
        if (operator.getHeuristicValue() != -1) return operator.getHeuristicValue();

        double maxHeuristic = 0.0;

        for (String successorId : operator.successors()) {
            Operator successor = nodes.get(successorId);

            // Recursive step
            double h_j = computeH(successor);

            // Find min configuration cost for successor j
            double minConfigCost = successor.validConfigurations().stream()
                    .mapToDouble(validConfig -> costModel.processingCost(successorId, validConfig))
                    .min()
                    .orElse(0.0);

            double pathCost = h_j + minConfigCost;
            if (pathCost > maxHeuristic) {
                maxHeuristic = pathCost;
            }
        }

        // Set the computed heuristic to the operator
        operator.setHeuristicValue(maxHeuristic);
        return maxHeuristic;
    }
}