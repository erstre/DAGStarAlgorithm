package org.dbanelas;
import org.dbanelas.serial.DAGStar;

import java.util.*;

public class PartialGraph implements Comparable<PartialGraph> {

    // Reference to the workflow
    // This serves as the know-it-all for nodes, costs, etc.
    private final Workflow workflow;

    // The actual state: operator name -> configuration assignments
    private Map<String, Configuration> assignments;

    // Set of terminal nodes
    // Used for the expansion and heuristic calculation
    private final Set<String> terminalNodes;

    private final Map<String, Double> terminalNodeRealCosts;

    // Costs
    private final double gCost;
    private final double fCost;

    private final int stateHash;

    // Deep copy constructor
    public PartialGraph(PartialGraph other) {
        this.workflow = other.workflow;
        this.assignments = new HashMap<>(other.assignments);
        this.terminalNodes = null;
        this.terminalNodeRealCosts = null;
        this.gCost = other.gCost;
        this.fCost = other.fCost;
        this.stateHash = other.stateHash;
    }

    // Plan based constructor (using a complete assignment)
    public PartialGraph(Workflow workflow,
                        Map<String, Configuration> initialAssignments) {
        if (initialAssignments.size() != workflow.getNodes().size()) {
            throw new IllegalStateException("Initial assignments do not cover all nodes in the workflow!");
        }
        this.workflow = workflow;
        this.assignments = new HashMap<>(initialAssignments);
        this.terminalNodes = null;
        this.terminalNodeRealCosts = null;
        this.gCost = 0;
        this.fCost = 0;
        this.stateHash = 0;
    }

    /**
     * Constructor for the initial Partial Graph (START node only)
     */
    public PartialGraph(Workflow workflow) {
        this.workflow = workflow;
        this.assignments = new HashMap<>();
        this.terminalNodes = new HashSet<>();
        this.terminalNodeRealCosts = new HashMap<>();

        // Get the START node from the workflow master graph
        String startId = workflow.getStartNodeId();
        Operator startNode = workflow.getNode(startId);

        // Only has the DUMMY configuration
        Configuration startConf = startNode.validConfigurations().get(0);

        // Initialize state
        this.assignments.put(startId, startConf);
        this.terminalNodes.add(startId);
        this.terminalNodeRealCosts.put(startId, 0.0);

        // Calculate initial gCost (0 for START)
        this.gCost = workflow.getCostModel().processingCost(startId, startConf);
        this.stateHash = computeNodeHash(startId, startConf);

        // Calculate initial fCost
        this.fCost = calculateFCost();
    }

    /**
     * Private constructor for expansion (creating a new state from an old one)
     * Will only be used from the expand() method.
     */
    private PartialGraph(Workflow workflow,
                         Map<String, Configuration> assignments,
                         Set<String> terminalNodes,
                         Map<String, Double> terminalNodeRealCosts,
                         double gScore,
                         int stateHash) {
        this.workflow = workflow;
        this.assignments = assignments;
        this.terminalNodes = terminalNodes;
        this.terminalNodeRealCosts = terminalNodeRealCosts;
        this.gCost = gScore;
        this.stateHash = stateHash;
        this.fCost = calculateFCost();
    }


    public List<PartialGraph> expandAll(Operator nextOperator) {
        List<PartialGraph> expansions = new ArrayList<>();

        for (Configuration config : nextOperator.validConfigurations()) {
            PartialGraph expanded = this.expand(nextOperator, config);
            expansions.add(expanded);
        }

        return expansions;
    }

    /**
     * Expands the current partial graph by adding one new node assignment.
     * Returns a NEW immutable PartialGraph.
     * @param nextOperator The node to add to the partial graph
     * @param config The configuration to assign to it
     */
    public PartialGraph expand(Operator nextOperator, Configuration config) {
        // copy the old assignments and terminal nodes
        // these will be modified to produce the new partial graph
        Map<String, Configuration> newAssignments = new HashMap<>(this.assignments);
        Set<String> newTerminalNodes = new HashSet<>(this.terminalNodes);
        Map<String, Double> newTerminalNodeRealCosts = new HashMap<>(this.terminalNodeRealCosts);
        String nextOpId = nextOperator.id();

        // Add new assignment
        newAssignments.put(nextOpId, config);

        // Add the new operator as a terminal one
        newTerminalNodes.add(nextOpId);

        // Remove predecessors from terminal operators, because their Su() is now non empty
        nextOperator.predecessors().forEach(newTerminalNodes::remove);

        // Calculate the processing cost of the new operator
        double nextOperatorAddedCost = workflow.getCostModel().processingCost(nextOpId, config);

        // Calculate the total cost of the newly added operator
        if (DAGStar.aggregationMethod == DAGStar.AggregationMethod.SUM) {
            double sumOfPredRealCosts = 0;
            double sumOfPredCommunicationCosts = 0;
            for (String predId : nextOperator.predecessors()) {
                Configuration predConfig = newAssignments.get(predId);
                sumOfPredRealCosts += newTerminalNodeRealCosts.get(predId);
                sumOfPredCommunicationCosts += workflow.getCostModel().communicationCost(predConfig, config);
            }
            newTerminalNodeRealCosts.put(nextOpId, nextOperatorAddedCost + sumOfPredRealCosts + sumOfPredCommunicationCosts);
        } else {
            double maxRealPlusCommCost = -1.0;
            for (String predId : nextOperator.predecessors()) {
                Configuration predConfig = newAssignments.get(predId);
                double predRealCost = newTerminalNodeRealCosts.get(predId);
                double predCommCost = workflow.getCostModel().communicationCost(predConfig, config);
                maxRealPlusCommCost = Math.max(maxRealPlusCommCost, predRealCost + predCommCost);
            }
            newTerminalNodeRealCosts.put(nextOpId, nextOperatorAddedCost + maxRealPlusCommCost);
        }

        double newGCost = newTerminalNodeRealCosts.values().stream()
                .mapToDouble(Double::doubleValue)
                .max()
                .orElseThrow(() -> new IllegalStateException("There should be at least one terminal node real cost"));

        // Zobrist Update: Incrementally XOR the new assignment's hash into the existing state hash.
        // commutative (Order A->B vs B->A results in the same
        int newHash = this.stateHash ^ computeNodeHash(nextOperator.id(), config);

        return new PartialGraph(workflow, newAssignments, newTerminalNodes, newTerminalNodeRealCosts, newGCost, newHash);
    }

    /**
     * Method to return the list of candidate nodes for expansion from the current partial graph
     */
    public List<Operator> getCandidateNodes() {
        List<Operator> candidates = new ArrayList<>();

        Set<String> visited = new HashSet<>();

        for (String currentGraphOperatorID : this.assignments.keySet()) {
            Operator currentGraphOperator  = workflow.getNode(currentGraphOperatorID);
            for (String succId : currentGraphOperator.successors()) {
                if (visited.contains(succId)) continue;
                if (assignments.containsKey(succId)) continue; // Already visited

                // Check if all the predecessors of the candidate that is evaluated currently
                // are already in the assignments (Rule 2)
                Operator succOperator = workflow.getNode(succId);
                boolean allPredsVisited = true;
                for (String predId : succOperator.predecessors()) {
                    if (!assignments.containsKey(predId)) {
                        allPredsVisited = false;
                        break;
                    }
                }

                if (allPredsVisited) candidates.add(succOperator);
                visited.add(succId);
            }
        }
        return candidates;
    }

    /**
     * Calculate the estimated (f) cost of this partial graph as the real cost (g) plus
     * the minimum heuristic among terminal nodes.
     */
    private double calculateFCost() {
        if (terminalNodes.isEmpty()) return gCost;

        double minHeuristic = terminalNodes.stream()
                .map(t -> workflow.getNode(t).getHeuristicValue())
                .min(Double::compare)
                .orElseThrow(() -> new IllegalStateException("There should be at least one terminal node!"));

        return gCost + minHeuristic;
    }

    public boolean isSolution() {
        return assignments.containsKey(workflow.getEndNodeId());
    }


    /**
     * Computes a hash for the specified node-configuration pair.
     * This is used to build the overall state hash.
     */
    private int computeNodeHash(String operatorId, Configuration config) {
        int h1 = operatorId.hashCode();
        int h2 = config.getID().hashCode();

        // Based on MurmurHash3 finalizer logic.
        int k = h1 ^ (h2 >>> 16) ^ (h2 << 16);
        k ^= k >>> 16;
        k *= 0x85ebca6b;
        k ^= k >>> 13;
        k *= 0xc2b2ae35;
        k ^= k >>> 16;

        return k;
    }

    public void assignOperatorToConfig(String operatorId, Configuration config) {
        this.assignments.put(operatorId, config);
    }


    @Override
    public int compareTo(PartialGraph other) {
        return Double.compare(this.fCost, other.fCost);
    }

    @Override
    public int hashCode() {
        return stateHash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof PartialGraph)) return false;
        PartialGraph other = (PartialGraph) obj;
        // Fast check with hash
        if (this.stateHash != other.stateHash) return false;
        // Full check
        return this.assignments.equals(other.assignments);
    }

    // Getters for external use
    public double getFCost() { return fCost; }
    public double getGCost() { return gCost; }
    public int getStateHash() { return stateHash; }
    public Map<String, Configuration> getAssignments() { return Collections.unmodifiableMap(assignments);}
}