package org.dbanelas;
import java.util.*;

public class Operator {
    private final String id;
    private final List<Configuration> validConfigurations;
    private final List<String> predecessors = new ArrayList<>();
    private final List<String> successors = new ArrayList<>();
    private double heuristicValue = -1.0;

    public Operator(String id, List<Configuration> validConfigurations) {
        this.id = id;
        this.validConfigurations = List.copyOf(validConfigurations);
    }

    public String id() { return id; }
    public List<Configuration> validConfigurations() { return validConfigurations; }

    public List<String> predecessors() { return Collections.unmodifiableList(predecessors); }
    public List<String> successors() { return Collections.unmodifiableList(successors); }

    public double getHeuristicValue() { return heuristicValue; }
    public void setHeuristicValue(double h) { this.heuristicValue = h; }

    // Logic to link nodes
    public void addSuccessor(String nodeId) { this.successors.add(nodeId); }
    public void addPredecessor(String nodeId) { this.predecessors.add(nodeId); }
}