package org.dbanelas;

// Represents a link between two operators in the DAG
public class Edge {
    String fromNodeId;
    String toNodeId;

    public Edge(String fromNodeId, String toNodeId) {
        this.fromNodeId = fromNodeId;
        this.toNodeId = toNodeId;
    }
}
