package org.dbanelas.cost;

import org.dbanelas.Configuration;
import org.dbanelas.Workflow;

import java.util.*;

public class DAGStarCostModel implements CostModel {

    // Operator -> Device -> Platform -> Cost
    private final Map<String, Map<String, Map<String, Double>>> processingCosts = new HashMap<>();

    // Source Device -> Target Device -> Latency
    private final Map<String, Map<String, Double>> communicationCosts = new HashMap<>();

    /**
     * Adds a processing cost entry.
     */
    public void addProcessingCost(String operator, String device, String platform, double cost) {
        processingCosts
                .computeIfAbsent(operator, k -> new HashMap<>())
                .computeIfAbsent(device, k -> new HashMap<>())
                .put(platform, cost);
    }
    /**
     * Adds a network latency entry.
     */
    public void addNetworkLatency(String fromDevice, String toDevice, double latency) {
        communicationCosts
                .computeIfAbsent(fromDevice, k -> new HashMap<>())
                .put(toDevice, latency);
    }

    @Override
    public double processingCost(String operator, Configuration config) {
        if (operator.equals(Workflow.START_NODE_ID) || operator.equals(Workflow.END_NODE_ID)) {
            return 0.0;
        }

        return processingCosts.getOrDefault(operator, Collections.emptyMap())
                .getOrDefault(config.deviceID(), Collections.emptyMap())
                .getOrDefault(config.platformID(), Double.MAX_VALUE);
    }

    @Override
    public double communicationCost(Configuration configA, Configuration configB) {
        if (configA.deviceID().equals(Workflow.START_END_DEVICE) ||
            configA.platformID().equals(Workflow.START_END_PLATFORM) ||
            configB.deviceID().equals(Workflow.START_END_DEVICE) ||
            configB.platformID().equals(Workflow.START_END_PLATFORM)) {
            return 0.0;
        }

        if (configA.deviceID().equals(configB.deviceID())) {
            return 0.0;
        }
        return communicationCosts.getOrDefault(configA.deviceID(), Collections.emptyMap())
                .getOrDefault(configB.deviceID(), Double.MAX_VALUE);
    }
}