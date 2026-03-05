package org.dbanelas.parsers;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.dbanelas.Configuration;
import org.dbanelas.cost.DAGStarCostModel;
import org.dbanelas.Operator;
import org.dbanelas.Workflow;

import java.io.IOException;
import java.util.*;

public class WorkflowParser {

    private static final ObjectMapper mapper = new ObjectMapper();

    /**
     * @return A fully constructed Workflow.
     * @throws IOException If JSON parsing fails.
     */
    public static Workflow parse(String workflowJson,
                                 String networkJson,
                                 String dictionaryJson,
                                 DAGStarCostModel costModel) throws IOException {
        // Parse the Dictionary to find where each operator is allowed to run
        // OperatorName (classKey) -> Set of DeviceIDs (sites)
        Map<String, Set<String>> operatorToValidDevices = parseDictionary(dictionaryJson);
        Map<String, List<Configuration>> configurationsPerDevice = parseNetwork(networkJson);

        // Create the base workflow object
        Workflow workflow = new Workflow(costModel);

        // Parse Topology JSON
        JsonNode workflowRoot = mapper.readTree(workflowJson);

        // Parse Operators
        JsonNode operatorsNode = workflowRoot.get("operators");
        if (operatorsNode != null && operatorsNode.isArray()) {
            for (JsonNode opNode : operatorsNode) {
                String opName = opNode.get("name").asText();

                List<Configuration> validConfigs = new ArrayList<>();
                Set<String> validDevices = operatorToValidDevices.get(opName);

                assert validDevices != null : "Operator " + opName + " not found in dictionary.";
                assert !validDevices.isEmpty() : "Operator " + opName + " has no valid sites in dictionary.";

                // Add all configurations from valid devices
                validDevices.forEach(d -> validConfigs.addAll(configurationsPerDevice.get(d)));

                workflow.addNode(new Operator(opName, validConfigs));
            }
        }

        // B. Parse Connections (Edges)
        JsonNode connectionsNode = workflowRoot.get("operatorConnections");
        if (connectionsNode != null && connectionsNode.isArray()) {
            for (JsonNode connNode : connectionsNode) {
                String fromOp = connNode.get("fromOperator").asText();
                String toOp = connNode.get("toOperator").asText();
                workflow.addEdge(fromOp, toOp);
            }
        }

        // Finalize the graph (link START/END)
        workflow.finalizeGraph();

        return workflow;
    }

    /**
     * Parses the network JSON string into a list of Configuration objects.
     * @param networkJson The JSON string content.
     * @return A list of Configuration objects
     */
    private static Map<String, List<Configuration>> parseNetwork(String networkJson) {
        Map<String, List<Configuration>> configurations = new HashMap<>();

        try {
            // Read the root node
            JsonNode rootNode = mapper.readTree(networkJson);
            JsonNode sitesNode = rootNode.path("sites");

            if (sitesNode.isArray()) {
                // Iterate over each site
                for (JsonNode siteNode : sitesNode) {
                    String deviceId = siteNode.path("siteName").asText();
                    configurations.computeIfAbsent(deviceId, k -> new ArrayList<>());

                    JsonNode platformsNode = siteNode.path("availablePlatforms");

                    if (platformsNode.isArray()) {
                        // Iterate over each platform within the site
                        for (JsonNode platformNode : platformsNode) {
                            String platformID = platformNode.size() == 1 ? "platform_0" : platformNode.path("platformName").asText();

                            // Create and add the Configuration object
                            configurations.computeIfPresent(deviceId, (k, v) -> {
                                v.add(new Configuration(deviceId, platformID));
                                return v;
                            });
                        }
                    }
                }
            }
        } catch (JsonProcessingException e) {
            System.out.println("Error parsing network JSON: " + e.getMessage());
        }

        return configurations;
    }

    /**
     * Parses the dictionary JSON to extract valid sites for each operator.
     */
    private static Map<String, Set<String>> parseDictionary(String json) throws IOException {
        Map<String, Set<String>> result = new HashMap<>();
        JsonNode root = mapper.readTree(json);

        JsonNode operatorsArray = root.get("operators");
        if (operatorsArray != null && operatorsArray.isArray()) {
            for (JsonNode opItem : operatorsArray) {
                if (!opItem.has("classKey") || !opItem.has("sites")) continue;

                String operatorName = opItem.get("classKey").asText();
                Set<String> sites = new HashSet<>();

                JsonNode sitesNode = opItem.get("sites");
                Iterator<String> siteFieldNames = sitesNode.fieldNames();
                while (siteFieldNames.hasNext()) {
                    String deviceId = siteFieldNames.next();
                    // Filter out metadata keys if they exist at this level
                    if (!deviceId.equals("migrationCosts") && !deviceId.equals("staticCost")) {
                        sites.add(deviceId);
                    }
                }

                result.put(operatorName, sites);
            }
        }
        return result;
    }
}