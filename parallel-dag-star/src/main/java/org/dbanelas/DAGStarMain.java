package org.dbanelas;

import org.dbanelas.cost.DAGStarCostModel;
import org.dbanelas.dp.DPDAGStar;
import org.dbanelas.esc.ESC;
import org.dbanelas.parsers.CostModelParser;
import org.dbanelas.parsers.WorkflowParser;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;

public class DAGStarMain {

    public static void main(String[] args) {


        int networkSize = 7;
        String networkFile = "test-files/net_" + networkSize +"/network_" + networkSize + "_1_optimizer.json";
        String dictionaryFile = "test-files/dict_" + networkSize + "_1_riot-train-ifogsim.json";
        String workflowFile = "test-files/riot-train-ifogsim.json";
        String costFile = "test-files/train_" + networkSize + "_avg.xlsx";
        String pairLatenciesFile = "test-files/net_" + networkSize + "/network_" + networkSize + "_1_pair_lat.txt";

        DAGStarCostModel costModel = null;
        try {
            System.out.println("Reading input files...");
            String workflowJson = new String(Files.readAllBytes(Paths.get(workflowFile)));
            String dictionaryJson = new String(Files.readAllBytes(Paths.get(dictionaryFile)));
            String networkJson = new String(Files.readAllBytes(Paths.get(networkFile)));
            String communicationCosts = new String(Files.readAllBytes(Paths.get(pairLatenciesFile)));

            // Load the Cost Model using the decoupled Loader
            try (InputStream costFileInputStream = Files.newInputStream(Paths.get(costFile))) {
                System.out.println("Loading Cost Model...");
                costModel = CostModelParser.load(costFileInputStream, Path.of(costFile).getFileName().toString(), communicationCosts);
            } catch (IOException e) {
                System.out.println("Failed to load Cost Model: " + e.getMessage());
                e.printStackTrace();
            }


            // Parse the Workflow
            assert costModel != null : "Cost Model failed to load.";
            System.out.println("Parsing Workflow...");
            Workflow workflow = WorkflowParser.parse(workflowJson, networkJson, dictionaryJson, costModel);

            // Initialize and Run DAG*
            System.out.println("Starting DAG* Solver...");
//            DPDAGStar solver = new DPDAGStar(workflow, 2, 1.2);
            ESC solver = new ESC(workflow, 1);
            long startTime = System.currentTimeMillis();
            PartialGraph solution = solver.solve();
            long endTime = System.currentTimeMillis();

            // Output Results
            System.out.println("==========================================");
            if (solution != null) {
                System.out.println("OPTIMAL SOLUTION FOUND");
                System.out.println("==========================================");
                System.out.println("Execution Time: " + (endTime - startTime) + " ms");
                // Assuming PartialGraph has a getCost() or getF() method for total cost
                System.out.println("Total Cost: " + solution.getFCost());
                System.out.println("Real Cost: " + solution.getGCost());

//                System.out.println("Placement Plan:");
//                Map<String, Configuration> assignments = solution.getAssignments();
//
//                // Print assignments nicely
//                assignments.forEach((operator, config) -> {
//                    System.out.printf("  %-20s -> Device: %-20s | Platform: %s%n",
//                            operator, config.deviceID(), config.platformID());
//                });

            } else {
                System.out.println("Execution Time: " + (endTime - startTime) + " ms");
                System.out.println("NO SOLUTION FOUND");
                System.out.println("==========================================");
                System.out.println("The algorithm explored the search space but could not find a valid complete plan.");
            }

        } catch (IOException e) {
            System.err.println("Error reading files: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("An unexpected error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}