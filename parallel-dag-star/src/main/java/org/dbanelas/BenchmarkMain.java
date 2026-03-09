package org.dbanelas;

import org.dbanelas.cost.DAGStarCostModel;
import org.dbanelas.dp.DPDAGStar;
import org.dbanelas.esc.ESC;
import org.dbanelas.serial.DAGStar;
import org.dbanelas.parsers.CostModelParser;
import org.dbanelas.parsers.WorkflowParser;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class BenchmarkMain {

    // Benchmark results holder
    static class BenchmarkResult {
        String algorithmName;
        String parallelism; // "Serial" or thread count
        long executionTimeMs;
        double fCost;
        boolean foundSolution;

        public BenchmarkResult(String name, String parallelism, long time, double cost, boolean found) {
            this.algorithmName = name;
            this.parallelism = parallelism;
            this.executionTimeMs = time;
            this.fCost = cost;
            this.foundSolution = found;
        }
    }

    public static void main(String[] args) {
        // Argument Parsing ---
        if (args.length < 5) {
            System.err.println("Usage: java org.dbanelas.BenchmarkMain <latencyThreshold> <NetworkSize> <WorkflowName> <Threads> <FilePrefixPath>");
            System.err.println("Example: java org.dbanelas.BenchmarkMain 5.8 1023 PRED 2,4,8 /home/user/parallel-dag-star/test-files/");
            System.exit(1);
        }

        int networkSize = 0;
        double latencyThreshold = args[0].equals("-") ? Double.MAX_VALUE : Double.parseDouble(args[0]);
        String workflowName = args[2];
        int[] threadCounts = null;
        String basePath = args[4];

        try {
            networkSize = Integer.parseInt(args[1]);

            // Parse threads (allows "4" or "2,4,8,16")
            String[] threadParts = args[3].split(",");
            threadCounts = new int[threadParts.length];
            for (int i = 0; i < threadParts.length; i++) {
                threadCounts[i] = Integer.parseInt(threadParts[i].trim());
            }
        } catch (NumberFormatException e) {
            System.err.println("Error: Network Size and Threads must be integers. Latency threshold must be a double.");
            System.exit(1);
        }

        // Path Construction
        // Using Paths.get ensures OS independence and handles slash separators correctly
        Path baseDir = Paths.get(basePath);

        String networkFile = baseDir.resolve("net_" + networkSize)
                .resolve("network_" + networkSize + "_1_optimizer.json").toString();

        String dictionaryFile = baseDir.resolve("dict_" + networkSize + "_1_riot-" + workflowName + "-ifogsim.json").toString();

        String workflowFile = baseDir.resolve("riot-" + workflowName + "-ifogsim.json").toString();

        String costFile = baseDir.resolve(workflowName + "_" + networkSize + "_avg.xlsx").toString();

        String pairLatenciesFile = baseDir.resolve("net_" + networkSize)
                .resolve("network_" + networkSize + "_1_pair_lat.txt").toString();

        // Execution
        List<BenchmarkResult> results = new ArrayList<>();

        System.out.println("Loading Cost Model from: " + costFile);
        DAGStarCostModel costModel = loadCostModel(costFile, pairLatenciesFile);

        if (costModel == null) {
            System.err.println("Could not load Cost Model. Aborting benchmark.");
            return;
        }

        System.out.println("Starting Benchmark Suite...");
        System.out.println("Network Size: " + networkSize);
        System.out.println("Workflow: " + workflowName);
        System.out.println("Base Path: " + baseDir);
        System.out.println("--------------------------------------------------");

        try {
            // Benchmark Serial DAG*
            System.out.println("Benchmarking Serial DAG*...");
            Workflow serialWorkflow = loadWorkflow(workflowFile, networkFile, dictionaryFile, costModel);
            DAGStar serialAlgo = new DAGStar(serialWorkflow, "MAX", latencyThreshold);
            results.add(runBenchmark(serialAlgo, "DAG* (Serial)", "1"));

            // Benchmark DP-DAG* (Parallel)
            for (int threads : threadCounts) {
                System.out.println("Benchmarking DP-DAG* [" + threads + " threads]...");
                Workflow workflow = loadWorkflow(workflowFile, networkFile, dictionaryFile, costModel);
                DPDAGStar dpAlgo = new DPDAGStar(workflow, threads, latencyThreshold);
                results.add(runBenchmark(dpAlgo, "DP-DAG*", String.valueOf(threads)));
            }

        } catch (Exception e) {
            System.err.println("Benchmark interrupted due to error: " + e.getMessage());
            e.printStackTrace();
        }

        // Print Final Table
        printResultsTable(results);
    }

    private static BenchmarkResult runBenchmark(OptimalSearchAlgorithm algorithm, String name, String parallelism) {
        long startTime = System.currentTimeMillis();
        PartialGraph solution = null;

        try {
            solution = algorithm.solve();
        } catch (Exception e) {
            System.err.println("Error running " + name + ": " + e.getMessage());
        }

        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        if (solution != null) {
            return new BenchmarkResult(name, parallelism, duration, solution.getFCost(), true);
        } else {
            return new BenchmarkResult(name, parallelism, duration, 0.0, false);
        }
    }

    private static void printResultsTable(List<BenchmarkResult> results) {
        System.out.println("\n\n========================= BENCHMARK RESULTS =========================");
        System.out.printf("%-20s | %-12s | %-15s | %-15s%n", "Algorithm", "Threads", "Time (ms)", "Total Cost (F)");
        System.out.println("---------------------------------------------------------------------");

        boolean hdFirst = true;
        boolean dpFirst = true;
        for (BenchmarkResult r : results) {
            String costString = r.foundSolution ? String.format("%.4f", r.fCost) : "FAILED";

            if (dpFirst && r.algorithmName.startsWith("DP-DAG*")) {
                System.out.println("---------------------------------------------------------------------");
                dpFirst = false;
            }

            if (hdFirst && r.algorithmName.startsWith("HD-DAG*")) {
                System.out.println("---------------------------------------------------------------------");
                hdFirst = false;
            }
            System.out.printf("%-20s | %-12s | %-15d | %-15s%n",
                    r.algorithmName,
                    r.parallelism,
                    r.executionTimeMs,
                    costString);
        }
        System.out.println("=====================================================================");
    }

    private static DAGStarCostModel loadCostModel(String costFileStr, String pairLatenciesFileStr) {
        try (InputStream costFileInputStream = Files.newInputStream(Paths.get(costFileStr))) {
            String communicationCosts = new String(Files.readAllBytes(Paths.get(pairLatenciesFileStr)));
            return CostModelParser.load(costFileInputStream, Path.of(costFileStr).getFileName().toString(), communicationCosts);
        } catch (IOException e) {
            System.err.println("Failed to load Cost Model: " + e.getMessage());
            return null;
        }
    }

    // Loads a FRESH workflow instance for every run to ensure no state pollution between algorithms
    private static Workflow loadWorkflow(String workflowFileStr, String networkFileStr, String dictionaryFileStr, DAGStarCostModel costModel) throws Exception {
        String workflowJson = new String(Files.readAllBytes(Paths.get(workflowFileStr)));
        String dictionaryJson = new String(Files.readAllBytes(Paths.get(dictionaryFileStr)));
        String networkJson = new String(Files.readAllBytes(Paths.get(networkFileStr)));

        return WorkflowParser.parse(workflowJson, networkJson, dictionaryJson, costModel);
    }
}