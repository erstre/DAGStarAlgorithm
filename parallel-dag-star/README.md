# Parallel DAG* algorithm

## 1. Overview

The `parallel-dag-star` contains the code for the parallelized DAG*
algorithm or using our informal name: Disjoint Parallel DAG*. The main class for running
benchmarks and experiments in this repository is
`org.dbanelas.BenchmarkMain`, which loads workflow/network/cost inputs
and prints benchmark timing and cost summaries. The main algorithmic code is in
`org.dbanelas.dp` package which contain the `DPDAGStarWorker.java` and
`DPDAGStar.java` classes. The `DPDAGStarWorker` class implements the
core parallelized logic, while `DPDAGStar` provides a wrapper for running the algorithm and returning
the optimized workflow. 


## 2. Repository Structure

```text
parallel-dag-star/
├── src/main/java/org/dbanelas/         # Core algorithms, models, parsers, entry points
├── test-files/                         # Workflows, dictionaries, network files, cost spreadsheets
├── target/                             # Build outputs (including shaded jar)
└── pom.xml                             # Maven configuration
```

## 3. Requirements

- Java 11+ (recommended for compatibility with current Maven compiler plugin settings)
- Maven 3.8+

Check versions:

```bash
java -version
mvn -version
```

## 4. Build

Build the project and produce jars:

```bash
mvn clean package
```

Expected artifacts include:
- `target/original-parallel-dag-star-1.0-SNAPSHOT.jar`

## 6. Run Benchmarks

CLI contract (from `BenchmarkMain`):

```bash
java -cp pdag.jar org.dbanelas.BenchmarkMain <latencyThreshold> <NetworkSize> <WorkflowName> <Threads> <FilePrefixPath>
```

Arguments:

- `<latencyThreshold>`: numeric value (or `-` for no effective limit)
- `<NetworkSize>`: network size, e.g. `7`, `1023`, `2047`
- `<WorkflowName>`: workflow token used in filenames (e.g. `train`, `pred`, `etl`, `stats`)
- `<Threads>`: single count (`4`) or comma-separated list (`2,4,8`)
- `<FilePrefixPath>`: base directory containing input files (typically `test-files`)

Examples:

```bash
# Single thread count
java -jar pdag.jar - 7 train 2 test-files

# Multiple thread counts
java -jar pdag.jar 5.8 1023 pred 2,4,8 test-files
```

## 7. Output

The benchmark prints:

- selected dataset context (network size, workflow, base path)
- per-run summary rows with:
  - algorithm name
  - thread count
  - execution time (ms)
  - total cost (F), or `FAILED` if no solution was returned

You can redirect output to a file if desired:

```bash
java -cp target/esc.jar org.dbanelas.BenchmarkMain - 7 train 2,4 test-files > results/train-7.txt
```
