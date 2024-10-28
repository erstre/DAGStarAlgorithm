# DAG*: A Novel A*-alike Algorithm for Optimal Workflow Execution across IoT Platforms

## Table of Contents

- [Description](#description)
- [Usage](#usage)
- [Workflows](#workflows)
- [License](#license)
- [Authors](#authors)

## Description
Many IoT applications from diverse domains rely on real-time, online analytics workflow execution to timely support decision making procedures. The efficient execution of analytics workflows requires the utilization of the processing power available across the cloud to edge continuum. Nonetheless, suggesting the optimal workflow execution over a large network of heterogeneous devices is a challenging task. The increased IoT network size increases the complexity of the optimization problem at hand. The ingested data streams exhibit highly volatile properties. The population of network devices dynamically changes. We introduce DAG*, an A*-alike algorithm that prunes large amounts of the search space explored for suggesting the most efficient workflow execution with formal optimality guarantees. We provide an incremental version of DAG* retaining the optimality property.

## Workflows
ETL: ingests incoming data streams in SenML format, performs data filtering of outliers on individual observation types using a Range and Bloom filter, and subsequently interpolates missing values. It then annotates additional meta-data into the observed fields of the message and then inserts the resulting tuples into Azure table storage, while also converting the data back to SenML and publishing it to MQTT. A dummy sink task shown is used for logging purposes.

STATS: parses the input messages that arrive in SenML format – typically from the ETL, but kept separate here for modularity. It then performs three types of statistical analytics in parallel on individual observation fields present in the message: an average over a 10 message window, Kalman filtering to smooth the observation fields followed by a sliding window linear regression, and an approximate count of distinct values that arrive. These three output streams are then grouped for each sensor IDs, plotted and the resulting image files zipped. These three tasks are tightly coupled and we combine them into a single meta-task for manageability, as is common. and the output file is written to Cloud storage for hosting on a portal.

TRAIN: application uses a timer to periodically (e.g., for every minute) trigger a model training run. Each run fetches data from the Azure table available since the last run and uses ti to train a Linear Regression model. In addition, these fetched tuples are also annotated to allow a Decision Tree classifier to be trained. Both these trained model files are then uploaded to Azure blob storage and their files URLs are published to the MQTT broker.

PRED: application subscribes to these notifications and fetches the new model files from the blob store, and updates the downstream prediction tasks. Meanwhile, the dataflow also consumes pre-processed messages streaming in, say from the ETL dataflow, and after parsing it forks it to the decision tree classifier and the multi-variate regression tasks. The classifier assigns messages into classes, such as good, average or poor, based on one or more of their field values, while linear regression predicts a numerical attribute value in the message using several others. The regression task also compares the predicted values against a moving average and estimates the residual error between them. The predicted classes, values and errors are published to the MQTT broker.

For more details please visit: [RIoTBench: A Real-time IoT Benchmark for Distributed Stream Processing Platforms](https://arxiv.org/pdf/1701.08530) paper or [riot-bench](https://github.com/dream-lab/riot-bench?tab=readme-ov-file) repository.


### Prerequisites

The following software needs to be installed.

- ```Python```

## License

Distributed under the GPL-3.0 License. See [`LICENSE`](LICENSE) for more information.

## Authors

[Errikos Streviniotis](https://www.linkedin.com/in/errikos-streviniotis/)
