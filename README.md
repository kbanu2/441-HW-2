# Krenar Banushi kbanu2@uic.edu CS 441 Homework 2 Textbook Group 2 #
https://youtu.be/XlPAT22rIOU

# Overview #
The DataProcessor program is designed to process textual data for machine learning applications, specifically focusing on generating embeddings and training language models. It utilizes Apache Spark for parallel processing and Deeplearning4j for building and training neural networks.

# Build and Runtime Process#
This project utilizes Apache Spark for parallel processing of sliding windows used for training a neural network model.  The data is generated from the results of Homework 1, where vector embeddings were created for a large amount of data

## Prerequisites
Ensure that you have the following installed:

Hadoop (Used if using Apache Spark Locally): https://hadoop.apache.org/releases.html
  If you are planning to test this with Apache Spark Locally, you will need to specify the HADOOP_HOME path in the run configuration

## Features ##
Configuration Loading: Load configurations from a YAML file.
Text Processing: Split and tokenize text data into manageable chunks.
Embeddings Generation: Generate vector embeddings for words using a neural network.
Sentence Generation: Generate sentences based on an input string using a model trained with the embeddings.
Logging: Comprehensive logging for debugging and monitoring.

# Input and Output
## Input
Vector Embeddings: Stored as Array of array of doubles to be loaded and used for model training
User input in form of string
## Output
Resulting sentence filled in using the model for word generation
Statistics file outlining some statistics of the model generation such as:
Number of epochs, Training loss, Accuracy, Gradient Norm, Memory Usage in MB, and time per epoch

# Cluster Configuration
Note: Remove note of .master(local[*]) if using a cloud Apache Spark service

## Create a New EMR Cluster:
## Deployment
Upload your project JAR file and resources to an S3 bucket.
You can also use a bootstrap action to install any additional libraries not included in the default EMR setup.

## Submitting Spark Jobs
Use the following command to submit your Spark job:
`aws emr add-steps --cluster-id <your-cluster-id> --steps Type=Spark,Name="DataProcessor",Args=[--class,Main,s3://<your-bucket>/DataProcessor.jar]`

## Spark Configuration
You may want to adjust the following configurations:
`spark.executor.memory=2g`
`spark.driver.memory=1g`
`spark.executor.cores=2`

## Prerequisites ##
Before running the program, ensure you have the following installed:

  Java Development Kit (JDK): Version 8 or later.
    Note: If using later than version 8, in VM Configurations you will need to add the line:
    `--add-opens=java.base/sun.nio.ch=ALL-UNNAMED`
  Scala: Version 2.13 or later.
  SBT (Scala Build Tool): For managing dependencies and building the project.
  Deeplearning4j: Ensure the Deeplearning4j library is included in your build.sbt.
  Apache Spark
  Apache Spark SQL

## Installation## 
### Clone the Repository ###

`git clone https://github.com/kbanu2/441-HW-2.git`
`cd DataProcessor`

### Build the Project ###

Use SBT to build the Scala project:
`sbt clean compile`

## Running the Program ##
### Scala Component ###
Set Up the Configuration File

Create a YAML configuration file at src/main/resources/application.yaml with the necessary parameters, such as shardSize and embeddingDim.
Ensure that you specify where the embedding input is stored

### Run the Scala Program ###

To run the Scala program, execute the following command from the project root:
`sbt run`

