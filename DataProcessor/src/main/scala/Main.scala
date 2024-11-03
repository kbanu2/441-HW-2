import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{EmbeddingLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scala.jdk.CollectionConverters._
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.factory.Nd4j
import scala.io.StdIn.readLine

object Main extends App {
  // Load configuration
  val configLoader = new ConfigLoader("DataProcessor/src/main/resources/application.yaml")
  val config = configLoader.appConfig

  // Extract configuration values
  val textFile = config("textFile").toString
  val outputFile = config("outputFile").toString
  val shardSize = config("shardSize").asInstanceOf[Int]
  val embeddingDim = config("embeddingDim").asInstanceOf[Int]
  val windowSize = config("windowSize").asInstanceOf[Int]
  val stepSize = config("stepSize").asInstanceOf[Int]
  val batchSize = config("batchSize").asInstanceOf[Int]
  val numEpochs = config("numEpochs").asInstanceOf[Int]

  val fileProcessor = new FileProcessorImpl
  val logger = LoggerFactory.getLogger(this.getClass)

  val sparkSession = SparkSession.builder()
    .appName("SparkApp")
    .master("local[*]")
    .getOrCreate()

  val sc = sparkSession.sparkContext

  // Load text data from the specified file path
  val textData = scala.io.Source.fromFile(textFile).getLines().toSeq
  val dataRDD: RDD[Seq[Double]] = sc.parallelize(textData).flatMap { chunk =>
    val results = fileProcessor.processChunk(chunk)
    results.vectorEmbedding.map(e => e.map(math.abs).toSeq).toSeq
  }

  // Define sliding window transformation
  def createSlidingWindows(embedding: Seq[Double], windowSize: Int, stepSize: Int): Seq[Seq[Double]] = {
    embedding.sliding(windowSize, stepSize).toSeq
  }

  // Generate sliding windows in parallel
  val slidingWindowsRDD: RDD[Seq[Seq[Double]]] = dataRDD.mapPartitions { partition =>
    partition.map { embedding =>
      createSlidingWindows(embedding, windowSize, stepSize)
    }
  }

  // Prepare training data as RDD of DataSet instances
  val trainingDataRDD: RDD[DataSet] = slidingWindowsRDD.flatMap { windows =>
    windows.sliding(windowSize, 1).collect {
      case Seq(input, target) =>
        val inputNd4j = Nd4j.create(input.toArray).reshape(windowSize, 1)
        val targetNd4j = Nd4j.create(target.toArray).reshape(windowSize, 1)
        new DataSet(inputNd4j, targetNd4j)
    }
  }

  // Collect training data to feed to Deeplearning4j model
  val trainingData = trainingDataRDD.collect()
  val dataSetIterator = new ListDataSetIterator(trainingData.toList.asJava, batchSize)

  // Model configuration
  private val modelConfig = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new EmbeddingLayer.Builder()
      .nIn(fileProcessor.dataProcessor.vocabulary.size + 2)
      .nOut(embeddingDim)
      .activation(Activation.IDENTITY)
      .build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
      .nIn(embeddingDim)
      .nOut(fileProcessor.dataProcessor.vocabulary.size + 3)
      .activation(Activation.SOFTMAX)
      .build())
    .build()

  val model = new MultiLayerNetwork(modelConfig)
  model.init()

  // Set up listeners to monitor training progress
  model.setListeners(new ScoreIterationListener(10))

  val statsPrinter = new StatisticsPrinter(fileProcessor.dataProcessor)
  for (epoch <- 0 until numEpochs) {
    logger.info(s"Starting epoch $epoch")
    val startTime = System.currentTimeMillis()

    model.fit(dataSetIterator)

    val endTime = System.currentTimeMillis()
    statsPrinter.logEpochStats(outputFile, model, epoch, startTime)
    logger.info(s"Epoch $epoch completed in ${(endTime - startTime)} ms")
  }

  val textGenerator = new TextGenerator(fileProcessor.dataProcessor)
  print("Enter Sentence: ")
  val text = readLine()
  print("Enter max sentence length (int): ")
  val length = readLine().toInt
  logger.info(textGenerator.generateSentence(text, model, length))

  sparkSession.stop()
}
