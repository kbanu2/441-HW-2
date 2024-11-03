import TextProcessing._
import com.knuddels.jtokkit.api.IntArrayList
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.omg.CORBA
import org.omg.CORBA.{Context, ContextList, DomainManager, ExceptionList, InterfaceDef, NVList, NamedValue, ORB, Policy, PolicyListHolder, Request, SetOverrideType}
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import scala.annotation.unused
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.break

// Implementation of the FileProcessor interface for processing text chunks.
class FileProcessorImpl extends FileProcessorPOA {
  private val logger: Logger = LoggerFactory.getLogger(classOf[FileProcessorImpl])
  val dataProcessor = new DataProcessor

  // Processes a chunk of text and returns the processing results.
  override def processChunk(chunkContent: String): ProcessingResult = {
    logger.info("Processing chunk of content...")

    // Load application configuration from YAML file.
    val configLoader = new ConfigLoader("DataProcessor/src/main/resources/application.yaml")
    val appConfig = configLoader.appConfig

    // Retrieve shard size and embedding dimension from the configuration.
    val shardSize = appConfig("shardSize").toString.toInt
    val embeddingDim = appConfig("embeddingDim").toString.toInt

    dataProcessor.setShardSize(shardSize)
    dataProcessor.changeData(chunkContent)

    logger.debug(s"Shard size: $shardSize, Embedding dimension: $embeddingDim")

    // Create a DataProcessor instance to handle the chunk of text.
    // Tokenize the input data into shards.
    val tokenizedSentences = dataProcessor.processData()

    // Calculate the total number of tokens generated from the tokenized sentences.
    val totalTokens = tokenizedSentences.map(_.size()).sum

    logger.debug(s"Total tokens: $totalTokens")

    // Create arrays to hold tokenized data and corresponding labels.
    val tokenizedArray = new Array[Array[Int]](totalTokens)
    val labelsArray = new Array[Array[Int]](totalTokens)

    var index = 0 // var used to keep track of position in array
    for (i <- tokenizedSentences.indices) {
      for (j <- 0 until tokenizedSentences(i).size()) {
        tokenizedArray(index) = Array(tokenizedSentences(i).get(j)) // Create a new array for each token
        index += 1 // Increment the index for the next position
      }
    }

    // Populate labelsArray with the next token's index for each token.
    for (i <- tokenizedArray.indices) {
      if (i < tokenizedArray.length - 1)
        labelsArray(i) = tokenizedArray(i + 1) // Assign the next token as the label
      else
        labelsArray(i) = Array(i) // Handle the last token case
    }

    // Convert tokenized data and labels into INDArray format for model training.
    val inputFeatures: INDArray = Nd4j.create(tokenizedArray)
    val outputLabels: INDArray = Nd4j.create(labelsArray)

    val vocabSize = dataProcessor.vocabulary.size + 100
    // Create an instance of ModelTrainer for training the embedding model.
    val modelTrainer = new ModelTrainer(vocabSize, embeddingDim, inputFeatures, outputLabels)
    modelTrainer.train()
    // Retrieve the trained embeddings.
    val embeddings = modelTrainer.getEmbeddings
    val result = new ProcessingResult()

    val wordMappings = new ArrayBuffer[WordMap]
    val wordFrequencies = new ArrayBuffer[WordFrequency]

    // Populate wordMappings with the vocabulary.
    for ((s, uid) <- dataProcessor.vocabulary) {
      val wordMap = new WordMap(s, uid)
      wordMappings += wordMap
    }

    // Populate wordFrequencies with the frequency of each word.
    for ((s, count) <- dataProcessor.vocabFrequency) {
      val wordFrequency = new WordFrequency(s, count)
      wordFrequencies += wordFrequency
    }

    logger.info(s"Vocabulary size: $vocabSize")

    // Assign word mappings, frequencies, and embeddings to the result
    result.wordMappings = wordMappings.toArray
    result.wordFrequencies = wordFrequencies.toArray
    result.vectorEmbedding = embeddings.toDoubleMatrix

    saveEmbeddings("test.bin", embeddings)
    modelTrainer.saveModel("model.zip")

    logger.info("Processing complete.")
    result
  }

  def createInputTokens(sentence: String): Seq[IntArrayList] = {
    dataProcessor.changeData(sentence)
    dataProcessor.processData()
  }

  def getDataProcessor: DataProcessor = {
    dataProcessor
  }

  def saveEmbeddings(filePath: String, embeddings: INDArray): Unit = {
    Nd4j.saveBinary(embeddings, new File(filePath))
    logger.info(s"Embeddings saved to $filePath")
  }

  // Write the processing results to output (console for now).
  override def writeResults(results: ProcessingResult): Unit = {
    logger.info("Writing results...")
    logger.debug(s"Word Frequencies: ${results.wordFrequencies.mkString("Array(", ", ", ")")}")
    logger.debug(s"Word Mappings: ${results.wordMappings.mkString("Array(", ", ", ")")}")
    logger.debug(s"Vector Embedding: ${results.vectorEmbedding.mkString("Array(", ", ", ")")}")
  }

  def createSlidingWindows(inputTokens: INDArray, windowSize: Int, stepSize: Int): Seq[INDArray] = {
    val totalTokens = inputTokens.rows
    val windows = ArrayBuffer[INDArray]()

    // Create sliding windows
    for (start <- 0 until totalTokens by stepSize) {
      // Calculate the end index of the current window
      val end = Math.min(start + windowSize, totalTokens)

      // Extract the current window
      val currentWindow = inputTokens.get(NDArrayIndex.interval(start, end), NDArrayIndex.all())

      // If the window is not empty, add it to the list
      if (currentWindow.rows > 0) {
        windows += currentWindow
      }

      // Stop if we reach the end of the input
      if (end >= totalTokens) {
        return windows.toSeq
      }
    }

    windows.toSeq
  }

  // Check if the object corresponds to the given repository identifier.
  override def _is_a(repositoryIdentifier: String): Boolean = {
    repositoryIdentifier == "IDL:FileProcessor:1.0"  // Assuming "FileProcessor" is the IDL type
  }

  // Check if the object does not exist.
  override def _non_existent(): Boolean = {
    false // Always returns false as the object exists
  }

  // Placeholder for retrieving the CORBA interface definition.
  override def _get_interface_def(): CORBA.Object = {
    null // No specific interface definition implemented
  }

  // All other CORBA methods can be left unimplemented or simplified as needed.
  def _set_policy_overrides(policies: Array[Policy], set_add: SetOverrideType): CORBA.Object = {
    this._this()  // Same as above
  }

  def _get_client_policy(`type`: Int): Policy = {
    null // No specific client policies implemented
  }

  def _get_policy_overrides(types: Array[Int]): Array[Policy] = {
    Array.empty // No overrides, return an empty array
  }

  def _validate_connection(inconsistent_policies: PolicyListHolder): Boolean = {
    true // Assume the connection is always valid
  }

  override def _get_component(): CORBA.Object = {
    null // No specific component, return null
  }

  def _get_orb(): ORB = {
    null // No specific ORB implementation
  }
}
