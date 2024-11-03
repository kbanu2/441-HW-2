import java.io.{File, PrintWriter}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

class StatisticsPrinter(dataProcessor: DataProcessor) {

  def printToConsole(): Unit = {
    println("Vocabulary Statistics:")
    dataProcessor.vocabulary.foreach { case (token, index) =>
      println(s"$token $index")
    }

    println("\nVocabulary Frequency:")
    dataProcessor.vocabFrequency.foreach { case (token, frequency) =>
      println(s"$token $frequency")
    }
  }

  def writeToCSV(filePath: String): Unit = {
    val writer = new PrintWriter(new File(filePath))

    // Writing Vocabulary
    writer.println("Token,Index")
    dataProcessor.vocabulary.foreach { case (token, index) =>
      writer.println(s"$token,$index")
    }

    // Writing Frequency
    writer.println("\nToken,Frequency")
    dataProcessor.vocabFrequency.foreach { case (token, frequency) =>
      writer.println(s"$token,$frequency")
    }

    writer.close()
  }

  // New function to log training statistics to a file
  private def writeTrainingStats(filePath: String, model: MultiLayerNetwork, epoch: Int, trainingLoss: Double,
                                 accuracy: Double, gradientNorm: Double, memoryUsage: Long, timePerEpoch: Long): Unit = {
    val writer = new PrintWriter(new File(filePath))

    // Writing headers for training statistics
    writer.println("Epoch,Training Loss,Accuracy,Gradient Norm,Memory Usage (MB),Time per Epoch (ms)")

    // Writing the actual training statistics for each epoch
    writer.println(s"$epoch,$trainingLoss,$accuracy,$gradientNorm,${memoryUsage / (1024 * 1024)},$timePerEpoch")

    writer.close()
  }

  // Helper function to calculate memory usage
  private def getMemoryUsage: Long = {
    val runtime = Runtime.getRuntime
    runtime.totalMemory - runtime.freeMemory
  }

  def logEpochStats(filePath: String, model: MultiLayerNetwork, epoch: Int, startTime: Long): Unit = {
    val endTime = System.currentTimeMillis()
    val timePerEpoch = endTime - startTime
    val trainingLoss = model.score()
    val accuracy = getAccuracy(model)
    val gradientNorm = model.gradient().gradient().norm2Number().doubleValue()
    val memoryUsage = getMemoryUsage

    writeTrainingStats(filePath, model, epoch, trainingLoss, accuracy, gradientNorm, memoryUsage, timePerEpoch)
  }

  // Placeholder method to calculate accuracy
  private def getAccuracy(model: MultiLayerNetwork): Double = {
    0.0
  }
}
