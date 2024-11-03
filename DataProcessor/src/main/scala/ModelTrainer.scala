import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{EmbeddingLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import java.io.File

// Class to train a neural network model for generating embeddings.
class ModelTrainer(vocabSize: Int, embeddingDim: Int, inputFeatures: INDArray, outputLabels: INDArray) {
  private val logger: Logger = LoggerFactory.getLogger(classOf[ModelTrainer])
  logger.info("Initializing model configuration...")

  // Configuring the neural network model with an embedding layer and an output layer.
  private val config = new NeuralNetConfiguration.Builder()
    .list()
    .layer(new EmbeddingLayer.Builder()
      .nIn(vocabSize)         // Number of input neurons (vocabulary size)
      .nOut(embeddingDim)     // Number of output neurons (embedding dimensions)
      .activation(Activation.IDENTITY) // Activation function for the embedding layer
      .build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT) // Output layer with sparse categorical cross-entropy loss
      .nIn(embeddingDim)      // Input size from the embedding layer
      .nOut(vocabSize + 1)    // Number of output neurons (vocabulary size + 1 for additional output)
      .activation(Activation.SOFTMAX) // Activation function for the output layer
      .build())
    .build()

  // Initialize the multi-layer network with the configuration defined above.
  private val model = new MultiLayerNetwork(config)
  private val epoch = 100

  // Train the model on the provided input features and output labels.
  def train(): Unit = {
    model.init()
    for (_ <- 0 until epoch) { // Iterate for the specified number of epochs.
      model.fit(inputFeatures, outputLabels)
    }

    logger.info("Model training completed.")
  }

  // Retrieve the embeddings (weights) from the embedding layer.
  def getEmbeddings: INDArray = {
    model.getLayer(0).getParam("W")
  }

  def saveModel(filePath: String): Unit = {
    ModelSerializer.writeModel(model, new File(filePath), true)
    logger.info(s"Model saved to $filePath")
  }
}
