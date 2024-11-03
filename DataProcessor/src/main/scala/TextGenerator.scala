import it.unimi.dsi.fastutil.ints.IntArrayList
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File

class TextGenerator(dataProcessor: DataProcessor){
  // Function to create and return model using a filePath
  def loadPretrainedModel(modelPath: String): MultiLayerNetwork = {
    val file = new File(modelPath)
    ModelSerializer.restoreMultiLayerNetwork(file)
  }

  // Helper function to generate next word based on context
  private def generateNextWord(context: INDArray, model: MultiLayerNetwork): String = {
    val output = model.output(context) // Forward pass through the model
    val predictedIndex = Nd4j.argMax(output, 1).getInt(0) // Get index of the highest probability
    convertIndexToWord(predictedIndex) // Convert index back to a word
  }

  // Function to generate a sentence of length maxWords using model and text
  def generateSentence(seedText: String, model: MultiLayerNetwork, maxWords: Int): String = {
    val generatedText = new StringBuilder(seedText)
    var context = tokenizeAndEmbed(seedText) // Initial embedding based on seed text

    for (_ <- 0 until maxWords) {
      val nextWord = generateNextWord(context, model)
      generatedText.append(" ").append(nextWord)

      // Update the context for the next prediction by adding the new word to it
      context = updateContext(generatedText.toString())
      if (nextWord == "." || nextWord == "END") return generatedText.toString()
    }
    generatedText.toString()
  }

  // Helper function used to transform Seq[IntArrayList] into INDArray
  // Input is supposed to be one chunk, so it can be translated to a 1D INDArray
  private def completelyFlattenList(tokenLists: Seq[com.knuddels.jtokkit.api.IntArrayList]): INDArray = {
    val list = new IntArrayList()

    for (arr <- tokenLists) {
      for (i <- 0 until arr.size()){
        list.add(arr.get(i))
      }
    }

    Nd4j.create(list)
  }

  private def tokenizeAndEmbed(text: String): INDArray = {
    dataProcessor.changeData(text)
    completelyFlattenList(dataProcessor.processData())
  }

  // Tokenize and embed updated text using sliding window if needed
  private def updateContext(newText: String): INDArray = {
    tokenizeAndEmbed(newText)
  }

  // Helper function to return word associated with token (key)
  private def convertIndexToWord(index: Int): String = {
    dataProcessor.decodeToken(index)
  }
}
