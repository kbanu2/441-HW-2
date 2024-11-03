import com.knuddels.jtokkit.api.IntArrayList
import scala.collection.mutable

class DataProcessor() {
  // Create a new EncodingRegistry and get an encoding type
  val vocabulary: mutable.TreeMap[String, Int] = mutable.TreeMap[String, Int]()
  val vocabFrequency: mutable.Map[String, Int] = mutable.Map[String, Int]()

  // Function to set shardSize
  def setShardSize(size : Int): Unit = {
    shardSize = size
  }

  // Function to update the data stored in data processor
  def changeData(newData: String): Unit = {
    data = newData
  }

  // Process the input data by splitting into shards and tokenizing
  def processData(): Seq[IntArrayList] = {
    // Split the input data into shards
    val shards = splitIntoShards(data)
    // Convert shards into tokens
    convertShardsToTokens(shards)
  }

  // Split the text into shards for parallel processing
  private def splitIntoShards(text: String): Seq[String] = {
    // Convert to lowercase and retain only English letters, punctuation, and spaces
    val cleanedText = text.toLowerCase.replaceAll("[^a-zA-Z\\p{Punct}\\s]", "")
    // Split the cleaned text into words and group them into shards
    cleanedText.split("\\s+").grouped(shardSize).map(_.mkString(" ")).toSeq
  }


  // Convert text shards into numerical tokens
  private def convertShardsToTokens(shards: Seq[String]): Seq[IntArrayList] = {
    shards.map { shard =>
      val tokens = shard.split("\\s+")
      val intArrayList = new IntArrayList()

      tokens.foreach { token =>
        // Add the token to the vocabulary if it does not exist
        val index = vocabulary.getOrElseUpdate(token, {
          val newIndex = vocabulary.size // Assign a unique index
          vocabulary += (token -> newIndex) // Add token to the vocabulary
          vocabFrequency += (token -> 1) // Initialize frequency count to 1
          newIndex // Return the new index
        })

        vocabFrequency(token) += 1 // Increment frequency count for existing token
        intArrayList.add(index) // Add the token's index to the IntArrayList
      }

      intArrayList
    }
  }

  // Decode numerical tokens back to text
  def decodeTokens(tokens: Seq[IntArrayList]): Seq[String] = {
    tokens.map { tokenList =>
      val decodedTokens = (0 until tokenList.size()).map(i => {
        val tokenIndex = tokenList.get(i)
        vocabulary.find(_._2 == tokenIndex).map(_._1).getOrElse("[UNK]")
      })
      decodedTokens.mkString(" ")
    }
  }

  def decodeToken(token: Int): String = {
    vocabulary.find(_._2 == token).map(_._1).getOrElse("UNK")
  }

  // vars used to update data without creating new objects
  // This allows us to keep context
  private var data = new String
  private var shardSize = 0
}
