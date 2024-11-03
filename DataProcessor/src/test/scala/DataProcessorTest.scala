import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.knuddels.jtokkit.api.IntArrayList

class DataProcessorTest extends AnyFlatSpec with Matchers {

  "DataProcessor" should "process data correctly into shards and tokens" in {
    val inputText = "hello world hello"
    val shardSize = 2
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    val tokens = dataProcessor.processData()

    // Check the number of shards created
    tokens.size shouldEqual 2

    // Check the content of the first shard
    val firstShardTokens = tokens.head
    firstShardTokens.size() shouldEqual 2
    firstShardTokens.get(0) shouldEqual 0 // "hello"
    firstShardTokens.get(1) shouldEqual 1 // "world"

    // Check the content of the second shard
    val secondShardTokens = tokens(1)
    secondShardTokens.size() shouldEqual 1
    secondShardTokens.get(0) shouldEqual 0 // "hello"
    secondShardTokens.get(1) shouldEqual 0 // "hello" again
  }

  it should "decode tokens back to text correctly" in {
    val inputText = "hello world"
    val shardSize = 1
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data to get tokens
    val tokens = dataProcessor.processData()

    // Decode the tokens back to text
    val decoded = dataProcessor.decodeTokens(tokens)

    // Check the decoded output
    decoded shouldEqual Seq("hello", "world")
  }

  it should "handle unknown tokens gracefully" in {
    val inputText = "unknown token test"
    val shardSize = 1
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    val tokens = dataProcessor.processData()

    // Decode the tokens
    val decoded = dataProcessor.decodeTokens(tokens)

    // Check that the decoded output matches the input
    decoded shouldEqual Seq("unknown", "token", "test")
  }

  it should "keep track of vocabulary frequencies" in {
    val inputText = "test test test example"
    val shardSize = 2
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    dataProcessor.processData()

    // Check the vocabulary frequencies
    dataProcessor.vocabFrequency("test") shouldEqual 3
    dataProcessor.vocabFrequency("example") shouldEqual 1
  }

  it should "process single word input correctly" in {
    val inputText = "hello"
    val shardSize = 1
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    val tokens = dataProcessor.processData()

    // Check that one shard is created with one token
    tokens.size shouldEqual 1
    tokens.head.size() shouldEqual 1
    tokens.head.get(0) shouldEqual 0 // Index of "hello"
  }

  it should "handle special characters in input" in {
    val inputText = "hello, world! #Scala"
    val shardSize = 2
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    val tokens = dataProcessor.processData()

    // Check the tokens created
    tokens.size shouldEqual 2
    val firstShardTokens = tokens.head
    firstShardTokens.size() shouldEqual 2
    firstShardTokens.get(0) shouldEqual 0 // "hello,"
    firstShardTokens.get(1) shouldEqual 1 // "world!"

    val secondShardTokens = tokens(1)
    secondShardTokens.size() shouldEqual 1
    secondShardTokens.get(0) shouldEqual 2 // "#Scala"
  }

  it should "tokenize input with varying shard sizes correctly" in {
    val inputText = "one two three four five six"
    val shardSize = 3
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    val tokens = dataProcessor.processData()

    // Check the number of shards created
    tokens.size shouldEqual 2

    // Check the content of the first shard
    val firstShardTokens = tokens.head
    firstShardTokens.size() shouldEqual 3
    firstShardTokens.get(0) shouldEqual 0 // "one"
    firstShardTokens.get(1) shouldEqual 1 // "two"
    firstShardTokens.get(2) shouldEqual 2 // "three"

    // Check the content of the second shard
    val secondShardTokens = tokens(1)
    secondShardTokens.size() shouldEqual 3
    secondShardTokens.get(0) shouldEqual 3 // "four"
    secondShardTokens.get(1) shouldEqual 4 // "five"
    secondShardTokens.get(2) shouldEqual 5 // "six"
  }

  it should "correctly update vocabulary frequencies with repeated words" in {
    val inputText = "hello world hello world"
    val shardSize = 2
    val dataProcessor = new DataProcessor()
    dataProcessor.changeData(inputText)
    dataProcessor.setShardSize(shardSize)

    // Process the data
    dataProcessor.processData()

    // Check the vocabulary frequencies
    dataProcessor.vocabFrequency("hello") shouldEqual 2
    dataProcessor.vocabFrequency("world") shouldEqual 2
  }
}
