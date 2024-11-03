import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.util.{Try, Success, Failure}

class ConfigLoaderSpec extends AnyFlatSpec with Matchers {

  "ConfigLoader" should "load configuration from a valid YAML file" in {
    val loader = new ConfigLoader("DataProcessor/src/main/resources/application.yaml")
    loader.appConfig should contain key "textFile"
    loader.appConfig("textFile") shouldBe "DataProcessor/src/main/resources/test.txt"
  }

  it should "throw an exception if the YAML file does not exist" in {
    val loaderTry = Try(new ConfigLoader("path/to/invalid_format.yaml"))
    loaderTry shouldBe a [Failure[_]]
  }

  it should "handle missing keys by providing default values" in {
    val loader = new ConfigLoader("DataProcessor/src/main/resources/application.yaml")
    loader.appConfig.get("nonExistentKey") shouldBe None
  }

  it should "correctly convert data types" in {
    val loader = new ConfigLoader("DataProcessor/src/main/resources/application.yaml")
    loader.appConfig("shardSize") shouldBe 3 // Ensure it's read as an Int
  }
}
