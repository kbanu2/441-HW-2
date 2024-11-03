ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.15"

lazy val root = (project in file("."))
  .settings(
          name := "DataProcessor",
          libraryDependencies ++= Seq(
                  "com.knuddels" % "jtokkit" % "1.1.0",
                  "org.yaml" % "snakeyaml" % "2.0",
                  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
                  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M1.1",
                  "org.nd4j" % "nd4j-native" % "1.0.0-M2.1",
                  "ch.qos.logback" % "logback-classic" % "1.5.6",
                  "org.jacorb" % "jacorb" % "3.9",
                  "org.jboss.spec.javax.rmi" % "jboss-rmi-api_1.0_spec" % "1.0.6.Final",
                  "org.scalatest" %% "scalatest" % "3.2.10" % Test,
                  "org.apache.spark" %% "spark-core" % "3.5.1",
                  "org.apache.spark" %% "spark-sql" % "3.5.1"
          ),
          unmanagedSourceDirectories in Compile += baseDirectory.value / "src" / "main" / "resources"
  )
