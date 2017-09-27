
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

fun main(args: Array<String>) {
    println("> MetaL <")

    val spark = SparkSession.builder()
            .master("local[*]")
            .appName("Example")
            .orCreate

    val covtypeData: Dataset<Row> = spark.sqlContext()
            .read()
            .option("inferSchema", true)
            .csv("/Users/bencecserna/Documents/Development/projects/MetaL/resources/covtype/covtype.data.gz")
            .limit(1000)
            .toDF()
            .withColumnRenamed("_c54", "label")
            .cache()

    covtypeData.printSchema()

    val labelledData = VectorAssembler()
            .setInputCols(Array(4, {"_c${it + 10}"}))
            .setOutputCol("features")
            .transform(covtypeData)
            .select("label", "features")
            .cache()

    val randomSplit = labelledData.randomSplit(doubleArrayOf(0.4, 0.6), 1L)
    val trainingSet = randomSplit[0].cache()
    val testSet = randomSplit[1].cache()


    println(trainingSet.first())
    trainingSet.show(3, false)

    val logisticRegression = LogisticRegression()
            .setLabelCol("label")
            .setFeaturesCol("features")
            .setMaxIter(10)
            .setFitIntercept(true)

    val classifier = OneVsRest()
            .setClassifier(logisticRegression)

    val oneVsRestModel = classifier.fit(trainingSet)
    val predictions = oneVsRestModel.transform(trainingSet)

    val evaluator = MulticlassClassificationEvaluator()
            .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: $accuracy")

    spark.stop()
}


