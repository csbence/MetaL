
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession

fun main(args: Array<String>) {
    println("_________________________")
    println(">>>>>>>>> MetaL <<<<<<<<<")
    println("The Meta Learning Library")
    println("_________________________")

    Logger.getLogger("org").level = Level.OFF
    Logger.getLogger("akka").level = Level.OFF

    val spark = SparkSession.builder()
            .master("local[*]")
            .appName("Example")
            .orCreate

    spark.sparkContext().setLogLevel("ERROR")

    val data = loadCovtype(spark)

    decisionTreeClassifier(data)
    multinomialRegression(data)
    testMulticassLogisticRegression(data)

    spark.stop()
}

fun multinomialRegression(data: Dataset<Row>) {
    val (train, test) = split(data)

    val lr = LogisticRegression()
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(train)

    // Print the coefficients and intercept for multinomial logistic regression
    println("Coefficients: \n${lrModel.coefficientMatrix()}")
    println("Intercepts: ${lrModel.interceptVector()}")
}

fun testMulticassLogisticRegression(data: Dataset<Row>) {
    val (train, test) = split(data)

    // instantiate the base classifier
    val classifier = LogisticRegression()
            .setMaxIter(100)
            .setTol(1E-6)
            .setFitIntercept(true)

    // instantiate the One Vs Rest Classifier.
    val ovr = OneVsRest().setClassifier(classifier)

    // train the multiclass model.
    val ovrModel = ovr.fit(train)

    // score the model on test data.
    val predictions = ovrModel.transform(test)

    predictions.show(10, false)

    // obtain evaluator.
    val evaluator = MulticlassClassificationEvaluator()
            .setMetricName("accuracy")

    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = $accuracy")
}

fun decisionTreeClassifier(data: Dataset<Row>) {
    val (train, test) = split(data)

    val decisionTreeClassifier = DecisionTreeClassifier()
            .setLabelCol("label")
            .setFeaturesCol("features")

    val decisionTreeClassificationModel = decisionTreeClassifier.fit(train)

    val predictions = decisionTreeClassificationModel.transform(test)

    predictions.show(10)

    // obtain evaluator.
    val evaluator = MulticlassClassificationEvaluator()
            .setMetricName("accuracy")

    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = $accuracy")
}

private fun split(data: Dataset<Row>): Pair<Dataset<Row>, Dataset<Row>> {
    val split = data.randomSplit(doubleArrayOf(0.8, 0.2))
    val train = split[0]
    val test = split[1]
    return Pair(train, test)
}

fun loadCovtype(spark: SparkSession): Dataset<Row> {
    val covtypeData: Dataset<Row> = spark.sqlContext()
            .read()
            .option("inferSchema", true)
            .csv("/Users/bencecserna/Documents/Development/projects/MetaL/resources/covtype/covtype.data.gz")
            .limit(10000)
            .toDF()
            .withColumnRenamed("_c54", "label")
            .cache()

    return VectorAssembler()
            .setInputCols(Array(54, { "_c$it" }))
            .setOutputCol("features")
            .transform(covtypeData)
            .select("label", "features")

//    val metadata = MetadataBuilder().putLong("numFeatures", 54).build()
//
//    val labelledData = VectorAssembler()
//            .setInputCols(Array(54, { "_c$it" }))
//            .setOutputCol("features")
//            .transform(covtypeData)
//            .select("label", "features")
//            .select(Column("label").multiply(1.0).`as`("label"), Column("features").`as`("features", metadata))
//            .cache()

}
