

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.configuration.Strategy;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.*;
import org.apache.spark.sql.types.DoubleType;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.util.*;


public class prgwinetrain {
	
	 public static void main(String[]args) {

		 	System.setProperty("hadoop.home.dir", "s3://wineprgpredict/");
	        System.out.println("Beginning of page");
	        
	        System.out.println("Beginning of page");
	        SparkConf conf = new SparkConf().setAppName("testWine").setMaster("local");

	        SparkSession spark = SparkSession
	                .builder()
	                .appName("Wine trainer")
	                .master("local")
	                .getOrCreate();
	        	  	        
	        
	        // Load and parse the data file, converting it to a DataFrame.
	        Dataset<Row> trainingData = spark
	                .read()
	                .format("csv")
	                .option("header", "true")
	                .load("s3://wineprgpredict/TrainingDataset.csv");

	        		trainingData = trainingData.withColumn("fixed acidity",trainingData.col("fixed acidity").cast("double"));
	        		trainingData = trainingData.withColumn("volatile acidity",trainingData.col("volatile acidity").cast("double"));
	        		trainingData = trainingData.withColumn("citric acid",trainingData.col("citric acid").cast("double"));
	        		trainingData = trainingData.withColumn("residual sugar",trainingData.col("residual sugar").cast("double"));
	        		trainingData = trainingData.withColumn("chlorides",trainingData.col("chlorides").cast("double"));
	        		trainingData = trainingData.withColumn("free sulfur dioxide",trainingData.col("free sulfur dioxide").cast("double"));
	        		trainingData = trainingData.withColumn("total sulfur dioxide",trainingData.col("total sulfur dioxide").cast("double"));
	        		trainingData = trainingData.withColumn("density",trainingData.col("density").cast("double"));
	        		trainingData = trainingData.withColumn("pH",trainingData.col("pH").cast("double"));
	        		trainingData = trainingData.withColumn("sulphates",trainingData.col("sulphates").cast("double"));
	        		trainingData = trainingData.withColumn("alcohol",trainingData.col("alcohol").cast("double"));
	        		trainingData = trainingData.withColumn("quality",trainingData.col("quality").cast("double"));
	        		trainingData.dtypes();
	        		
      		
	        		
	        		
	        VectorAssembler assembler = new VectorAssembler()
	                .setInputCols(new String[]
	                        {"fixed acidity",
	                        "volatile acidity",
	                        "citric acid",
	                        "residual sugar",
	                        "chlorides",
	                        "free sulfur dioxide",
	                        "total sulfur dioxide",
	                        "density",
	                        "pH",
	                        "sulphates",
	                        "alcohol",
	                        "quality"})
	                .setOutputCol("features");

	               System.out.println("Test");
	               
	               
	        LogisticRegression lr = new LogisticRegression()
	                .setMaxIter(10)
	                .setRegParam(0.3)
	                .setElasticNetParam(0.8)
	                .setFeaturesCol("features")
	                .setLabelCol("quality");

	        System.out.println("s3://wineprgpredict/trainingdata.model/");
	        
	        
	        Dataset<Row> output = assembler.transform(trainingData);
	        output.select("features", "quality").show();


	        //Create pipeline
	        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler,lr});

	        //Fit the model
	        PipelineModel lrModel = pipeline.fit(trainingData);


	    

//	        // Print the coefficients and intercept for logistic regression
	        System.out.println("Coefficients: " + lrModel.explainParams());

	        
	       LogisticRegression mlr = new LogisticRegression()
	                .setMaxIter(10)
	                .setRegParam(0.3)
	              .setElasticNetParam(0.8)
	               .setFamily("multinomial");

	       
	        Dataset<Row> ValidationDataset = spark
	                .read()
	                .format("csv")
	                .option("header", "true")
	                .load("s3://wineprgpredict/ValidationDataset.csv");
	        
     		ValidationDataset = ValidationDataset.withColumn("fixed acidity",ValidationDataset.col("fixed acidity").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("volatile acidity",ValidationDataset.col("volatile acidity").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("citric acid",ValidationDataset.col("citric acid").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("residual sugar",ValidationDataset.col("residual sugar").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("chlorides",ValidationDataset.col("chlorides").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("free sulfur dioxide",ValidationDataset.col("free sulfur dioxide").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("total sulfur dioxide",ValidationDataset.col("total sulfur dioxide").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("density",ValidationDataset.col("density").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("pH",ValidationDataset.col("pH").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("sulphates",ValidationDataset.col("sulphates").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("alcohol",ValidationDataset.col("alcohol").cast("double"));
     		ValidationDataset = ValidationDataset.withColumn("quality",ValidationDataset.col("quality").cast("double"));
     		ValidationDataset.dtypes();	       
	        

     		
     		
     		Dataset <Row> prediction = lrModel.transform(ValidationDataset);
     		
     		System.out.println(prediction);
     		prediction.select("features", "quality").show();
     		

     		
	    }
	    

	}
