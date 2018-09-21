import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class DtreeMain {
	
	public static void main(String args[]) throws Exception
	{
		// read csv file
		// convert to arff/instances
		// create model from training set
		// apply on validation set to find accuracy
		// generate output for test set
		DtreeMain dtreeMain = new DtreeMain();
		Instances train = dtreeMain.generateArff("training_set");
		
		Instances validation = dtreeMain.generateArff("validation_set");
		Instances test = dtreeMain.generateArff("test_set");
		
		Classifier cls = new J48();		
		cls.buildClassifier(train);
		SerializationHelper.write("/Users/vinoth/Downloads/data_sets1/j48.model", cls);
		
		System.out.println("Validation Set: Stats");
		dtreeMain.generateConfustionMatrix(validation);
		System.out.println("\n\n\n**********************\n\n\n");
		System.out.println("Test Set: Stats");
		dtreeMain.generateConfustionMatrix(test);
		
		
		
		
		
		
	}
	
	public void generateConfustionMatrix(Instances data) throws Exception
	{
		Classifier clsFromModel  = (Classifier) SerializationHelper.read("/Users/vinoth/Downloads/data_sets1/j48.model");
		int truePositive = 0,trueNegative=0,falsePositive=0 ,falseNegative=0;
		for(Instance instance:data)
		{
			double actualValue = instance.classValue();
			
			double predictedValue = clsFromModel.classifyInstance(instance);
			
			if(actualValue==0.0 && predictedValue==0.0)
			{
				truePositive++;
			}
			else if(actualValue==0.0 && predictedValue==1.0)
			{
				trueNegative++;
			}
			else if(actualValue==1.0 && predictedValue==0.0)
			{
				falsePositive++;
			}else
			{
				falseNegative++;
			}
			
			
		}
		
		System.out.println("Confusion Matrix");
		System.out.println("\tTRUE\tFALSE");
		System.out.println("TRUE\t"+truePositive+"\t"+trueNegative);
		System.out.println("FALSE\t"+falsePositive+"\t"+falseNegative);
		
		System.out.println("Accuracy:"+((truePositive+falseNegative)*100.0/data.numInstances()));

	}
	
	public Instances generateArff(String fileName) throws IOException
	{
		
		CSVLoader loader = new CSVLoader();
		loader.setNoHeaderRowPresent(false);
		loader.setSource(new File("/Users/vinoth/Downloads/data_sets1/"+fileName+".csv"));
		Instances data = loader.getDataSet();
		
		data.setClassIndex(data.numAttributes() - 1);
//		
//		// arff can be saved for later usage (try different classifiers)
//		ArffSaver saver = new ArffSaver();
//		saver.setInstances(data);
//		
//		File file = new File("/Users/vinoth/Downloads/data_sets1/"+fileName+".arff");
//		if(file.exists())
//			file.delete(); 
//		
//		saver.setFile(new File("/Users/vinoth/Downloads/data_sets1/"+fileName+".arff"));
//		saver.writeBatch();
//		
//		BufferedReader reader =new BufferedReader(new FileReader("/Users/vinoth/Downloads/data_sets1/"+fileName+".arff"));
//		ArffReader arff = new ArffReader(reader);
//		Instances readData = arff.getData();
//		readData.setClassIndex(readData.numAttributes() - 1);
		return data;
	}

}
