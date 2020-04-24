import weka.attributeSelection.CorrelationAttributeEval;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.File;

public class Preprocessing {
    public static void main(String[] args) throws Exception {
        /* Handle Missing Value Process */
        // load dataset
        DataSource source = new DataSource("src/data/Autism-Adult-Data.arff");
        Instances dataset = source.getDataSet();

        //create ReplaceMissingValues object (this is filter class)
        ReplaceMissingValues missingValues = new ReplaceMissingValues();
        //pass dataset to the filter
        missingValues.setInputFormat(dataset);
        //apply filter
        Instances newData = Filter.useFilter(dataset, missingValues);

        //save the dataset to a new file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File("src/data/missing.arff"));
        saver.writeBatch();

        /* Detect outliers and extreme values process */
        // load dataset1 for InterquartileRange
        DataSource source1 = new DataSource("src/data/missing.arff");
        //DataSource source1 = new DataSource("src/data/stl1.arff");
        Instances dataset1 = source1.getDataSet();

        //use a simple filter to remove a certain attribute
        //set up options to remove first and last attribute
        String opt1[] = new String[]{"-R", "first-last", "-O", "3.0", "-E", "6.0"};
        //create an InterquartileRange object (this is a filter class for outliers and extreme values)
        InterquartileRange range = new InterquartileRange();
        //set the filter option
        range.setOptions(opt1);
        //pass dataset to the filter
        range.setInputFormat(dataset1);
        //apply filter
        Instances rangedData = Filter.useFilter(dataset1, range);

        //save the dataset to the new file
        saver.setInstances(rangedData);
        saver.setFile(new File("src/data/ranged.arff"));
        saver.writeBatch();

        /* Remove the outliers and extreme value table of attributes that only have 1 value in outlier */
        // load dataset
        DataSource source2 = new DataSource("src/data/ranged.arff");
        Instances dataset2 = source2.getDataSet();

        String optOutlier[] = new String[]{"-S", "0.0", "-C", "22", "-L", "last"};
        String optExtreme[] = new String[]{"-S", "0.0", "-C", "23", "-L", "last"};

        RemoveWithValues removeOutlier = new RemoveWithValues();
//        removeOutlier.setAttributeIndex("22");
//        removeOutlier.setNominalIndices("last");
        removeOutlier.setOptions(optOutlier);
        removeOutlier.setInputFormat(dataset2);
        Instances removedOutlierData = Filter.useFilter(dataset2, removeOutlier);
        saver.setInstances(removedOutlierData);
        saver.setFile(new File("/src/data/outlierRemoved.arff"));

        DataSource source3 = new DataSource("src/data/outlierRemoved.arff");
        Instances dataset3 = source3.getDataSet();
        RemoveWithValues removeExtremeValue = new RemoveWithValues();
//        removeExtremeValue.setAttributeIndex("23");
//        removeExtremeValue.setNominalIndices("last");
        removeExtremeValue.setOptions(optExtreme);
        removeExtremeValue.setInputFormat(dataset3);
        Instances extreme = Filter.useFilter(dataset3, removeExtremeValue);
        saver.setInstances(extreme);
        saver.setFile(new File("/src/data/extremeRemoved.arff"));


        source = new DataSource("src/data/extremeRemoved.arff");
        dataset = source.getDataSet();
        //use a simple filter to remove a certain attribute
        //set up options to remove 22nd, 23rd attributes (Outlier and Extreme Value column)
        String opt2[] = new String[]{"-R", "22,23"};
        //create Remove object (this is a filter class)
        Remove remove = new Remove();
        //set the filter option
        remove.setOptions(opt2);
        //pass the dataset to the filter
        remove.setInputFormat(dataset);
        //apply filter
        newData = Filter.useFilter(dataset, remove);

        // save the dataset to the new file
        saver.setInstances(newData);
        saver.setFile(new File("src/data/final.arff"));
        saver.writeBatch();

        /* Correlation Analysis process */
        //load dataset
        source = new DataSource("src/data/final.arff");
        dataset = source.getDataSet();

        CorrelationAttributeEval cEval= new CorrelationAttributeEval();
        // String rename[]={"A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","Age","Gender","Ethnicity","Jundice","Austim","Country","Used_App_Before","Result","Age_desc","Relation","ASD"};
        System.out.print("        ");
        for (int i=dataset.numAttributes()-1;i>=0;i--){
            if (i<=10){
                System.out.print("A["+i+"]"+"  a");
            }
            else System.out.print("A["+i+"]"+" ");
        }
        System.out.println("");
        for( int i= dataset.numAttributes()-1; i>=0; i--){
            dataset.setClassIndex(i);
            cEval.buildEvaluator(dataset);
            for (int j=0; j<=i;j++) {
                if (j==0){
                    if (i>=10) {
                        System.out.print("A[" + i + "]" + "   " + String.format("%.2f", cEval.evaluateAttribute(j)) + "   ");
                    }
                    else System.out.print("A[" + i + "]" + "    " + String.format("%.2f", cEval.evaluateAttribute(j)) + "   ");
                }
                else System.out.print(String.format("%.2f",cEval.evaluateAttribute(j)) + "  " );
            }
            System.out.println("\t");
        }



        //PrincipalComponents pcom= new PrincipalComponents();
        //pcom.getCorrelationMatrix();

        /* Discretize Attributes process*/
        source = new DataSource("src/data/aad.arff");
        dataset = source.getDataSet();

        //use a simple filter to remove a certain attribute
        //set up options to findNumBins, 10 bins, -1.0 desiredWeightOfInstancesPerInterval, 6 binRangePrecision
        String op3[] = new String[]{"-O", "-B", "10", "-M", "-1.0", "-R", "first-last", "-precision", "6"};
        //create Discretize object (this is a filter object)
        Discretize discretize = new Discretize();
        //set the filter option
        discretize.setOptions(op3);
        //pass the dataset to the filter
        discretize.setInputFormat(dataset);
        //apply filter
        newData = Filter.useFilter(dataset, discretize);

        //save the dataset to the new file
        saver.setInstances(newData);
        saver.setFile(new File("src/data/discretized.arff"));
        saver.writeBatch();
    }
}

