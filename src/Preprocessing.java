import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.InterquartileRange;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

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
        saver.setFile(new File("src/data/aad1.arff"));
        saver.writeBatch();

        /* Detect outliers process */
        // load dataset1 for InterquartileRange
        source = new DataSource("src/data/aad1.arff");
        //DataSource source1 = new DataSource("src/data/stl1.arff");
        dataset = source.getDataSet();

        //use a simple filter to remove a certain attribute
        //set up options to remove first and last attribute
        String opt1[] = new String[]{"-R", "first-last", "-O", "3.0", "-E", "6.0", "-P"};
        //create an InterquartileRange object (this is a filter class for outliers and extreme values)
        InterquartileRange range = new InterquartileRange();
        //set the filter option
        range.setOptions(opt1);
        //pass dataset to the filter
        range.setInputFormat(dataset);
        //apply filter
        newData = Filter.useFilter(dataset, range);

        //save the dataset to the new file
        saver.setInstances(newData);
        saver.setFile(new File("src/data/aad2.arff"));
        saver.writeBatch();

        /* Remove the outliers and extreme value table of attributes that only have 1 value in outlier */
        // load dataset
        source = new DataSource("src/data/aad2.arff");
        dataset = source.getDataSet();

        //use a simple filter to remove a certain attribute
        //set up options to remove 22nd, 23rd, 24th, 25th attribute
        String opt2[] = new String[]{"-R", "22,23,24,25"};
        //create Remove object (this is a filter class)
        Remove remove = new Remove();
        //set the filter option
        remove.setOptions(opt2);
        //pass the dataset to the filter
        remove.setInputFormat(dataset);
        //apply filter
        newData = Filter.useFilter(dataset, remove);

        //save the dataset to the new file
        saver.setInstances(newData);
        saver.setFile(new File("src/data/aad3.arff"));
        saver.writeBatch();

        /* Correlation Analysis process */

        /* Discretize Attributes process*/
        source = new DataSource("src/data/aad3.arff");
        dataset = source.getDataSet();

        //use a simple filter to remove a certain attribute
        //set up options to findNumBins, 10 bins, -1.0 desiredWeightOfInstancesPerInterval, 6 binRangePrecision
        String op3[] = new String[]{"-O", "-B", "10", "-M", "-1.0", "-R", "first-last", "-V", "-precision", "6"};
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
        saver.setFile(new File("src/data/aad4.arff"));
        saver.writeBatch();
    }
}
