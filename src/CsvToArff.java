import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class CsvToArff {
    public static void main(String[] args) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/data/Autism-Adult-Data.csv"));
        Instances data = loader.getDataSet();//get instances object

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);//set the dataset we want to convert
        //and save as ARFF
        saver.setFile(new File("src/data/Autism-Adult-Data.arff"));
        saver.writeBatch();

    }
}
