import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;

import java.io.File;

public class ArffToCsv {
    public static void main(String[] args) throws Exception {

        // load ARFF
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File("src/data/Autism-Adult-Data.arff"));
        Instances data = loader.getDataSet();//get instances object

        // save CSV
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);//set the dataset we want to convert
        //and save as CSV
        saver.setFile(new File("src/data/Autism-Adult-Data.csv"));
        saver.writeBatch();
    }
}
