import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Classification {
    public static String path = "src/data/Autism-Adult-Data-binary.arff";
    private static DataSource getSource(String path) throws Exception{
        return new DataSource(path);
    }
    private static Instances getData(String path) throws Exception {
        DataSource source = getSource(path);
        return source.getDataSet();
    }
    public static void main(String[] args) throws Exception{
        Instances data = getData(path);
        data.setClassIndex(data.numAttributes() - 1);

        Evaluation eval = new Evaluation(data);

        NaiveBayes nbModel = NaiveBayesClassifier();
        eval.crossValidateModel(nbModel, data, 10, new Random(1));

        System.out.println("Naive Bayes Model");
        System.out.println(nbModel);

        eval.toMatrixString("=== Confusion Matrix ===\n");

        J48 j48model = J48Classifier();
        j48model.buildClassifier(data);
        eval.crossValidateModel(j48model, data, 10, new Random(1));

        System.out.println("J48 Model");
        System.out.println(j48model);

        eval.toMatrixString("=== Confusion Matrix ===\n");

        ZeroR baseline = ZeroRClassifier();
        baseline.buildClassifier(data);
        eval.crossValidateModel(baseline, data, 10, new Random(1));

        System.out.println("ZeroR Model");
        System.out.println(baseline);
        eval.toMatrixString("=== Confusion Matrix ===\n");
    }

    public static NaiveBayes NaiveBayesClassifier() throws Exception{
        NaiveBayes model = new NaiveBayes();
        String option[] = new String[]{"-K"};
        model.setOptions(option);

        return model;
    }

    public static J48 J48Classifier() throws Exception{
        J48 model = new J48();
        String option[] = new String[]{"-U", "-C", "0.25", "M", "2"};
        model.setOptions(option);

        return model;
    }

    private static ZeroR ZeroRClassifier() throws Exception{
        return new ZeroR();
    }

//    public static void printNaiveBayes(){
//        // Get NaiveBayes
//    NaiveBayes nb = NaiveBayesClassifier();
//
//    // Train Naive Bayes
//    nb.buildClassifier(data);
//
//    // Output generated model
//    System.out.println(nb);
//    }


}
