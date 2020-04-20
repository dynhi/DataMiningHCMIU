import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Classification {
    public static String path = "src/data/final.arff";
    private static DataSource source;
    private static Instances data;
    private static Evaluation eval;

    public static void main(String[] args) throws Exception{
        zeroRClassifier();
        naiveBayesClassifier();
        j48Classifier();

    }

    public static void naiveBayesClassifier() throws Exception{
        source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        NaiveBayes model = new NaiveBayes();
        String option[] = new String[]{"-K"};
        model.setOptions(option);

        eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));
        model.buildClassifier(data);
        System.out.println("=== Naive Bayes Model ===\n");
        System.out.println(model);
        printConfusionMatrix();
    }

    public static void j48Classifier() throws Exception{
        source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        J48 tree = new J48();
        String option[] = new String[4];
        option[0] = "-C";
        option[1] = "0.25";
        option[2] = "-M";
        option[3] = "2";
        tree.setOptions(option);

        eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));
        tree.buildClassifier(data);
        System.out.println("=== J48 Model ===\n");
        System.out.println(tree);
        System.out.println(tree.graph());
        printConfusionMatrix();
    }

    private static void zeroRClassifier() throws Exception{
        source = new DataSource(path);
        data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        ZeroR baseline = new ZeroR();

        eval = new Evaluation(data);
        eval.crossValidateModel(baseline, data, 10, new Random(1));
        baseline.buildClassifier(data);
        System.out.println("=== ZeroR Model ===\n");
        System.out.println(baseline);
        printConfusionMatrix();
    }

    private static void printConfusionMatrix() throws Exception{
        System.out.println();
        System.out.println(eval.toMatrixString("=== Confusion matrix for fold ===\n"));
        System.out.println("Correct % = "+eval.pctCorrect());
        System.out.println("Incorrect % = "+eval.pctIncorrect());
        System.out.println("AUC = "+eval.areaUnderROC(1));
        System.out.println("kappa = "+eval.kappa());
        System.out.println("MAE = "+eval.meanAbsoluteError());
        System.out.println("RMSE = "+eval.rootMeanSquaredError());
        System.out.println("RAE = "+eval.relativeAbsoluteError());
        System.out.println("RRSE = "+eval.rootRelativeSquaredError());
        System.out.println("Precision = "+eval.precision(1));
        System.out.println("Recall = "+eval.recall(1));
        System.out.println("fMeasure = "+eval.fMeasure(1));
        System.out.println("Error Rate = "+eval.errorRate());
    }
}
