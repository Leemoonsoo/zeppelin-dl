package com.nflabs.zeppelin.dl;

import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.commons.io.IOUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.zeppelin.helium.Application;
import org.apache.zeppelin.helium.ApplicationArgument;
import org.apache.zeppelin.helium.ApplicationException;
import org.apache.zeppelin.helium.Signal;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterContextRunner;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.apache.zeppelin.interpreter.InterpreterResult.Code;
import org.apache.zeppelin.interpreter.data.ColumnDef;
import org.apache.zeppelin.interpreter.data.TableData;
import org.apache.zeppelin.interpreter.dev.ZeppelinApplicationDevServer;
import org.apache.zeppelin.resource.ResourceInfo;
import org.apache.zeppelin.resource.ResourceKey;
import org.apache.zeppelin.resource.WellKnownResource;

public class DeepLearning  extends Application {
  private SparkContext sc;
  private JavaSparkContext jsc;
  private InterpreterContext context;
  private LinearRegressionModel model;
  private TableData tableData;
  private int labelCol;
  private int valueCol;


  @Override
  protected void onChange(String name, Object oldObject, Object newObject) {
    System.err.println("Change " + name + " : " + oldObject + " -> " + newObject);
    if (name.equals("run") && newObject.equals("running")) {
      int numIterations = Integer.parseInt(this.get(context, "iteration").toString());

      LinkedList<LabeledPoint> vectors = new LinkedList<LabeledPoint>();

      for (int i = 0 ; i < tableData.length(); i++) {
        double label = Double.parseDouble(tableData.getData(i, labelCol).toString());
        double val = Double.parseDouble(tableData.getData(i, valueCol).toString());
        vectors.add(new LabeledPoint(label, Vectors.dense(val)));
      }


      JavaRDD<LabeledPoint> inputVector = jsc.parallelize(vectors).cache();
      this.model = LinearRegressionWithSGD.train(inputVector.rdd(), numIterations);
      this.put(context, "run", "idle");
      this.put(context, "predictBtn", 1);
      this.watch(context, "predictBtn");
    }
    if (name.equals("predictBtn")) {
      double input = Double.parseDouble(newObject.toString());
      double label = model.predict(Vectors.dense(input));
      this.put(context, "predictedValue", label);
    }
  }

  public void refresh() {
    for (InterpreterContextRunner runner : context.getRunners()) {
      if (context.getNoteId().equals(runner.getNoteId()) &&
          context.getParagraphId().equals(runner.getParagraphId())) {
        System.err.println("RUN");
        runner.run();
      }
    }
  }

  @Override
  public void signal(Signal signal) {

  }

  @Override
  public void load() throws ApplicationException, IOException {

  }

  @Override
  public void run(ApplicationArgument arg, InterpreterContext context) throws ApplicationException,
      IOException {

    // load resource from classpath
    context.out.writeResource("dl/DeepLearning.html");


    this.put(context, "run", "idle");
    this.put(context, "iteration", 10);
    this.watch(context, "run");

    this.put(context, "predictBtn", 0);
    this.watch(context, "predictBtn");
    this.put(context, "predictInput", 0);
    this.put(context, "predictedValue", 0);

    this.context = context;


    // get TableData
    tableData = (TableData) context.getResourcePool().get(
        arg.getResource().location(), arg.getResource().name());

    // get spark context
    Collection<ResourceInfo> infos = context.getResourcePool().search(
        WellKnownResource.SPARK_CONTEXT.type() + ".*");
    if (infos == null || infos.size() == 0) {
      throw new ApplicationException("SparkContext not available");
    }

    Iterator<ResourceInfo> it = infos.iterator();
    while (it.hasNext()) {
      ResourceInfo info = it.next();
      sc = (SparkContext) context.getResourcePool().get(info.name());
      if (sc != null) {
        break;
      }
    }

    jsc = new JavaSparkContext(sc);

    labelCol = -1;
    valueCol = -1;
    // find first numeric column
    ColumnDef[] columnDef = tableData.getColumnDef();


    for (int c = 0; c < columnDef.length; c++) {
      try {
        Float.parseFloat((String) tableData.getData(0, c));
        if (labelCol == -1) {
          labelCol = c;
          continue;
        } else {
          valueCol = c;
          break;
        }
      } catch (Exception e) {
        continue;
      }
    }

    if (labelCol == -1 || valueCol == -1) {
      throw new ApplicationException("Numeric column not found");
    }
  }

  @Override
  public void unload() throws ApplicationException, IOException {

  }


  private static String generateData() throws IOException {
    InputStream ins = ClassLoader.getSystemResourceAsStream("dl/mockdata.txt");
    String data = IOUtils.toString(ins);
    return data;
  }

  /**
   * Development mode
   * @param args
   * @throws Exception
   */
  public static void main(String [] args) throws Exception {
    // create development server
    ZeppelinApplicationDevServer dev = new ZeppelinApplicationDevServer(DeepLearning.class.getName());

    TableData tableData = new TableData(new InterpreterResult(Code.SUCCESS, generateData()));

    dev.server.getResourcePool().put("tabledata", tableData);

    // set application argument
    ApplicationArgument arg = new ApplicationArgument(new ResourceKey(
        dev.server.getResourcePoolId(),
        "tabledata"
        ));
    dev.setArgument(arg);


    // set sparkcontext
    // create spark conf
    SparkConf conf = new SparkConf();
    conf.setMaster("local[*]");
    conf.setAppName("Deeplearning");

    // create spark context
    SparkContext sc = new SparkContext(conf);
    dev.server.getResourcePool().put(WellKnownResource.SPARK_CONTEXT.type() + "#aaa", sc);

    // start
    dev.server.start();
    dev.server.join();
  }
}
