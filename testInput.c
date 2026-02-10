/*
 * author: Samuel Tong
 * date: December 7, 2025
 *
 * Optimization of the minimization of the error function for
 * an N-layer network using the Backpropagation algorithm
 *
 * Functionality for reading network config info,
 * randomized weight values, independently running/training
 * the network, saving/loading weight values
 *
 * Training requires feedforward run for train, which saves theta values of
 * the sum of the dot products of activation and weights, also
 * saving psi values to perform backpropagation optimization algorithm
 * to minimize the error between the output values and a given truth table
 *
 *
 * Table of Contents:
 * -------------------------------------------------------------------------
 * public void init(String[] cmdln)
 * private void config(String cmdln)
 * private void echo()
 * private void allocate()
 * private void populate()
 * private void setTestCases()
 * private void setTruthTable()
 * private void initRand()
 * private double randomNum(double randLow, double randHigh)
 * private void loadActivation(double[] tc)
 * public void train()
 * private String reportResults()
 * private void runForRun()
 * private double runForTrain(int testcase)
 * public void run()
 * public void reportRun()
 * private boolean saveWeights(String filename)
 * private boolean loadWeights(String filename)
 * private double sigmoid(double x)
 * private double dSigmoid(double x)
 * private double tanh(double x)
 * private double dtanh(double x)
 * private double activationFn(double x)
 * private double dActivationFn(double x)
 * public void execute()
 * public static void main(String[] args)
 * -------------------------------------------------------------------------
 */

import javax.xml.crypto.Data;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class NLayerNetwork
{
   private int[] activationCounts;     // stores the number of nodes in each layer

   private static final int INPUT = 0; // index of input layer
   private int numActivationLayers;
   private int outputLayerIndex;
   private int finalHiddenLayerIndex;

/*
 * network arrays
 */

   private double[][] a;
   private double[][][] weights;
   private double[][] theta;
   private double[][] psi;

   private int numTestCases;        // number of test cases
   private double[][] testCases;    // test cases to fill activation layer
   private double[][] truthTable;   // truth values for each test case

   private double runResult[][];    // results of only running

   private double randLow;          // RNG lower bound
   private double randHigh;         // RNG upper bound
   private int maxIterations;       // max iterations for training
   private double errorThreshold;   // max acceptable average error
   private double lambda;           // learning factor
   private int keepAlive;        // iterations between messages

   private boolean training;        // training mode
   private boolean running;         // running mode
   private boolean loading;         // loading weights
   private boolean saving;          // saving weights

   private String fileIn;           // input file name
   private String fileOut;          // output file name
   private String fileTestCase;     // test cases file name
   private String fileTruthTable;   // truthtable file name
   private String fileConfig;       // name of the config file
   private boolean fileBinary;      // using binary testcase files
   private boolean printTestCases;  // print testcases when reporting results

   private int iteration;           // iterations needed for training
   private double averageError;     // average error of the network

   private double trainTime;        // time to train the network (ms)
   private double runTime;          // time to run the network (nms)

   private static final double NANO_TO_MILLI = 1000000.0; // convert nano -> millisecond
   private static final int CONFIG_ARGUMENT_INDEX = 0; // access config in "javac ABC...java config.txt"

/**
 * initializes all configurable parameters
 * @param cmdln command line execution string
 * @postcondition configured settings, echoed settings, allocate arrays
 *                populate test cases, truth table, weight values
 */
   public void init(String[] cmdln)
   {
      config(cmdln[CONFIG_ARGUMENT_INDEX]);
      echo();
      allocate();
      populate();
   }

/**
 * sets configurable values
 * @param cmdln config file path
 * @postcondition config values initialized to user specified values
 */
   private void config(String cmdln)
   {
      Path p = Paths.get(cmdln);
      String line;
      String[] parse;
      Map<String, String> read = new LinkedHashMap<String, String>();

      this.fileConfig = cmdln;

      try (BufferedReader br = Files.newBufferedReader(p))
      {
         while ((line = br.readLine()) != null)
         {
            parse = line.split(",");
            if (parse.length == 2)
            {
               read.put(parse[0], parse[1]);
            }
         } // while ((line = br.readLine()) != null)

         try
         {
            String[] network = read.get("layer").split("-");

            this.numActivationLayers = network.length;
            this.outputLayerIndex = this.numActivationLayers - 1;
            this.finalHiddenLayerIndex = this.outputLayerIndex - 1;

            this.activationCounts = new int[numActivationLayers];

            for (int layer = 0; layer < numActivationLayers; layer++)
            {
               this.activationCounts[layer] = Integer.parseInt(network[layer]);
            }

            this.numTestCases = Integer.parseInt(read.get("num_test_cases"));
            this.running = read.get("is_running").equals("y");
            this.saving = read.get("is_saving").equals("y");
            this.loading = read.get("is_loading").equals("y");
            this.fileBinary = read.get("binarytc").equals("y");
            this.printTestCases = read.get("printtc").equals("y");

            if (saving)
            {
               this.fileOut = read.get("weights_out_file");
            }

            if (loading)
            {
               this.fileIn = read.get("weights_in_file");
            }

            this.fileTestCase = read.get("testcase_file");
            this.training = read.get("is_training").equals("y");

            if (training)
            {
               this.randLow = Double.parseDouble(read.get("random_lower_bound"));
               this.randHigh = Double.parseDouble(read.get("random_upper_bound"));
               this.maxIterations = Integer.parseInt(read.get("max_iterations"));
               this.errorThreshold = Double.parseDouble(read.get("error_threshold"));
               this.lambda = Double.parseDouble(read.get("lambda"));

               this.fileTruthTable = read.get("truthtable_file");

               this.keepAlive = Integer.parseInt(read.get("keepalive"));
            } // if (training)
         } // try
         catch (Exception e)
         {
            System.err.printf("exception in file reading: %s", e.getMessage());
         }
      } // try (BufferedReader br = Files.newBufferedReader(p))
      catch (IOException ioe)
      {
         System.err.printf("CONFIG reading file exception: error: %s", ioe.getMessage());
      }
   } // private void config(String cmdln)

/**
 * prints values of all configurable values
 */
   private void echo()
   {
      System.out.println("------------------------------");
      System.out.println("CONFIG ECHO\n");
      System.out.printf("READING CONFIG FROM: %s\n", fileConfig);

      for (int layer = 0; layer < numActivationLayers; layer++)
      {
         System.out.printf("%d-", activationCounts[layer]);
      }
      System.out.println("network\n");

      System.out.printf("using random weights = %s\n", !loading);
      System.out.printf("training true/false = %b\n", training);
      System.out.printf("running true/false = %b\n", running);

      if (training)
      {
         System.out.printf("rand range: [%.2f, %.2f]\n", randLow, randHigh);
         System.out.printf("max iterations = %d\n", maxIterations);
         System.out.printf("error threshold = %.5f\n", errorThreshold);
         System.out.printf("lambda = %.2f\n", lambda);
         System.out.printf("getting truth table from: %s\n", fileTruthTable);
         System.out.printf("keep alive (iterations between messages) = %d\n", this.keepAlive);
      } // if (training)

      System.out.printf("getting %d test cases from %s\n", numTestCases, fileTestCase);
      System.out.printf("using binary test case file: %b\n", fileBinary);

      if (loading)
      {
         System.out.printf("loading weights from: %s\n", fileIn);
      }

      if (saving)
      {
         System.out.printf("saving weights into %s\n", fileOut);
      }

      System.out.println("------------------------------");
   } // private void echo()

/**
 * allocates memory for network arrays
 * @precondition positive/non zero array lengths
 * @postcondition array memory allocated
 */
   private void allocate()
   {
/*
 * initialize all the activation layers
 */
      a = new double[numActivationLayers][];
      for (int layer = 0; layer < numActivationLayers; layer++)
      {
         a[layer] = new double[activationCounts[layer]];
      }
/*
 * initialize the arrays that hold the weights
 * that connect previous activation layer to the next
 */
      weights = new double[numActivationLayers - 1][][];
      for (int layer = 0; layer < numActivationLayers - 1; layer++)
      {
         weights[layer] = new double[activationCounts[layer]][activationCounts[layer+1]];
      }

/*
 * initialize the testcase array
 * that fills the first activation layer (input layer)
 */
      int n1 = INPUT;
      testCases = new double[numTestCases][activationCounts[n1]];

/*
 * initialize the arrays for training
 * theta arrays for the second and third network arrays (starting from array index 1)
 * psi arrays for the third and fourth network arrays (starting from array index 2)
 * truth table array that holds the true value of each testcase
 */
      if (training)
      {
         theta = new double[numActivationLayers - 1][];
         for (int layer = 1; layer < numActivationLayers - 1; layer++)
         {
            theta[layer] = new double[activationCounts[layer]];
         }

         psi = new double[numActivationLayers][];
         for (int layer = 2; layer < numActivationLayers; layer++)
         {
            psi[layer] = new double[activationCounts[layer]];
         }

         int n2 = outputLayerIndex;
         truthTable = new double[numTestCases][activationCounts[n2]];
      } // if (training)

      if (running)
      {
         int n3 = outputLayerIndex;
         runResult = new double[numTestCases][activationCounts[n3]];
      }

   } // private void allocate()

/**
 * sets test cases, truth table, weights to numerical values
 * @precondition memory allocated for arrays
 * @postconditions values set
 */
   private void populate()
   {
      if (fileBinary)
      {
         setBinaryTestCases();
      }
      else
      {
         setTestCases();
      }

      if (training)
      {
         setTruthTable();
      }

      if (!loading)
      {
         initRand();
      }
      else
      {
         System.out.println("Loading weights");
      }
   } // private void populate()

/**
 * reads configuration file and initializes the test cases
 * of this network, values to fill the activation nodes with
 *
 * @precondition memory allocated for arrays
 * @postconditions values set
 */
   private void setTestCases()
   {
      Path p = Paths.get(this.fileTestCase);
      String line;
      String[] parse;

      try (BufferedReader br = Files.newBufferedReader(p))
      {
         for (int tc = 0; tc < numTestCases; tc++)
         {
            line = br.readLine();
            parse = line.split(",");

            int n = INPUT;

            for (int m = 0; m < activationCounts[n]; m++)
            {
               testCases[tc][m] = Double.parseDouble(parse[m]);
            }
         } // for (tc = 0; tc < numTestCases; tc++)
      } // try (BufferedReader br = Files.newBufferedReader(p))
      catch (Exception e)
      {
         System.err.printf("SETTEST reading file exception: %s", e.getMessage());
      }
   } // private void setTestCases()

   private void setBinaryTestCases()
   {
      File f = new File(this.fileTestCase);

      try (DataInputStream dis = new DataInputStream(new FileInputStream(f)))
      {
         for (int tc = 0; tc < numTestCases; tc++)
         {
            int n = INPUT;

            for (int m = 0; m < activationCounts[n]; m++)
            {
               testCases[tc][m] = dis.readDouble();
            }
         } // for (int tc = 0; tc < numTestCases; tc++)
      } // try (DataInputStream dis = new DataInputStream(new FileInputStream(f)))
      catch (Exception e)
      {
         System.err.printf("SETBINARY reading file exception: %s", e.getMessage());
      }
   } // private void setBinaryTestCases()

/**
 * sets the truth table for the network to numerical values
 *
 * @precondition memory allocated for arrays, is training
 * @postconditions values set
 */
   private void setTruthTable()
   {
      Path p = Paths.get(this.fileTruthTable);

      try (BufferedReader br = Files.newBufferedReader(p))
      {
         int n = numActivationLayers - 1;
         for (int i = 0; i < activationCounts[n]; i++)
         {
            String line = br.readLine();
            String[] parse = line.split(",");

            for (int tc = 0; tc < numTestCases; tc++)
            {
               truthTable[tc][i] = Double.parseDouble(parse[tc]);
            }
         } // for (int i = 0; i < activationCounts[n]; i++)
      } // try (BufferedReader br = Files.newBufferedReader(p))
      catch (Exception e)
      {
         System.err.printf("setting truth table exception: %s\n", e.getMessage());
         e.printStackTrace();
      }
   } // private void setTruthTable()

/**
 * fills weights with random numbers in user defined bounds
 * @precondition memory allocated for arrays
 * @postcondition values set
 */
   private void initRand()
   {
      for (int n = 0; n < numActivationLayers - 1; n++)
      {
         for (int k = 0; k < activationCounts[n]; k++)
         {
            for (int j = 0; j < activationCounts[n+1]; j++)
            {
               weights[n][k][j] = randomNum(randLow, randHigh);
            } // for (int j = 0; j < activationCounts[n+1]; j++)
         } // for (int k = 0; k < activationCounts[n]; k++)
      } // for (int n = 0; n < numActivationLayers; n++)
   } // private void initRand()

/**
 * generates a random number in given bounds
 * @param randLow lower bound
 * @param randHigh upper bound
 * @return random double in [randLow, randHigh]
 */
   private double randomNum(double randLow, double randHigh)
   {
      return randLow + (randHigh - randLow) * Math.random();
   }

/**
 * fills the activation layer with a test case
 * @param tc test case values
 * @precondition memory allocated for activation layer
 * @postcondition values loaded
 */
   private void loadActivation(double[] tc)
   {
      int n = INPUT;
      for (int m = 0; m < activationCounts[n]; m++)
      {
         a[n][m] = tc[m];
      }
   }

/**
 * trains the network by changing weights to minimize the
 * error of the neural network across all test cases
 * computes the change in each weight to perform an optimized
 * gradient descent to minimize the error
 * @precondition all configs set, initialized, memory allocated,
 *               arrays filled, is in training mode
 * @postcondition weights changed to values that minimize the error
 *                below a threshold or max iterations reached
 */
   public void train()
   {
      int iteration = 0;
      double averageError = Double.MAX_VALUE;

      while (iteration < maxIterations && averageError > errorThreshold)
      {
         double iterationError = 0.0;

         for (int tc = 0; tc < numTestCases; tc++)
         {
            loadActivation(testCases[tc]);
            iterationError += runForTrain(tc);
/*
 * index of first hidden layer is the index of the input layer plus 1
 */
            for (int n = finalHiddenLayerIndex; n > INPUT + 1; n--)
            {
               for (int j = 0; j < activationCounts[n]; j++)
               {
                  double omegaj = 0.0;
                  for (int i = 0; i < activationCounts[n+1]; i++)
                  {
                     omegaj += psi[n+1][i] * weights[n][j][i];
                     weights[n][j][i] += (lambda * a[n][j] * psi[n+1][i]);
                  }
                  psi[n][j] = omegaj * dActivationFn(theta[n][j]);
               } // for (int j = 0; j < activationCounts[n]; j++)
            } // for (int n = OUT - 1; n > INPUT + 1; n--)

            int n = INPUT + 1;
            for (int k = 0; k < activationCounts[n]; k++)
            {
               double omegak = 0.0;
               for (int J = 0; J < activationCounts[n+1]; J++)
               {
                  omegak += psi[n+1][J] * weights[n][k][J];
                  weights[n][k][J] += (lambda * a[n][k] * psi[n+1][J]);
               }

               double psik = omegak * dActivationFn(theta[n][k]);

               int n1 = INPUT;
               for (int m = 0; m < activationCounts[n1]; m++)
               {
                  weights[n1][m][k] += (lambda * a[n1][m] * psik);
               }
            } // for (int k = 0; k < activationCounts[n]; k++)
         } // for (int tc = 0; tc < numTestCases; tc++)

         iterationError /= 2.0;
         averageError = iterationError / (double) (numTestCases);
         iteration++;

         if ((keepAlive != 0) && ((iteration % keepAlive) == 0))
         {
            System.out.printf("Iteration %d, Error = %f\n", iteration, averageError);
         }
      } // while (iteration <= maxIterations && averageError > errorThreshold)

      this.iteration = iteration;
      this.averageError = averageError;
   } // public void train()

/**
 * reports the result of training
 * @precondition train() has been run
 * @return reason for stopping, how many iterations trained,
 *         average error of the network after training
 */
   private String reportResults()
   {
      String report = "";

      if (this.iteration >= this.maxIterations)
      {
         report += ("max iterations reached; iterations: " + this.iteration + "\n");
      }
      else
      {
         report += this.iteration + " iterations reached\n";
      }

      if (this.averageError > this.errorThreshold)
      {
         report += ("training failed; error is: " + this.averageError + "\n");
      }
      else
      {
         report += ("training successful; error is: " + this.averageError + "\n");
      }

      report += ("training took " + this.trainTime + " milliseconds\n");

      return report;
   } // private String reportResults()

/**
 * forward pass on the neural network for running
 * dots the data in the activation layers and the weights arrays
 * @precondition activation layers and weights initialized
 * @postcondition hidden layers populated, output value set
 */
   private void runForRun()
   {
      for (int n = 1; n < numActivationLayers; n++)
      {
         for (int j = 0; j < activationCounts[n]; j++)
         {
            double theta = 0.0;
            for (int k = 0; k < activationCounts[n-1]; k++)
            {
               theta += a[n-1][k] * weights[n-1][k][j];
            } // for (int k = 0; k < activationCounts[n-1]; k++)
            a[n][j] = activationFn(theta);
         } // for (int j = 0; j < activationCounts[n]; j++)
      } // for (int n = 1; n < numActivationLayers; n++)
   } // private void runForRun()

/**
 * forward pass on the neural network for training
 * dots the data in the activation layers and the weights arrays
 * @precondition activation layers and weights initialized
 * @postcondition hidden layers populated, output value set
 * @param testcase which testcase in the activation
 * @return error for the training cycle
 */
   private double runForTrain(int testcase)
   {
      double error = 0.0;

/*
 * number of activation layers minus one is the output layer
 * minus another one is the final hidden layer
 */

      for (int n = 1; n <= finalHiddenLayerIndex; n++)
      {
         for (int j = 0; j < activationCounts[n]; j++)
         {
            double thetaNJ = 0.0;
            for (int k = 0; k < activationCounts[n-1]; k++)
            {
               thetaNJ += a[n-1][k] * weights[n-1][k][j];
            } // for (int k = 0; k < activationCounts[n-1]; k++)
            a[n][j] = activationFn(thetaNJ);
            theta[n][j] = thetaNJ;
         } // for (int j = 0; j < activationCounts[n]; j++)
      } // for (int n = 1; n <= finalHiddenLayerIndex; n++)

      int n = outputLayerIndex;
      for (int i = 0; i < activationCounts[n]; i++)
      {
         double thetai = 0.0;
         for (int j = 0; j < activationCounts[n-1]; j++)
         {
            thetai += a[n-1][j] * weights[n-1][j][i];
         }
         a[n][i] = activationFn(thetai);

         double omegai = truthTable[testcase][i] - a[n][i];
         psi[n][i] = omegai * dActivationFn(thetai);
         error += (omegai * omegai);
      } // for (int i = 0; i < activationCounts[n]; i++)

      return error;
   } // private double runForTrain(int testcase)

/**
 * runs neural net on set weight values for all test cases
 * gets calculated values for each test case
 */
   public void run()
   {
      for (int tc = 0; tc < numTestCases; tc++)
      {
         loadActivation(testCases[tc]);
         runForRun();

         int n = outputLayerIndex;
         for (int i = 0; i < activationCounts[n]; i++)
         {
            runResult[tc][i] = a[n][i];
         }
      } // for (int tc = 0; tc < numTestCases; tc++)
   } // public void run()

/**
 * report results
 */
   public void reportRun()
   {
      int n;

      System.out.print("inputs\t\t| outputs\t\t");

      if (training)
      {
         System.out.print("| truth table");
      }
      System.out.println("\n------------------------------");

      for (int tc = 0; tc < numTestCases; tc++)
      {
         n = INPUT;
         if (printTestCases)
         {
            for (int i = 0; i < activationCounts[n]; i++)
            {
               System.out.print(testCases[tc][i] + "  ");
            }
         }

         System.out.print("\t| ");

         n = outputLayerIndex;
         for (int i = 0; i < activationCounts[n]; i++)
         {
            System.out.printf("%.2f ", runResult[tc][i]);
         }
         System.out.print("\t\t");

         if (training)
         {
            System.out.print("| ");

            n = outputLayerIndex;
            for (int i = 0; i < activationCounts[n]; i++)
            {
               System.out.printf("%.2f ", truthTable[tc][i]);
            }
         } // if (training)

         System.out.println();
      } // for (int tc = 0; tc < numTestCases; tc++)

      System.out.println("------------------------------");
      System.out.print("running took " + runTime + " milliseconds\n");
   } // public void run()

/**
 * attempts to save the node parameters and weights into a binary file
 * @param filename output file
 * @postcondition saves the weights or throws IOException
 * @return whether or not error occured
 */
   private boolean saveWeights(String filename)
   {
      boolean erred = false;

      try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
      {
         for (int layer = 0; layer < numActivationLayers; layer++)
         {
            dos.writeInt(activationCounts[layer]);
         }

         for (int n = 0; n < numActivationLayers - 1; n++)
         {
            for (int K = 0; K < activationCounts[n]; K++)
            {
               for (int j = 0; j < activationCounts[n + 1]; j++)
               {
                  dos.writeDouble(weights[n][K][j]);
               }
            }
         } // for (int n = 0; n < numActivationLayers; n++)

         System.out.println("saved weights");
      } // try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
      catch (IOException e)
      {
         System.err.printf("error saving weights: %s\n", e.getMessage());
         erred = true;
      }

      return erred;
   } // private void saveWeights(String filename)

/**
 * attempts to load a binary file of node and weight information
 * @param filename input file
 * @postcondition loads the weights into memory or throws IOEcception
 * @return whether error occured
 */
   private boolean loadWeights(String filename)
   {
      boolean erred = false;
      File file = new File(filename);

      if (!file.exists())
      {
         System.err.printf("file %s does not exist\n", filename);
         erred = true;
      }
      else
      {
         try (DataInputStream dis = new DataInputStream(new FileInputStream(file)))
         {
            for (int layer = 0; layer < numActivationLayers; layer++)
            {
               if (activationCounts[layer] != dis.readInt())
               {
                  System.err.printf("file's network format does not match config\n");
                  erred = true;
               }
            }

            if (!erred)
            {
               for (int n = 0; n < numActivationLayers - 1; n++)
               {
                  for (int k = 0; k < activationCounts[n]; k++)
                  {
                     for (int j = 0; j < activationCounts[n + 1]; j++)
                     {
                        weights[n][k][j] = dis.readDouble();
                     }
                  }
               } // for (int n = 0; n < numActivationLayers; n++)
            } // if (!erred)
            System.out.println("weights loaded");

         } // try (DataInputStream dis = new DataInputStream(new FileInputStream(file)))
         catch (IOException e)
         {
            System.err.printf("error loading weights: %s\n", e.getMessage());
            erred = true;
         }
      } // else
      return erred;
   } // private void loadWeights(String filename)

/**
 * sigmoid function
 * @param x (-inf, inf)
 * @return sigmoid of x
 */
   private double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
 * derivative of sigmoid function
 * @param x (-inf, inf)
 * @return d sigmoid(x) / dx
 */
   private double dSigmoid(double x)
   {
      double sig = sigmoid(x);
      return sig * (1.0 - sig);
   } // private double dActivationFn(double x)

/**
 * hyperbolic tangent function
 * @param x (-inf, inf)
 * @return tanh of x
 */
   private double tanh(double x)
   {
      double epsilon = (x < 0.0) ? 1.0 : -1.0;
      double e = Math.exp(epsilon * 2.0 * x);

      double hyptan = epsilon * ((e - 1.0) / (e + 1.0));

      return hyptan;
   } // private double tanh(double x)

/**
 * derivative of hyperbolic tangent function
 * @param x (-inf, inf)
 * @return d tanh(x) / dx
 */
   private double dtanh(double x)
   {
      double hyptan = tanh(x);

      return 1.0 - (hyptan * hyptan);
   } // private double dtanh(double x)

/**
 * activation function to transform the
 * activation state of the network's nodes
 *
 * @param x (-inf, inf)
 * @return the transformed input by the activation function
 */
   private double activationFn(double x)
   {
      return sigmoid(x);
   }

/**
 * derivative of the activation function to transform the
 * activation state of the network's nodes
 *
 * @param x (-inf, inf)
 * @return the transformed input by the
 *         derivative of the activation function
 */
   private double dActivationFn(double x)
   {
      return dSigmoid(x);
   }

/**
 * operational mode switch
 */
   public void execute()
   {
      boolean erred = false;

      if (loading)
      {
         System.out.printf("LOADING WEIGHTS FROM %s\n", fileIn);
         erred = loadWeights(fileIn);
      }

      if (training)
      {
         trainTime = System.nanoTime();

         System.out.print("TRAINING MODE\n");
         train();

         trainTime = System.nanoTime() - trainTime;
         trainTime /= NANO_TO_MILLI;

         System.out.println(reportResults());
      } // if (training)

      if (running)
      {
         System.out.print("RUNNING MODE\n");

         if (!erred)
         {
            runTime = System.nanoTime();

            run();

            runTime = System.nanoTime() - runTime;
            runTime /= NANO_TO_MILLI;

            reportRun();
         } // if (!erred)
      } // if (running)

      if (saving)
      {
         System.out.printf("SAVING WEIGHTS INTO %s\n", fileOut);
         saveWeights(fileOut);
      }
   } // public void execute(boolean training)

/**
 * main executable
 * @param args config file string as first parameter
 */
   public static void main(String[] args)
   {
      NLayerNetwork m = new NLayerNetwork();

      if (args.length == 0)
      {
         args = new String[1];
         args[CONFIG_ARGUMENT_INDEX] = "training.txt";
      }

      m.init(args);
      m.execute();
   } // public static void main(String[] args)
} // public class NLayerNetwork
