/**
 * Elijah L
 * Created 2025-09-06
 * Does all the math related to the arrays for activation states and
 * weights for the n-layer network. Can run and train the network, and 
 * this file's methods are called by main.c. 
 *
 * ==================================================================
 *
 * run.c
 * int printDoubleArray(double* values, int length, bool highPrecision)
 * int allocateMemory()
 * void propagateWeightsRandom()
 * void propagateWeightsManual()
 * int propagateWeightsFromFile()
 * int propagateTestCases()
 * int propagateTestCasesFromFile()
 * void propagateTruthTable()
 * int propagateTruthTableFromFile()
 * void getInputValues()
 * void run()
 * double runWhilstTraining(int iteration)
 * void updateInputActivations(int testCaseIndex)
 * void runTestCases()
 * void callRunTestCases()
 * int train(double* finalError, int* totalIterations)
 * void outputTrainingResult(bool successful, double finalError, int
 *                                              totalIterations)
 * int callTrain()
 * void printTimingInformation()
 * void printNetworkInfo()
 * int saveWeights()
 * void freeMemory()
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "../include/constants.h"
#include "../include/run.h"
#include "../lib/table.h"

/**
 * see documentation in main.c
 */
extern int activationLayers;

extern int* lengths;

extern int testCaseLength;

extern double randomUpperBound;
extern double randomLowerBound;

extern double (*activationFunction)(double input);
extern double (*activationFunctionPrime)(double input);

extern bool training;
extern bool runningTestCases;
extern bool randomWeightPropagation;

extern int maxIterations;
extern double maxAcceptableError;
extern double λ;
extern void (*truthFunction)(double* input, double* output);
extern bool printInputTable;
extern bool printTruthTable;
extern bool outputNetworkInfo;
extern int keepAlive;
extern int reordering;

extern char loadFileName[MAX_FILE_NAME_SIZE];
extern char saveFileName[MAX_FILE_NAME_SIZE];

extern bool readTestCases;
extern char testCasesFileName[MAX_FILE_NAME_SIZE];

extern bool readTruthTable;
extern char truthTableFileName[MAX_FILE_NAME_SIZE];

extern char networkConfiguration[NETWORK_CONFIGURATION_LENGTH];

/**
 * the total number of times that propagateWeightsRandom() has been 
 * called; this is to ensure that, if called multiple times in the
 * same second, it returns different values.
 */
int randomIterations = 0;

/*
 * the array of k, j, and i activation states. Indexes for these arrays
 * are stored as constants.
 */
double** a = NULL; 

/**
 * an array of weights where the second index represents the index of
 * the left layer and the third index represents the index of the 
 * right layer, relative such that the left array would be represented
 * in the a array at an index equivalent to the value of the first 
 * index.
 */
double*** weights = NULL;

/**
 * represents the input activation states for each of the test cases.
 */
double** testCases = NULL;

/**
 * represents the true result for each of the test cases.
 */
double** truthTable = NULL;

/**
 * represents the pointer to the beginning of allocated memory for the
 * Θ arrays.
 */
double** trueΘ = NULL;

/**
 * uses a pointer offset to refer to trueΘ in a way such that the
 * indices of a given column are equivalent to the index to reference 
 * the corresponding column in the a 2d array. As a result, this array
 * is effectively 1-indexed and should not be freed, nor is it
 * allocated itself; all memory allocation should utilize the trueΘ 
 * pointer.
 */
double** Θ = NULL;

/**
 * represents the pointer to the beginning of allocated memory for the
 * Ψ arrays.
 */
double** trueΨ = NULL;

/**
 * represents the array of Ψ arrays for columns aside from the first
 * column, as well as the second column, where the array is
 * not necessary. The indices are also aligned with that of the a
 * array, meaning that the first two elements are empty. For the last
 * column, it actually refers to the lowercase ψ array. Similarly to
 * the Θ array, this array includes a pointer offset to make it
 * effectively 2-indexed in order that the indices line up with those
 * of the a array.
 */
double** Ψ = NULL;

/**
 * time, in seconds, from the time the runTestCases function is called,
 * to the time when its execution finishes.
 */
double runDuration = 0.0;

/**
 * time, in seconds, from the time the train function is called, to the
 * time when its execution finishes.
 */
double trainDuration = 0.0;

/**
 * Neatly prints without a newline at the end a double array of the
 * specified length.
 * Prints 4 decimal places unless highPrecision is true, in which case
 * it prints 17 decimal places.
 */
int printDoubleArray(double* values, int length, bool highPrecision)
{
   if (length == 0)
   {
      printf("[]");
   }
   else
   {
      if (highPrecision)
      {
         printf("[%0.17f", values[0]);
      }
      else
      {
         printf("[%0.4f", values[0]);
      }

      for (int it = 1; it < length; it++)
      {
         if (highPrecision)
         {
            printf(", %0.17f", values[it]);
         }
         else
         {
            printf(", %0.4f", values[it]);
         }
      }
      printf("]");
   } // else [if (length == 0)]
   return 0;
} // int printDoubleArray (double* values, int length)


/**
 * Allocates the arrays for weights (kjWeights and jiWeights), as well
 * as the a, h, and F arrays, and arrays necessary for training, if
 * training.
 */
int allocateMemory()
{
   int returnValue = 0;
   int n;

   a = (double**)calloc(activationLayers, sizeof(double*));

   if (a == NULL)
   {
      returnValue = -1;
   }
   else
   {
      for (int n = 0; n < activationLayers; n++)
      {
         a[n] = (double*)calloc(lengths[n], sizeof(double));
         if (a[n] == NULL) returnValue = -1;
      }
   } // else [if (a == NULL)]

   weights = (double***)malloc(NUM_LAYER_INTERVALS * 
         sizeof(double**));

   if (weights == NULL)
   {
      returnValue = -1;
   }
   else
   {
      for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
      {
         weights[n] = (double**)malloc(lengths[n] * sizeof(double*));

         if (weights[n] == NULL)
         {
            returnValue = -1;
         }
         else
         {
            for (int k = 0; k < lengths[n]; k++)
            {
               weights[n][k] = (double*)malloc(lengths[n + 1] *  sizeof(double));
            }
         }
      } // for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
   } // else [if (weights == NULL)]

   testCases = (double**)malloc(testCaseLength * sizeof(double*));

   if (testCases == NULL) 
   {
      returnValue = -1;
   }
   else
   {
      for (int it = 0; it < testCaseLength; it++)
      {
         testCases[it] = (double*)malloc(lengths[INPUT_LAYER_INDEX] * sizeof(double));
         if (testCases[it] == NULL) returnValue = -1;
      }
   }

   if (training | printTruthTable)
   {
      truthTable = (double**)malloc(testCaseLength * sizeof(double));

      if (truthTable == NULL)
      {
         returnValue = -1;
      }
      else
      {
         for (int it = 0; it < testCaseLength; it++)
         {
            truthTable[it] = (double*)malloc(lengths[OUTPUT_LAYER_INDEX] * sizeof(double));
            if (truthTable[it] == NULL) returnValue = -1;
         }
      }
   } // if (training | printTruthTable)
   
   if (training)
   {
      trueΘ = (double**)malloc(NUM_THETA_LAYERS * sizeof(double*));
      if (trueΘ == NULL)
      {
         returnValue = -1;
      }
      else
      {
         Θ = trueΘ - 1;

         for (int n = INITIAL_THETA_INDEX; n < THETA_ARRAY_LIMIT; n++)
         {
            Θ[n] = (double*)malloc(lengths[n] * sizeof(double));
            if (Θ[n] == NULL) returnValue = -1;
         }
      }

      trueΨ = (double**)malloc(NUM_PSI_LAYERS * sizeof(double*));
      if (trueΨ == NULL)
      {
         returnValue = -1;
      }
      else
      {
         Ψ = trueΨ - 2;

         for (int n = INITIAL_PSI_INDEX; n < PSI_ARRAY_LIMIT; n++)
         {
            Ψ[n] = (double*)malloc(lengths[n] * sizeof(double));
            if (Ψ[n] == NULL) returnValue = -1;
         }
      }
   } // if (training)

   if (returnValue == -1)
   {
      printf(ERROR("Some memory could not be initialized"));
   }
   
   return returnValue;
} // int allocateMemory()

/**
 * Puts the values of the weights to random, specified by the 
 * randomUpperBound and randomLowerBound user-configurable variables.
 */
void propagateWeightsRandom()
{
   srand(time(NULL) + randomIterations);

   for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
   {
      for (int k = 0; k < lengths[n]; k++)
      {
         for (int j = 0; j < lengths[n + 1]; j++)
         {
            weights[n][k][j] = 
               RAND(randomUpperBound, randomLowerBound);
         }
      }
   }

   randomIterations++;
   return;
} // void propagateWeightsRandom()

/**
 * Puts the values of the weights manually, based on user input.
 */
void propagateWeightsManual()
{
   weights[INPUT_LAYER_INDEX][0][0] = 0.5;
   weights[INPUT_LAYER_INDEX][0][1] = 0.5;
   weights[INPUT_LAYER_INDEX][1][0] = 0.5;
   weights[INPUT_LAYER_INDEX][1][1] = 0.5;

   return;
} // void propagateWeightsManual()

/**
 * Puts the values of the weights based on data from the file whose name
 * is specified in the loadFileName variable. Ensures that the input
 * file has the same network configuration as this one.
 */
int propagateWeightsFromFile()
{
   int returnValue = 0;

   FILE* file = fopen(loadFileName, "rb");

   if (file == NULL)
   {
      printf(ERROR("Could not open file with name \"%s\"."), 
            loadFileName);
      returnValue = -1;
   }
   else
   {
      while(fgetc(file) != '\n');
      
      for (int n = 0; n < activationLayers; n++)
      {
         int fileKLength;
         fread(&fileKLength, sizeof(int), 1, file);
         fseek(file, 1, SEEK_CUR); // moves one byte forward
         
         if (fileKLength != lengths[n])
         {
            printf(ERROR("Input Weights File has an incompatable network"
                     " configuration (%d as %dth element incompatable with %s)"),
                  fileKLength, n, networkConfiguration);
            returnValue = -1;
         }
      } // for (int n = 0; n < activationLayers; n++)

      for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
      {
         for (int k = 0; k < lengths[n]; k++)
         {
            if (fread(weights[n][k], sizeof(double), lengths[n + 1], file) < lengths[n + 1])
            {
               printf(ERROR("Could not read %dth %d-%d weight fully"),
                     k, n, n + 1);
               returnValue = -1;
            }

            fseek(file, 1, SEEK_CUR);
         }
      } // for (int n = 0; n < activationLayers - 1; n++)

      fclose(file);
   } // else [if (file == NULL)]

   return returnValue;
} // int propagateWeightsFromFile()

/**
 * Sets the values of the test cases to testing every combination
 * of 0s and 1s for each of the activation states. Only works if
 * testCaseLength is 4 (requiring kLength >= 2) or 8 (requiring
 * kLength >= 3).
 */
int propagateTestCases()
{
   int returnValue = 0;

   if (testCaseLength == 4)
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;
      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;
      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;
      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;
   } 
   else if (testCaseLength == 8)
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;
      testCases[0][2] = 0.0;
      testCases[1][0] = 0.0;
      testCases[1][1] = 0.0;
      testCases[1][2] = 1.0;
      testCases[2][0] = 0.0;
      testCases[2][1] = 1.0;
      testCases[2][2] = 0.0;
      testCases[3][0] = 0.0;
      testCases[3][1] = 1.0;
      testCases[3][2] = 1.0;

      testCases[4][0] = 1.0;
      testCases[4][1] = 0.0;
      testCases[4][2] = 0.0;
      testCases[5][0] = 1.0;
      testCases[5][1] = 0.0;
      testCases[5][2] = 1.0;
      testCases[6][0] = 1.0;
      testCases[6][1] = 1.0;
      testCases[6][2] = 0.0;
      testCases[7][0] = 1.0;
      testCases[7][1] = 1.0;
      testCases[7][2] = 1.0;
   } // else if (testCaseLength == 8) [if (testCaseLength == 4)]
   else
   {
      printf(ERROR("No preset test cases for length %d."), testCaseLength);
      returnValue = -1;
   }

   return returnValue;
} // int propagateTestCases()


int propagateTestCasesFromFile()
{
   int returnValue = 0;

   FILE* file = fopen(testCasesFileName, "rb");

   if (file == NULL)
   {
      printf(ERROR("Could not open test case file."));
      returnValue = -1;
   }
   else
   {
      int fileInputLength = 0;
      fread(&fileInputLength, sizeof(int), 1, file);

      int fileTestCaseLength = 0;
      fread(&fileTestCaseLength, sizeof(int), 1, file);

      if (fileInputLength != lengths[INPUT_LAYER_INDEX])
      {
         printf(ERROR("Input Test Cases File has an incompatable"
               " network configuration (inputLength %d != %d)."),
               fileInputLength, 
               lengths[INPUT_LAYER_INDEX]);
         returnValue = -1;
      }

      if (fileTestCaseLength != testCaseLength)
      {
         printf(ERROR("Input Test Cases File has an incompatable"
               " network configuration (testCaseLength %d != %d)."),
               fileTestCaseLength,
               testCaseLength);
         returnValue = -1;
      }

      for (int it = 0; it < testCaseLength; it++)
      {
         if (fread(testCases[it], sizeof(double), 
                  lengths[INPUT_LAYER_INDEX], file) 
                  < lengths[INPUT_LAYER_INDEX])
         {
            printf(ERROR("Could not successfully fully read %dth "
                   "test case from the file."), it);
            returnValue = -1;
         }
      }
   } // else [if (file == NULL)]
   
   return returnValue;
} // propagateTestCasesFromFile

/**
 * Sets the values of the truth table using the function pointer
 * truthFunction.
 */
void propagateTruthTable()
{
   for (int it = 0; it < testCaseLength; it++)
   {
      truthFunction(testCases[it], truthTable[it]);
   }
   return;
}

/**
 * Sets the value of the truth table using the binary file whose name
 * is specified by truthTableFileName.
 */
int propagateTruthTableFromFile()
{
   int returnValue = 0;

   FILE* file = fopen(truthTableFileName, "rb");

   if (file == NULL)
   {
      printf(ERROR("Could not open truth table file."));
      returnValue = -1;
   }
   else
   {
      int fileOutputLength = 0;
      fread(&fileOutputLength, sizeof(int), 1, file);

      int fileTestCaseLength = 0;
      fread(&fileTestCaseLength, sizeof(int), 1, file);

      if (fileOutputLength != lengths[OUTPUT_LAYER_INDEX])
      {
         printf(ERROR("Input Truth Table File has an incompatable"
               " network configuration (outputLength %d != %d)."),
               fileOutputLength, 
               lengths[OUTPUT_LAYER_INDEX]);
         returnValue = -1;
      }

      if (fileTestCaseLength != testCaseLength)
      {
         printf(ERROR("Input Test Cases File has an incompatable"
               " test case length (testCaseLength %d != %d)."),
               fileTestCaseLength,
               testCaseLength);
         returnValue = -1;
      }

      for (int it = 0; it < testCaseLength; it++)
      {
         if (fread(truthTable[it], sizeof(double), 
                  lengths[OUTPUT_LAYER_INDEX], file) 
                  < lengths[OUTPUT_LAYER_INDEX])
         {
            printf(ERROR("Could not successfully fully read %dth "
                   "test case from the file."), it);
            returnValue = -1;
         }
      }
   } // else [if (file == NULL)]
   fclose(file);
   return returnValue;
} // int propagateTruthTableFromFile()

/**
 * fills in the a array with user specified input values;
 * this is called by the run function to test a specific set of input.
 */
void getInputValues()
{
   a[INPUT_LAYER_INDEX][0] = 0.0;
   a[INPUT_LAYER_INDEX][1] = 1.0;
   return;
}

/**
 * Runs the network using the kjWeights and jiWeights defining the h
 * array and f as the final result, using the specified
 * activationFunction.
 */
void run()
{
   for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
   {
      for (int j = 0; j < lengths[n + 1]; j++)
      {
         double Θ = 0.0;
         for (int k = 0; k < lengths[n]; k++)
         {
            Θ += a[n][k] * weights[n][k][j];
         }
         a[n + 1][j] = activationFunction(Θ);
      }
   }
   
   return;
} // void run()

/**
 * Same as run(), but updates the Θ array
 * because of the necessities of training
 *
 * Returns the error calculated for this iteration.
 */
double runWhilstTraining(int iteration)
{
   double ESum = 0.0;
   int n;
   double Θj;
   double ω;

   for (n = 0; n < NUM_LAYER_INTERVALS_EXEMPTING_THE_LAST; n++)
   {
      for (int j = 0; j < lengths[n + 1]; j++)
      {
         Θj = 0.0;
         for (int k = 0; k < lengths[n]; k++)
         {
            Θj += a[n][k] * weights[n][k][j];
         }

         Θ[n + 1][j] = Θj;
         a[n + 1][j] = activationFunction(Θj);
      }
   } // for (n = 0; n < activationLayers - 2; n++)

   n = OUTPUT_LAYER_INDEX - 1;

   for (int j = 0; j < lengths[n + 1]; j++)
   {
      Θj = 0.0;
      for (int k = 0; k < lengths[n]; k++)
      {
         Θj += a[n][k] * weights[n][k][j];
      }

      a[n + 1][j] = activationFunction(Θj);
      ω = truthTable[iteration][j] - a[n + 1][j];
      Ψ[n + 1][j] = ω * activationFunctionPrime(Θj);

      ESum += ω * ω;
   } // for (int i = 0; i < iLength; i++)
   return ESum / 2.0;
} // void runWhilstTraining()

/**
 * Sets the values of the input activation states to those specified
 * in the testCases array at the input index.
 */
void updateInputActivations(int testCaseIndex)
{
   for (int k = 0; k < lengths[INPUT_LAYER_INDEX]; k++)
   {
      a[INPUT_LAYER_INDEX][k] = testCases[testCaseIndex][k];
   }
   return;
}


/**
 * Gets the output value from the network for each of the possible
 * test cases, creating a chart. Has the option of printing the input
 * of each test case, as well as the value according to the truth table.
 */
void runTestCases()
{
   int outputLayerLength = lengths[OUTPUT_LAYER_INDEX];

   Table outputs = bareInitializeTable(outputLayerLength);

   zeroHeaders(outputs);

   outputs.headers[0] = "Output";

   setAllColumnTypes(outputs, float_2x2);

   groupAllColumns(outputs);

   updateWidths(outputs);


   Table finalTable = outputs;

   if (printInputTable)
   {
      int inputLayerLength = lengths[0];
      Table inputs = bareInitializeTable(inputLayerLength);

      zeroHeaders(inputs);
      inputs.headers[0] = "Inputs";


      setAllColumnTypes(inputs, float_2x4);

      groupAllColumns(inputs);

      updateWidths(inputs);

      finalTable = appendTables(inputs, finalTable, DONE_TABLE);
   } // if (printInputTable)

   
   if (printTruthTable)
   {
      int outputLayerLength = lengths[OUTPUT_LAYER_INDEX];
      Table truths = bareInitializeTable(outputLayerLength);
      
      zeroHeaders(truths);
      truths.headers[0] = "Truth";

      setAllColumnTypes(truths, float_2x2);

      groupAllColumns(truths);

      updateWidths(truths);

      finalTable = appendTables(finalTable, truths, DONE_TABLE);
   } // if (printTruthTable)
   
   Table indices = defineTable(DEFAULT_CONFIGS, 1, 0, "", int16);
   
   finalTable = appendTables(indices, finalTable, DONE_TABLE);


   printHeader(finalTable);

   int trueIteration;
   for (int it = 0; it < testCaseLength; it++)
   {
      trueIteration = it;

      if (reordering != 0)
      {
         trueIteration = reordering * (it % reordering) + it / reordering;
      }


      updateInputActivations(trueIteration);
      run();

      printEntry(finalTable, trueIteration);

      if (printInputTable)
      {
         for (int k = 0; k < lengths[INPUT_LAYER_INDEX]; k++)
         {
            printEntry(finalTable, a[INPUT_LAYER_INDEX][k]);
         }
      }

      for (int k = 0; k < lengths[OUTPUT_LAYER_INDEX]; k++)
      {
         printEntry(finalTable, a[OUTPUT_LAYER_INDEX][k]);
      }

      if (printTruthTable)
      {
         for (int k = 0; k < lengths[OUTPUT_LAYER_INDEX]; k++)
         {
            printEntry(finalTable, truthTable[trueIteration][k]);
         }
      }
   } // for (int it = 0; it < testCaseLength; it++)

   freeTable(finalTable);

   return;
} // void runTestCases()

/**
 * Calls the runTestCases function, setting up the timing functionality
 * and saving it to the variable runDuration.
 */
void callRunTestCases()
{
   double initialTime = clock();

   runTestCases();
   
   runDuration = 
      (clock() - initialTime) / NUMBER_OF_MILLISECONDS_IN_A_SECOND;
   
   return;
} // void callRunTestCases()

/**
 * Runs a training algorithm of gradient descent, with optional logging
 * features.
 * It runs until a max number of iterations, iterating through every
 * binary input combination. If the error, averaged through the test
 * cases, is less than the specified maxAcceptableError, then the
 * network finishes successfully; if it doesn't before running the
 * max number of iterations, it finishes unsuccessfully
 *
 * This function returns 1 if training fails, 0 if training succeeds,
 * and -1 if training could not finish because of some error.
 */
int train(double* finalError, int* totalIterations)
{
   int returnValue = 0;

   int iterationsCount = 0;

   double EAverage = 0.0;
   double ESum = 0.0;

   int n;

   double Ω;
   double Ψk;

   Table table;

   if (keepAlive != 0)
   {
      table = defineTable(DEFAULT_CONFIGS, 2, 0, "Iterations", int32, "Error", float_4x8);
      printHeader(table);
   }

   while (iterationsCount <= maxIterations && 
         (iterationsCount == 0 || EAverage > maxAcceptableError))
   {
      iterationsCount++;

      for (int it = 0; it < testCaseLength; it++)
      {
         updateInputActivations(it);

         ESum += runWhilstTraining(it);


         for (n = PENULTIMATE_LAYER; n >= 2; n--)
         {
            for (int k = 0; k < lengths[n]; k++)
            {
               Ω = 0.0;
               for (int j = 0; j < lengths[n + 1]; j++)
               {
                  Ω += Ψ[n + 1][j] * weights[n][k][j];
                  weights[n][k][j] += λ * a[n][k] * Ψ[n + 1][j];
               }

               Ψ[n][k] = Ω * activationFunctionPrime(Θ[n][k]);
            }
         } // for (n = activationLayers - 2; n >= 2; n--)

         n = 1;
         for (int k = 0; k < lengths[n]; k++)
         {
            Ω = 0.0;
            for (int j = 0; j < lengths[n + 1]; j++)
            {
               Ω += Ψ[n + 1][j] * weights[n][k][j];

               weights[n][k][j] += λ * a[n][k] * Ψ[n + 1][j];
            }
            
            Ψk = Ω * activationFunctionPrime(Θ[n][k]);

            for (int m = 0; m < lengths[n - 1]; m++)
            {
               weights[INPUT_LAYER_INDEX][m][k] += 
                  λ * a[INPUT_LAYER_INDEX][m] * Ψk;
            }
         } // for (int k = 0; k < kLength; k++)
      } // for (int it = 0; it < testCaseLength; it++)

      EAverage = ESum / (double)testCaseLength;
      ESum = 0.0;

      if ((keepAlive != 0) && (iterationsCount % keepAlive == 0))
      {
         printRow(table, iterationsCount, EAverage);
      }

   } // while (iterationsCount <= maxIterations && ...

   if (keepAlive != 0) freeTable(table);

   if (EAverage > maxAcceptableError)
   {
      returnValue = 1;
   }

   *finalError = EAverage;
   *totalIterations = iterationsCount;


   return returnValue;
} // int train(double* finalError, int* totalIterations)



/**
 * Prints information about the success of the training and information
 * pertaining to the reason why training failed, if it failed.
 */
void outputTrainingResult(bool successful, double finalError, 
                         int totalIterations)
{
   printf("Training completed ");

   if (successful)
   {
      printf(ANSI_GREEN_TEXT "successfully" ANSI_CLEAR_TEXT ".\n");
      printf("Reached error %0.4f ≤ %0.4f after " ANSI_BOLD "%d" 
            ANSI_CLEAR " iterations.\n",
            finalError, maxAcceptableError, 
            totalIterations);
   }
   else
   {
      printf(ANSI_RED_TEXT "unsucessfully" ANSI_CLEAR_TEXT ".\n");
      printf("Went through %d iterations and achieved error "
            "%0.4f > %0.4f.\n", 
            maxIterations, finalError, maxAcceptableError);
   }
   return;
} // void outputTrainingResult()


/**
 * Calls the train function, times it, and outputs the data collected by calling
 * the outputTrainingResult function.
 */
int callTrain()
{
   double finalError;
   int totalIterations;

   double initialTime = (double)clock();

   int returnValue = train(&finalError, &totalIterations);
   
   trainDuration = ((double)clock() - initialTime) / NUMBER_OF_MILLISECONDS_IN_A_SECOND;
   
   outputTrainingResult(returnValue == 0, finalError, totalIterations);

   return returnValue;
} // int callTrain()

/**
 * Outputs the time, in seconds, which execution of training and
 * running took place, if applicable
 */
void printTimingInformation()
{
   if (training)
   {
      printf("Training took %0.3f seconds.\n", trainDuration);
   }

   if (training | runningTestCases)
   {
      printf("Running test cases took %0.3f seconds.\n", runDuration);
   }

   return;
} // void printTimingInformation()

/**
 * Outputs a variety of information about the network's current state,
 * including user input values and the value of the a, h, and F arrays 
 * when it was last set, as well as the current weights.
 */
void printNetworkInfo()
{
   int n;

   printf("WEIGHTS\n");
   
   for (int n = 0; n < activationLayers - 1; n++)
   {
      if (n != 0)
      {
         printf("--------\n");
      }

      for (int k = 0; k < lengths[n]; k++)
      {
         for (int j = 0; j < lengths[n + 1]; j++)
         {
            printf("%0.4f\t", weights[n][k][j]);
         }
         printf("\n");
      }
   } // for (int n = 0; n < activationLayers - 1; n++)

   printf("\n");

   for (int n = 0; n < activationLayers; n++)
   {
      printf("\nACTIVATION STATES %d\n", n);

      for (int k = 0; k < lengths[n]; k++)
      {
         printf("%0.4f\t", a[n][k]);
      }

   }

   if (training)
   {
      for (int n = INITIAL_THETA_INDEX; n < THETA_ARRAY_LIMIT; n++)
      {
         printf("\nΘ VALUES %d\n", n);
         for (int k = 0; k < lengths[n]; k++)
         {
            printf("%0.4f\t", Θ[n][k]);
         }
      }
   } // if (training)
   printf("\n");


   return;
} // void printNetworkInfo()

/**
 * Echoes the state of the weights arrays in a file specified by the 
 * user-configurable variable saveFileName. It also adds a 
 * human-readable header and the network configuration.
 */
int saveWeights()
{
   int returnValue = 0;

   FILE* file;
   file = fopen(saveFileName, "wb");
   if (file == NULL)
   {
      printf(ERROR("Could not open file with name \"%s\"."), saveFileName);
      returnValue = -1;
   }
   else
   {
      char* header = malloc(HEADER_STRING_LENGTH);
      int result = snprintf(header, HEADER_STRING_LENGTH, "weight file %s\n", networkConfiguration);

      if (result > HEADER_STRING_LENGTH || result == -1)
      {
         printf(ERROR("Header could not be fully written due to the "
               "buffer's being too small."));
         returnValue = -1;
         free(header);
      }
      else
      {
         bool incompleteFileWriting = false;

         incompleteFileWriting |= 
            fwrite(header, strlen(header), 1, file) < 1;
         free(header);


         char newLine = '\n'; // defines the network config in the file
         
         for (int n = 0; n < activationLayers; n++)
         {
            incompleteFileWriting |=
               fwrite(&lengths[n], sizeof(int), 1, file)  < 1;
            incompleteFileWriting |=
               fwrite(&newLine, sizeof(char), 1, file) < 1;
         }


         for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
         {
            for (int k = 0; k < lengths[n]; k++) // actual data
            {
               incompleteFileWriting |=
                  fwrite(weights[n][k], sizeof(double), lengths[n + 1], 
                        file) < lengths[n + 1];
               incompleteFileWriting |=
                  fwrite(&newLine, sizeof(char), 1, file) < 1;
            }
         }
         int n;

         if (fclose(file) != 0)
         {
            printf(ERROR("could not close save file"));
            returnValue = -1;
         }

         if (incompleteFileWriting)
         {
            printf(ERROR("Could not fully write save file"));
            returnValue = -1;
         }
      } // else [if result > HEADER_STRING_LENGTH || result == -1)   
   } // else [if(file == NULL)]
   
   fclose(file);
   return returnValue;
} // int saveWeights()

/**
 * Frees all memory allocated in the allocateMemory() function
 */
void freeMemory()
{
   for (int it = 0; it < activationLayers; it++)
   {
      free(a[it]);
   }
   free(a);

   for (int it = 0; it < testCaseLength; it++)
   {
      free(testCases[it]);
   }
   free(testCases);

   for (int n = 0; n < NUM_LAYER_INTERVALS; n++)
   {
      for (int k = 0; k < lengths[n]; k++)
      {
         free(weights[n][k]);
      }
      free(weights[n]);
   }
   

   if (training || printTruthTable)
   {
      for (int k = 0; k < lengths[OUTPUT_LAYER_INDEX]; k++)
      {
         free(truthTable[k]);
      }
      free(truthTable);
   }

   if (training)
   {
      for (int n = INITIAL_THETA_INDEX; n < THETA_ARRAY_LIMIT; n++)
      {
         free(Θ[n]);
      }

      free(trueΘ);

      for (int n = INITIAL_PSI_INDEX; n < PSI_ARRAY_LIMIT; n++)
      {
         free(Ψ[n]);
      }

      free(trueΨ);
   } // if (training)
 

   printf(ANSI_PROJECT_COLOR
         "Exit routine completed successfully.\n"
         ANSI_CLEAR_TEXT);

   return;
} // void freeMemory()
