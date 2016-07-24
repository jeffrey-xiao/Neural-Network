using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MNIST_NN {
    public partial class Form1 : Form {
        private static string TRAIN_PATH;
        private static string TEST_PATH;
        private static string WEIGHTS_PATH;

        private const int TRAIN_SIZE = 60000;
        private const int TEST_SIZE = 10000;

        private const int INPUT_SIZE = 784;
        private const int HIDDEN_SIZE = 75;
        private const int OUTPUT_SIZE = 10;

        private Random rand = new Random();
        private double[,] w1 = new double[INPUT_SIZE, HIDDEN_SIZE];
        private double[,] w2 = new double[HIDDEN_SIZE, OUTPUT_SIZE];

        private double[] b1 = new double[HIDDEN_SIZE];
        private double[] b2 = new double[OUTPUT_SIZE];

        private double[][] train;
        private double[][] test;

        private int[] trainAnswer;
        private int[] testAnswer;

        private double correct = 0;
        private double total = 0;
        private double iterations = 0;
        private double cost = 0;
        private Queue<double> costList = new Queue<double>();
        private Queue<bool> correctList = new Queue<bool>();

        private bool training = false;

        public Form1 () {
            InitializeComponent();
        }

        private void Initialize_Weights_Click (object sender, EventArgs e) {
            total = 0;
            correct = 0;
            cost = 0;
            costList.Clear();
            correctList.Clear();
            iterations = 0;
            double range1 = 1 / Math.Sqrt(INPUT_SIZE);
            double range2 = 1 / Math.Sqrt(HIDDEN_SIZE);

            for (int i = 0; i < INPUT_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    w1[i, j] = GetRandomGaussian(range1);

            for (int i = 0; i < HIDDEN_SIZE; i++)
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    w2[i, j] = GetRandomGaussian(range2);

            for (int i = 0; i < HIDDEN_SIZE; i++)
                b1[i] = 0;

            for (int i = 0; i < OUTPUT_SIZE; i++)
                b2[i] = 0;

            this.Text = "Finished initializing weights!";
        }

        private void LoadData_Click (object sender, EventArgs e) {
            TRAIN_PATH = TrainingData.Text;
            TEST_PATH = TestingData.Text;

            train = new double[TRAIN_SIZE][];
            trainAnswer = new int[TRAIN_SIZE];

            test = new double[TEST_SIZE][];
            testAnswer = new int[TEST_SIZE];

            using (StreamReader reader = new StreamReader(TRAIN_PATH)) {
                for (int i = 0; i < TRAIN_SIZE; i++) {
                    string inputData = reader.ReadLine();
                    string[] data = inputData.Split(',');
                    trainAnswer[i] = int.Parse(data[0]);
                    train[i] = new double[INPUT_SIZE];
                    for (int j = 1; j <= INPUT_SIZE; j++)
                        train[i][j - 1] = double.Parse(data[j]) / 255.0;
                }
            }

            using (StreamReader reader = new StreamReader(TEST_PATH)) {
                for (int i = 0; i < TEST_SIZE; i++) {
                    string inputData = reader.ReadLine();
                    string[] data = inputData.Split(',');
                    testAnswer[i] = int.Parse(data[0]);
                    test[i] = new double[INPUT_SIZE];
                    for (int j = 1; j <= INPUT_SIZE; j++)
                        test[i][j - 1] = double.Parse(data[j]) / 255.0;
                }
            }
            this.Text = "Finished loading data!";
        }

        private void Stop_Click (object sender, EventArgs e)
        {
            training = false;
        }

        private void Train_Click (object sender, EventArgs e) {
            total = 0;
            correct = 0;
            cost = 0;
            costList.Clear();
            correctList.Clear();
            training = true;

            while (training) {
                ShuffleTraining();
                for (int j = 0; j < TRAIN_SIZE && training; j++) {
                    Application.DoEvents();
                    double alpha = 1000.0 / (iterations + 10000);
                    if (Predict(RunNetwork(train[j]), trainAnswer[j])) {
                        correct++;
                        correctList.Enqueue(true);
                    } else {
                        correctList.Enqueue(false);
                    }

                    double currCost = BackPropagate(train[j], trainAnswer[j], alpha);

                    cost += currCost;
                    costList.Enqueue(currCost);

                    total++;
                    iterations++;

                    if (total > 60000) {
                        total--;
                        if (correctList.Dequeue())
                            correct--;
                        cost -= costList.Dequeue();
                    }

                    this.Text = correct + "/" + total + " " + String.Format("{0:0.0000} {1:0.0000} {2}", correct / total * 100.0, cost / total, iterations);
                }
            }
        }

        private void TestButton_Click (object sender, EventArgs e) {
            total = 0;
            correct = 0;
            correctList.Clear();
            for (int i = 0; i < TEST_SIZE; i++) {
                if (Predict(RunNetwork(test[i]), testAnswer[i]))
                    correct++;
                total++;
                this.Text = correct + "/" + total + " " + String.Format("{0:0.0000}", correct / total * 100.0);
            }
        }

        private void ShuffleTraining () {
            for (int i = 0; i < TRAIN_SIZE; i++) {
                int swapIndex = rand.Next(TRAIN_SIZE - i) + i;
                double[] tempArray = train[i];
                train[i] = train[swapIndex];
                train[swapIndex] = tempArray;

                int temp = trainAnswer[i];
                trainAnswer[i] = trainAnswer[swapIndex];
                trainAnswer[swapIndex] = temp;
            }
        }

        private bool Predict (double[] output, int ans) {
            int max = 0;
            for (int i = 1; i < OUTPUT_SIZE; i++)
                if (output[i] > output[max])
                    max = i;

            return max == ans;
        }

        private double[] RunNetwork (double[] input) {
            Debug.Assert(input.GetLength(0) == INPUT_SIZE);

            double[] a1 = new double[HIDDEN_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                a1[i] = b1[i];

            for (int i = 0; i < INPUT_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    a1[j] += input[i] * w1[i, j];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                a1[i] = Math.Tanh(a1[i]);

            double[] a2 = new double[OUTPUT_SIZE];

            for (int i = 0; i < OUTPUT_SIZE; i++)
                a2[i] = b2[i];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    a2[j] += a1[i] * w2[i, j];

            for (int i = 0; i < OUTPUT_SIZE; i++)
                a2[i] = Math.Tanh(a2[i]);

            return a2;
        }

        // returns true if correct and false if incorrect
        private double BackPropagate (double[] input, int answer, double learningRate) {
            Debug.Assert(input.GetLength(0) == INPUT_SIZE);

            double[] a1 = new double[HIDDEN_SIZE];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                a1[i] = b1[i];

            for (int i = 0; i < INPUT_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    a1[j] += input[i] * w1[i, j];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                a1[i] = Math.Tanh(a1[i]);

            double[] a2 = new double[OUTPUT_SIZE];

            for (int i = 0; i < OUTPUT_SIZE; i++)
                a2[i] = b2[i];

            for (int i = 0; i < HIDDEN_SIZE; i++)
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    a2[j] += a1[i] * w2[i, j];

            for (int i = 0; i < OUTPUT_SIZE; i++)
                a2[i] = Math.Tanh(a2[i]);

            double[] e2 = new double[OUTPUT_SIZE];
            double[] e1 = new double[HIDDEN_SIZE];

            double cost = 0;

            for (int i = 0; i < OUTPUT_SIZE; i++) {
                e2[i] = (a2[i] - (i == answer ? 1 : -1)) * (1 - a2[i] * a2[i]);
                cost += (a2[i] - (i == answer ? 1 : -1)) * (a2[i] - (i == answer ? 1 : -1));
            }

            for (int i = 0; i < OUTPUT_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    e1[j] += w2[j, i] * e2[i] * (1 - a1[j] * a1[j]);

            for (int i = 0; i < INPUT_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    w1[i, j] -= input[i] * e1[j] * learningRate;

            for (int i = 0; i < HIDDEN_SIZE; i++)
                b1[i] -= e1[i] * learningRate;

            for (int i = 0; i < HIDDEN_SIZE; i++)
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    w2[i, j] -= a1[i] * e2[j] * learningRate;

            for (int i = 0; i < OUTPUT_SIZE; i++)
                b2[i] -= e2[i] * learningRate;

            return cost;
        }

        private double GetRandomGaussian (double stdDev) {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return stdDev * randStdNormal;
        }

        private void SaveWeights_Click (object sender, EventArgs e) {
            WEIGHTS_PATH = WeightsPath.Text;
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(WEIGHTS_PATH)) {
                file.WriteLine(iterations);
                for (int i = 0; i < INPUT_SIZE; i++)
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        file.WriteLine(w1[i, j]);

                for (int i = 0; i < HIDDEN_SIZE; i++)
                    for (int j = 0; j < OUTPUT_SIZE; j++)
                        file.WriteLine(w2[i, j]);
                
                for (int i = 0; i < HIDDEN_SIZE; i++)
                    file.WriteLine(b1[i]);
                
                for (int i = 0; i < OUTPUT_SIZE; i++)
                    file.WriteLine(b2[i]);
            }
            this.Text = "Finished saving weights!";
        }

        private void LoadWeights_Click (object sender, EventArgs e) {
            WEIGHTS_PATH = WeightsPath.Text;
            total = 0;
            correct = 0;
            correctList.Clear();
            using (StreamReader reader = new StreamReader(WEIGHTS_PATH)) {
                iterations = double.Parse(reader.ReadLine());
                for (int i = 0; i < INPUT_SIZE; i++)
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        w1[i, j] = double.Parse(reader.ReadLine());

                for (int i = 0; i < HIDDEN_SIZE; i++)
                    for (int j = 0; j < OUTPUT_SIZE; j++)
                        w2[i, j] = double.Parse(reader.ReadLine());


                for (int i = 0; i < HIDDEN_SIZE; i++)
                    b1[i] = double.Parse(reader.ReadLine());

                for (int i = 0; i < OUTPUT_SIZE; i++)
                    b2[i] = double.Parse(reader.ReadLine());
            }
            this.Text = "Finished loading weights!";
        }
    }
}
