package pl.edu.agh.miss.neuroValor.main;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.strategy.StopTrainingStrategy;
import org.encog.neural.networks.training.strategy.end.EndIterationsStrategy;
import org.encog.util.simple.EncogUtility;

import pl.edu.agh.miss.neuroValor.genetics.BasicNetworkEvaluator;
import pl.edu.agh.miss.neuroValor.genetics.BasicNetworkStats;
import pl.edu.agh.miss.neuroValor.genetics.BasicNetworkStructure;
import pl.edu.agh.miss.neuroValor.genetics.GeneticAlgorithmSettings;
import pl.edu.agh.miss.neuroValor.genetics.Genetizer;
import pl.edu.agh.miss.neuroValor.genetics.Genetizer.EvolvableFitness;
import pl.edu.agh.miss.neuroValor.tools.CandleStick;
import pl.edu.agh.miss.neuroValor.tools.Tools;

import com.jkgh.dee.client.RemoteComputationDispatcherConnection;

public class Main {

	private static final int NEURONNET_POPULATION_SIZE = 20;
	private static final int EVALUATION_TRIES = 3;

	// public static void main2(String[] args) throws Exception {
	//
	// final double[] temporal = disturbedSine();
	// Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(temporal), 200),
	// "Temporal");
	//
	// final int trainingPoints = 200;
	//
	// BasicNetworkEvaluator evaluator = new BasicNetworkEvaluator() {
	//
	// private static final long serialVersionUID = 92836491222L;
	//
	// @Override
	// public double evalute(BasicNetworkStructure bns) {
	// double ret = 0.0;
	// for (int i = 0; i < EVALUATION_TRIES; ++i) {
	// ret += evaluateAndVisualize(bns, false, trainingPoints,
	// temporal);
	// }
	// return ret / EVALUATION_TRIES;
	// }
	//
	// };
	//
	// RemoteComputationDispatcherConnection rc = new
	// RemoteComputationDispatcherConnection(
	// "127.0.0.1", 44444);
	// List<BasicNetworkStructure> population = new
	// ArrayList<BasicNetworkStructure>();
	// for (int i = 0; i < NEURONNET_POPULATION_SIZE; ++i) {
	// population.add(new BasicNetworkStructure(evaluator, randomize(100,
	// 0.8), randomize(100, 0.8), randomize(100, 0.8), false));
	// }
	//
	// Genetizer<BasicNetworkStructure> genetizer = new
	// Genetizer<BasicNetworkStructure>(
	// new GeneticAlgorithmSettings(0.5, 0.05, 0.5, 0.1), population,
	// rc);
	//
	// for (int i = 0; i < 2; ++i) {
	// System.out.println("Generation " + i + ":");
	// for (EvolvableFitness<BasicNetworkStructure> f : genetizer
	// .getGenerationFitnesses()) {
	// System.out.println(f.getEvolvable() + " = " + f.getFitness());
	// }
	// genetizer.step();
	// }
	//
	// EvolvableFitness<BasicNetworkStructure> best = genetizer
	// .getGenerationFitnesses().get(0);
	// System.out
	// .println("-----------------------------------------------------------");
	// System.out.println("Najlepsza siec ma theoretic fitness "
	// + best.getFitness()
	// + " i test fitness "
	// + evaluateAndVisualize(best.getEvolvable(), true,
	// trainingPoints, temporal));
	// }

	public static double[] fibonacci() {
		double[] ret = new double[800];
		double current = 0.0;
		int n1 = 1;
		int n2 = 1;
		int fi = 0;
		for (int i = 0; i < ret.length; ++i) {
			if (fi < n2) {
				++fi;
			} else {
				int nn = n1 + n2;
				n1 = n2;
				n2 = nn;
				fi = 0;
				current = 1.0 - current;
			}
			ret[i] = current;
		}
		return ret;
	}

	public static double[] expanding() {
		double[] ret = new double[800];
		double current = 0.0;
		int n = 10;
		int fi = 0;
		for (int i = 0; i < ret.length; ++i) {
			if (fi < n) {
				++fi;
			} else {
				n = n * 11 / 10;
				fi = 0;
				current = 1.0 - current;
			}
			ret[i] = current;
		}
		return ret;
	}

	public static double[] disturbedSine() {
		double[] ret = new double[800];
		for (int i = 0; i < ret.length; ++i) {
			ret[i] = 0.1 * ((Math.sin(i / 20.0) + 1) / 2.0) + Math.random()
					* 0.9;
		}
		return ret;
	}

	public static double[] mackeyGlass() {
		double[] mgts = new double[1000];

		int tau = 20;
		for (int i = 0; i < mgts.length; ++i) {
			mgts[i] = i - 1 < tau ? 0.6 : 0.9 * mgts[i - 1]
					+ (0.2 * mgts[i - 1 - tau])
					/ (1 + Math.pow(mgts[i - 1 - tau], 10.0));
		}
		double[] stable = new double[800];
		System.arraycopy(mgts, mgts.length - stable.length, stable, 0,
				stable.length);
		final double[] temporal = Tools.normalizeHalf(stable);
		return temporal;
	}

	private static int randomize(int from, double by) {
		return from + (int) ((1 - 2 * Math.random()) * by * from);
	}

	private static double evaluateAndVisualize(BasicNetworkStructure bns,
			boolean visualize, int trainingPoints, double[] temporal) {
		BasicNetwork nn = EncogUtility
				.simpleFeedForward(bns.getInputCount(), bns.getFirstCount(),
						bns.getSecondCount(), 1, bns.isUsingTanh());

		double[][] in = new double[trainingPoints][bns.getInputCount()];
		double[][] out = new double[trainingPoints][1];

		for (int i = 0; i < trainingPoints; ++i) {
			for (int j = 0; j < bns.getInputCount(); ++j) {
				in[i][j] = temporal[i + j];
			}
			out[i][0] = temporal[i + bns.getInputCount()];
		}

		NeuralDataSet dataSet = new BasicNeuralDataSet(in, out);
		Train train = new ResilientPropagation(nn, dataSet);
		train.addStrategy(new StopTrainingStrategy(0.00001, 10));
		train.addStrategy(new EndIterationsStrategy(100));
		EncogUtility.trainToError(train, nn, dataSet, 0.000000001);

		double[] prediction = new double[temporal.length];
		for (int i = 0; i < trainingPoints; ++i) {
			prediction[i] = temporal[i];
		}
		for (int i = trainingPoints; i < prediction.length; ++i) {
			double[] tin = new double[bns.getInputCount()];
			double[] tout = new double[1];
			for (int j = 0; j < bns.getInputCount(); ++j) {
				tin[j] = prediction[i - bns.getInputCount() + j];
			}
			nn.compute(tin, tout);
			prediction[i] = tout[0];
		}

		double sum = 0.0;
		double[] difference = new double[prediction.length];
		for (int i = 0; i < difference.length; ++i) {
			difference[i] = prediction[i] - temporal[i];
			sum += difference[i] * difference[i];
		}

		if (visualize) {
			Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(temporal),
					200), "Temporal");
			Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(prediction),
					200, trainingPoints, trainingPoints + bns.getInputCount()),
					"Prediction of " + bns);
			Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(difference),
					200, trainingPoints, trainingPoints + bns.getInputCount()),
					"Difference");
		}

		return 1 / sum;
	}

	public static void main(String[] args) throws IOException,
			ClassNotFoundException, SecurityException, NoSuchMethodException,
			IllegalArgumentException, IllegalAccessException,
			InvocationTargetException {

		List<CandleStick> allSticks = Tools.loadMSTToSubsequentValues(new File(
				"data/KREZUS.mst"));
		final List<CandleStick> sticks = new ArrayList<CandleStick>(allSticks
				.subList(allSticks.size() - 1000, allSticks.size()));

		BasicNetworkEvaluator evaluator = new BasicNetworkEvaluator() {

			private static final long serialVersionUID = -406486340129467301L;
			private BasicNetworkStats basicNetworkStats;

			@Override
			public double evalute(BasicNetworkStructure bnn) {

				int waiting = 7;
				int tests = 100;
				int ret = 0;

				BasicNetwork nn = EncogUtility.simpleFeedForward(
						CandleStick.FACTORS * bnn.getInputCount(), bnn
								.getFirstCount(), bnn.getSecondCount(), 1,
						false);

				int trainingPoints = sticks.size() - 1 - waiting - tests
						- bnn.getInputCount();
				double[][] in = new double[trainingPoints][];
				double[][] out = new double[in.length][];
				for (int i = 0; i < in.length; ++i) {
					double[] ini = new double[CandleStick.FACTORS
							* bnn.getInputCount()];
					for (int j = 0; j < bnn.getInputCount(); ++j) {
						ini[CandleStick.FACTORS * j] = sticks.get(i + j)
								.getBlackDojiWhite();
						ini[CandleStick.FACTORS * j + 1] = sticks.get(i + j)
								.getBodyShortNeutralLong();
						ini[CandleStick.FACTORS * j + 2] = sticks.get(i + j)
								.getHighShadeShortNeutralLong();
						ini[CandleStick.FACTORS * j + 3] = sticks.get(i + j)
								.getLowShadeShortNeutralLong();
						ini[CandleStick.FACTORS * j + 4] = sticks.get(i + j)
								.getVolumeLowNeutralHigh();
						ini[CandleStick.FACTORS * j + 5] = sticks.get(i + j)
								.getLowerSameHigher();
					}

					double neededLevel = 1.01 * sticks.get(
							i + bnn.getInputCount() - 1).getCloseValue();
					boolean returned = false;
					for (int j = 0; j < waiting; ++j) {
						if (sticks.get(i + bnn.getInputCount() + j)
								.getCloseValue() > neededLevel) {
							returned = true;
							break;
						}
					}

					in[i] = ini;
					out[i] = new double[] { returned ? 1.0 : 0.0 };
				}

				NeuralDataSet dataSet = new BasicNeuralDataSet(in, out);
				Train train = new ResilientPropagation(nn, dataSet);
				train.addStrategy(new StopTrainingStrategy(0.0001, 10));
				train.addStrategy(new EndIterationsStrategy(227));
				EncogUtility.trainToError(train, nn, dataSet, 0.00001);

				int oks = 0;
				int wrongs = 0;

				for (int i = trainingPoints; i < sticks.size()
						- bnn.getInputCount() - waiting - 1; ++i) {
					double[] testIn = new double[CandleStick.FACTORS
							* bnn.getInputCount()];
					for (int j = 0; j < bnn.getInputCount(); ++j) {
						testIn[CandleStick.FACTORS * j] = sticks.get(i + j)
								.getBlackDojiWhite();
						testIn[CandleStick.FACTORS * j + 1] = sticks.get(i + j)
								.getBodyShortNeutralLong();
						testIn[CandleStick.FACTORS * j + 2] = sticks.get(i + j)
								.getHighShadeShortNeutralLong();
						testIn[CandleStick.FACTORS * j + 3] = sticks.get(i + j)
								.getLowShadeShortNeutralLong();
						testIn[CandleStick.FACTORS * j + 4] = sticks.get(i + j)
								.getVolumeLowNeutralHigh();
						testIn[CandleStick.FACTORS * j + 5] = sticks.get(i + j)
								.getLowerSameHigher();
					}
					double neededLevel = 1.01 * sticks.get(
							i + bnn.getInputCount() - 1).getCloseValue();
					boolean returned = false;
					for (int j = 0; j < waiting; ++j) {
						if (sticks.get(i + bnn.getInputCount() + waiting)
								.getCloseValue() > neededLevel) {
							returned = true;
							break;
						}
					}
					double[] testOut = new double[] { returned ? 1.0 : 0.0 };
					double[] predictedOut = new double[1];

					nn.compute(testIn, predictedOut);

					double threshold = 0.25;
					if (Math.abs(predictedOut[0] - 0.5) > threshold) {
						boolean ok = (predictedOut[0] > 0.5 ? 1.0 : 0.0) == testOut[0];
						if (ok) {
							++oks;
						} else {
							++wrongs;
						}
					}
				}
				ret += oks - wrongs;
				basicNetworkStats = new BasicNetworkStats(oks, wrongs, tests);
				return ret;

			}

			@Override
			public BasicNetworkStats getBasicNetworkStats() {
				return basicNetworkStats;
			}

		};

		RemoteComputationDispatcherConnection rc = new RemoteComputationDispatcherConnection(
				"student.agh.edu.pl", 44444);
		List<BasicNetworkStructure> population = new ArrayList<BasicNetworkStructure>();
		for (int i = 0; i < NEURONNET_POPULATION_SIZE; ++i) {
			population.add(new BasicNetworkStructure(evaluator, randomize(20,
					0.8), CandleStick.FACTORS * randomize(20, 0.8),
					CandleStick.FACTORS * randomize(20, 0.8), false));
		}

		Genetizer<BasicNetworkStructure> genetizer = new Genetizer<BasicNetworkStructure>(
				new GeneticAlgorithmSettings(0.5, 0.05, 0.5, 0.5), population,
				rc);

		for (int i = 0; i < 1000; ++i) {
			System.out.println("Generation " + i + ":");
			for (EvolvableFitness<BasicNetworkStructure> f : genetizer
					.getGenerationFitnesses()) {
				System.out.println(f.getEvolvable() + " = " + f.getFitness());
			}
			genetizer.step();
		}
	}
}
