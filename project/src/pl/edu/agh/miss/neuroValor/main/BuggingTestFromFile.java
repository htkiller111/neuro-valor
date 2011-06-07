package pl.edu.agh.miss.neuroValor.main;

import java.io.File;
import java.io.IOException;
import java.net.UnknownHostException;
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

public class BuggingTestFromFile {

	/**
	 * @param args
	 * @throws ClassNotFoundException
	 * @throws IOException
	 * @throws UnknownHostException
	 */
	public static void main(String[] args) throws UnknownHostException,
			IOException, ClassNotFoundException {
		List<BasicNetworkStructure> basicNetworkStructures = new ArrayList<BasicNetworkStructure>();

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

		// fill networks
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				185, 42, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				215, 33, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 26,
				266, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 71,
				165, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				108, 161, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				241, 12, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				191, 76, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 69,
				171, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 94,
				157, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				208, 3, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 39,
				176, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 39,
				190, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 98,
				192, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 72,
				176, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 90,
				153, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				162, 76, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				234, 12, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 72,
				146, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 24, 40,
				136, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22, 72,
				154, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 22,
				185, 73, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 9, 215,
				33, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 35,
				241, 12, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 15,
				144, 191, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 35,
				241, 1, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 34,
				201, 72, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 27, 2,
				190, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 26, 26,
				153, false));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 15, 45,
				171, true));
		basicNetworkStructures.add(new BasicNetworkStructure(evaluator, 6, 39,
				176, false));
		
		// TODO belowe is just copy from other main FIXME

		RemoteComputationDispatcherConnection rc = new RemoteComputationDispatcherConnection(
				"student.agh.edu.pl", 44444);

		Genetizer<BasicNetworkStructure> genetizer = new Genetizer<BasicNetworkStructure>(
				new GeneticAlgorithmSettings(0.5, 0.05, 0.5, 0.5),
				basicNetworkStructures, rc);

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
