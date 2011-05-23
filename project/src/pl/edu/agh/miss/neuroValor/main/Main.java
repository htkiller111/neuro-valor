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
import pl.edu.agh.miss.neuroValor.genetics.BasicNetworkStructure;
import pl.edu.agh.miss.neuroValor.genetics.GeneticAlgorithmSettings;
import pl.edu.agh.miss.neuroValor.genetics.Genetizer;
import pl.edu.agh.miss.neuroValor.genetics.Genetizer.EvolvableFitness;
import pl.edu.agh.miss.neuroValor.tools.CandleStick;
import pl.edu.agh.miss.neuroValor.tools.Tools;

import com.jkgh.dee.client.RemoteComputationDispatcherConnection;


public class Main {

	private static final int NEURONNET_POPULATION_SIZE = 20;
	
	public static void main2(String[] args) throws Exception {
		
		int next = 1;
		int nextHop = 1;
		final double[] temporal = new double[600];
		for (int i=0; i<temporal.length; ++i) {
			if (next == 0) {
				next = 5+(int) (0.5+4*Math.sin(nextHop/7.0));
				++nextHop;
				temporal[i] = nextHop % 2;
			} else {
				--next;
				temporal[i] = 1-nextHop % 2;
			}
		}
		
		final int trainingPoints = 300;
		
		BasicNetworkEvaluator evaluator = new BasicNetworkEvaluator() {

			private static final long serialVersionUID = 92836491222L;

			@Override
			public double evalute(BasicNetworkStructure bns) {
				return evaluateAndVisualize(bns, false, trainingPoints, temporal);
			}
			
		};
		
		RemoteComputationDispatcherConnection rc = new RemoteComputationDispatcherConnection("127.0.0.1", 44444);
		List<BasicNetworkStructure> population = new ArrayList<BasicNetworkStructure>();
		for (int i=0; i<NEURONNET_POPULATION_SIZE; ++i) {
			population.add(new BasicNetworkStructure(evaluator, randomize(100, 0.2), randomize(75, 0.2), randomize(50, 0.2), false));
		}
		
		Genetizer<BasicNetworkStructure> genetizer = new Genetizer<BasicNetworkStructure>(new GeneticAlgorithmSettings(0.2, 0.05, 0.3, 1.0), population, rc);
		
		while (true) {
			genetizer.step();
			EvolvableFitness<BasicNetworkStructure> best = genetizer.getGenerationFitnesses().get(0);
			System.out.println("-----------------------------------------------------------");
			System.out.println("Najlepsza siec ma theoretic fitness "+best.getFitness()+" i test fitness "+evaluateAndVisualize(best.getEvolvable(), true, trainingPoints, temporal));
			System.in.read(new byte[4]);
		}
	}
	
	private static int randomize(int from, double by) {
		return from + (int) ((1-2*Math.random())*by*from);
	}

	private static double evaluateAndVisualize(BasicNetworkStructure bns, boolean visualize, int trainingPoints, double[] temporal) {
		BasicNetwork nn = EncogUtility.simpleFeedForward(bns.getInputCount(), bns.getFirstCount(), bns.getSecondCount(), 1, bns.isUsingTanh());
		
		double[][] in = new double[trainingPoints][bns.getInputCount()];
		double[][] out = new double[trainingPoints][1];
		
		for (int i=0; i<trainingPoints; ++i) {
			for (int j=0; j<bns.getInputCount(); ++j) {
				in[i][j] = temporal[i+j];
			}
			out[i][0] = temporal[i+bns.getInputCount()];
		}
		
		NeuralDataSet dataSet = new BasicNeuralDataSet(in, out);
		Train train = new ResilientPropagation(nn, dataSet);
		train.addStrategy(new StopTrainingStrategy(0.001, 10));
		train.addStrategy(new EndIterationsStrategy(100));
		EncogUtility.trainToError(train, nn, dataSet, 0.000000001);
		
		double[] prediction = new double[temporal.length];
		for (int i=0; i<trainingPoints; ++i) {
			prediction[i] = temporal[i];
		}
		for (int i=trainingPoints; i<prediction.length; ++i) {
			double[] tin = new double[bns.getInputCount()];
			double[] tout = new double[1];
			for (int j=0; j<bns.getInputCount(); ++j) {
				tin[j] = prediction[i-bns.getInputCount()+j];
			}
			nn.compute(tin, tout);
			prediction[i] = tout[0];
		}
		

		double sum = 0.0;
		double[] difference = new double[prediction.length];
		for (int i=0; i<difference.length; ++i) {
			difference[i] = prediction[i]-temporal[i];
			sum += difference[i]*difference[i];
		}
		
		if (visualize) {
			Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(temporal), 200), "Temporal");
			Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(prediction), 200, trainingPoints, trainingPoints+bns.getInputCount()), "Prediction of "+bns);
			Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(difference), 200, trainingPoints, trainingPoints+bns.getInputCount()), "Difference");
		}
		
		return 1/sum;
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, SecurityException, NoSuchMethodException, IllegalArgumentException, IllegalAccessException, InvocationTargetException {
			
		List<CandleStick> sticks = Tools.loadMSTToSubsequentValues(new File("data/KREZUS.mst"));
		sticks = sticks.subList(sticks.size()-500, sticks.size());
		
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getCloseValue")), 100), "[RAW] CloseValue");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getLowerSameHigher")), 200), "LowerSameHigher");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getBlackDojiWhite")), 200), "BlackDojiWhite");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getBodyShortNeutralLong")), 200), "BodyShortNeutralLong");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getLowShadeShortNeutralLong")), 200), "LowShadeShortNeutralLong");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getHighShadeShortNeutralLong")), 200), "HighShadeShortNeutralLong");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getVolumeLowNeutralHigh")), 200), "VolumeLowNeutralHigh");
		
		int history = 7;
		BasicNetwork nn = EncogUtility.simpleFeedForward(CandleStick.FACTORS*history, 100, 50, 1, false);
		
		int waiting = 7;

		int tests = 100;
		int trainingPoints = sticks.size()-2-waiting-tests-history;
		double[][] in = new double[trainingPoints][];
		double[][] out = new double[in.length][];
		for (int i=0; i<in.length; ++i) {
			double[] ini = new double[CandleStick.FACTORS*history];
			for (int j=0; j<history; ++j) {
				ini[CandleStick.FACTORS*j] = sticks.get(i+j).getBlackDojiWhite();
				ini[CandleStick.FACTORS*j+1] = sticks.get(i+j).getBodyShortNeutralLong();
				ini[CandleStick.FACTORS*j+2] = sticks.get(i+j).getHighShadeShortNeutralLong();
				ini[CandleStick.FACTORS*j+3] = sticks.get(i+j).getLowShadeShortNeutralLong();
				ini[CandleStick.FACTORS*j+4] = sticks.get(i+j).getVolumeLowNeutralHigh();
				ini[CandleStick.FACTORS*j+5] = sticks.get(i+j).getLowerSameHigher();
			}
			
			double neededLevel = 1.01*sticks.get(i+history-1).getCloseValue();
			boolean returned = false;
			for (int j=0; j<waiting; ++j) {
				if (sticks.get(i+history+j).getCloseValue() > neededLevel) {
					returned = true;
					break;
				}
			}
			
			in[i] = ini;
			out[i] = new double[] {returned ? 1.0 : 0.0};
		}
		
		int yes = 0;
		int no = 0;
		float[] gain = new float[trainingPoints];
		for (int i=0; i<trainingPoints; ++i) {
			gain[i] = (float) out[i][0];
			if (gain[i] > 0.5) {
				++yes;
			} else {
				++no;
			}
		}
		Tools.showPlot(Tools.constructPlot(gain, 300), "[COMPUTED] gain 1% after waiting?");
		System.out.println(yes+" yes, "+no+" no");
		
		NeuralDataSet dataSet = new BasicNeuralDataSet(in, out);
		Train train = new ResilientPropagation(nn, dataSet);
		train.addStrategy(new StopTrainingStrategy(0.001, 10));
		EncogUtility.trainToError(train, nn, dataSet, 0.0001);
		
		double mega = 0.0001;
		
		int oks = 0;
		int wrongs = 0;
		int megaOks = 0;
		int megaWrongs = 0;
		
		for /*(int i=0; i<trainingPoints; ++i) { */(int i=trainingPoints; i<sticks.size()-history-waiting-1; ++i) {
			double[] testIn = new double[CandleStick.FACTORS*history];
			for (int j=0; j<history; ++j) {
				testIn[CandleStick.FACTORS*j] = sticks.get(i+j).getBlackDojiWhite();
				testIn[CandleStick.FACTORS*j+1] = sticks.get(i+j).getBodyShortNeutralLong();
				testIn[CandleStick.FACTORS*j+2] = sticks.get(i+j).getHighShadeShortNeutralLong();
				testIn[CandleStick.FACTORS*j+3] = sticks.get(i+j).getLowShadeShortNeutralLong();
				testIn[CandleStick.FACTORS*j+4] = sticks.get(i+j).getVolumeLowNeutralHigh();
				testIn[CandleStick.FACTORS*j+5] = sticks.get(i+j).getLowerSameHigher();
			}
			double neededLevel = 1.01*sticks.get(i+history-1).getCloseValue();
			boolean returned = false;
			for (int j=0; j<waiting; ++j) {
				if (sticks.get(i+history+waiting).getCloseValue() > neededLevel) {
					returned = true;
					break;
				}
			}
			double[] testOut = new double[] {returned ? 1.0 : 0.0};
			double[] predictedOut = new double[1];
			
			nn.compute(testIn, predictedOut);
		
			boolean ok = (predictedOut[0]>0.5 ? 1.0 : 0.0) == testOut[0];
			if (predictedOut[0]>1-mega) {
				System.out.println("I strongly recommended to buy, and you would "+(testOut[0] == 1.0 ? "win!" : "lose! :((((("));
			}
			if (ok) {
				if (predictedOut[0] > 1-mega || predictedOut[0] < mega) {
					++megaOks;
				}
				++oks;
			} else {
				if (predictedOut[0] > 1-mega || predictedOut[0] < mega) {
					++megaWrongs;
				}
				++wrongs;
			}
			System.out.println(predictedOut[0]+" <-> "+testOut[0]+" = "+(ok ? "OK!!!" : "WRONG :(((((((((((((("));
		}
		
		System.out.println(oks+"("+megaOks+") <> "+wrongs+"("+megaWrongs+")");
	}
}
