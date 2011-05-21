package pl.edu.agh.miss.neuroValor.main;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;

import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.Train;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.strategy.StopTrainingStrategy;
import org.encog.util.simple.EncogUtility;

import pl.edu.agh.miss.neuroValor.tools.CandleStick;
import pl.edu.agh.miss.neuroValor.tools.Tools;


public class Main {

	private static final int NEURONNET_POPULATION_SIZE = 20;

	public static void main(String[] args) throws IOException, ClassNotFoundException, SecurityException, NoSuchMethodException, IllegalArgumentException, IllegalAccessException, InvocationTargetException {
			
		List<CandleStick> sticks = Tools.loadMSTToSubsequentValues(new File("data/OPTIMUS.mst"));
		sticks = sticks.subList(sticks.size()-365, sticks.size());

		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getCloseValue")), 100), "[RAW] CloseValue");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getLowerSameHigher")), 200), "LowerSameHigher");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getBlackDojiWhite")), 200), "BlackDojiWhite");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getBodyShortNeutralLong")), 200), "BodyShortNeutralLong");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getLowShadeShortNeutralLong")), 200), "LowShadeShortNeutralLong");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getHighShadeShortNeutralLong")), 200), "HighShadeShortNeutralLong");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(sticks, CandleStick.class.getMethod("getVolumeLowNeutralHigh")), 200), "VolumeLowNeutralHigh");
		
		int history = 10;
		BasicNetwork nn = EncogUtility.simpleFeedForward(CandleStick.FACTORS*history, CandleStick.FACTORS*history*3/2, CandleStick.FACTORS*history*2/3, 1, false);
		
		int waiting = 7;

		int tests = 10;
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
				if (sticks.get(i+history+waiting).getCloseValue() > neededLevel) {
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
			if (gain[i] > 0.1) {
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
		EncogUtility.trainToError(train, nn, dataSet, 0.01);
		
		double mega = 0.001;
		
		int oks = 0;
		int wrongs = 0;
		int megaOks = 0;
		int megaWrongs = 0;
		
		for (int i=trainingPoints; i<sticks.size()-history-waiting-1; ++i) {
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
