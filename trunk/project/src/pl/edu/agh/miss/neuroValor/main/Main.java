package pl.edu.agh.miss.neuroValor.main;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import pl.edu.agh.miss.neuroValor.NeuronNet;
import pl.edu.agh.miss.neuroValor.genetics.GeneticAlgorithmSettings;
import pl.edu.agh.miss.neuroValor.genetics.Genetizer;
import pl.edu.agh.miss.neuroValor.predictor.NetStructureConfigurationGenotype;
import pl.edu.agh.miss.neuroValor.predictor.NeuronNetFitnessEstimator;
import pl.edu.agh.miss.neuroValor.tools.Tools;

import com.jkgh.dee.client.RemoteComputationDispatcherConnection;


public class Main {

	private static final int NEURONNET_POPULATION_SIZE = 16;

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		float[] vs = Tools.loadMSTToSubsequentValues(new File("data"+File.separator+"OPTIMUS.mst"));
		double[] vsc = Tools.toDoubleArray(Tools.toFloatArray(Tools.computeChangeLevels(vs)));
		
		final double[] nvsc = Tools.toDoubleArray(Tools.generateInLoop(200, 0.2, 0.8, 0.8));//Tools.normalizeChanges(vsc, 0.2);
		
		//Tools.showPlot(Tools.constructPlot(vs, 200), "vs");
		//Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(vsc), 200), "vsc");
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(nvsc), 200), "nvsc");
		
		System.out.println(nvsc.length-1+" learning points.");
		
		NeuronNetFitnessEstimator estimator = new NeuronNetFitnessEstimator() {

			private static final long serialVersionUID = -2723358366117729352L;

			@Override
			public double estimate(NeuronNet nn) {

				int testCount = 16;
				
				if (nn.getInputCount() >= nvsc.length-testCount) {
					return 0.0;
				}
				
				for (int i=nn.getInputCount(); i<nvsc.length-1-testCount; ++i) {
					nn.learnOn(Tools.asList(nvsc).subList(i-nn.getInputCount(), i), Arrays.asList(nvsc[i]));
				}
				
				double errorSum = 0.0;
				System.out.println(nn);
				for (int i=0; i<testCount; ++i) {
					
					int listFrom = nvsc.length-nn.getInputCount()-testCount+i;
					int listTo = nvsc.length-testCount+i;
					double predicted = nn.compute(Tools.asList(nvsc).subList(listFrom, listTo))[0];
					double real = nvsc[listTo];
					double error = Math.abs(predicted - real);
					
					System.out.println("from "+listFrom+" to "+listTo+" error: "+error+" (predicted "+predicted+" while should be "+real+")");
					
					errorSum += error*error;
				}
				return 1.0/Math.sqrt(errorSum);
			}
			
		};
		
		List<NetStructureConfigurationGenotype> population = new ArrayList<NetStructureConfigurationGenotype>();
		for (int i=0; i<NEURONNET_POPULATION_SIZE; ++i) {
			population.add(new NetStructureConfigurationGenotype(estimator));
		}
		
		RemoteComputationDispatcherConnection dispatcherConnection = new RemoteComputationDispatcherConnection("127.0.0.1", 33333);
		Genetizer<NetStructureConfigurationGenotype> g = new Genetizer<NetStructureConfigurationGenotype>(new GeneticAlgorithmSettings(0.2, 0.05, 0.2, 1.0), population, dispatcherConnection);
		
		for (int i=0; i<20; ++i) {
			System.err.println(i+") BEST FITNESS: "+g.getGenerationFitnesses().get(0));
			g.step();
		}

		NeuronNet bestNN = g.getGenerationFitnesses().get(0).getEvolvable().buildNeuronNet();
		
		double[] prediction = new double[2*nvsc.length];
		for (int i=0; i<nvsc.length; ++i) {
			prediction[i] = nvsc[i];
		}
		for (int i=nvsc.length; i<prediction.length; ++i) {
			prediction[i] = bestNN.compute(Tools.asList(prediction).subList(i-bestNN.getInputCount(), i))[0];
		}
		
		Tools.showPlot(Tools.constructPlot(Tools.toFloatArray(prediction), 200), "prediction");
	}
}
