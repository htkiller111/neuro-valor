package pl.edu.agh.miss.neuroValor.predictor;

import java.util.ArrayList;
import java.util.List;

import pl.edu.agh.miss.neuroValor.NeuronNet;
import pl.edu.agh.miss.neuroValor.functions.DifferentiableFunction;
import pl.edu.agh.miss.neuroValor.genetics.Evolvable;
import pl.edu.agh.miss.neuroValor.genetics.Gene;
import pl.edu.agh.miss.neuroValor.genetics.SingleNaturalNumberGene;
import pl.edu.agh.miss.neuroValor.genetics.SinglePositiveNumberGene;
import pl.edu.agh.miss.neuroValor.helpers.LayerStructureConfiguration;
import pl.edu.agh.miss.neuroValor.helpers.NetStructureConfiguration;

public class NetStructureConfigurationGenotype implements Evolvable<NetStructureConfigurationGenotype>{

	private static final long serialVersionUID = -5871105439321460936L;

	private final List<Gene> genes;
		
	private final NeuronNetFitnessEstimator estimator;
	
	public NetStructureConfigurationGenotype(NeuronNetFitnessEstimator estimator) {
		this(estimator, 1+(int) (Math.random()*100), 1+(int) (Math.random()*200), 0.2+Math.random()*0.4, 0.1+Math.random()*0.2, 1+(int) (Math.random()*200), 0.2+Math.random()*0.4, 0.1+Math.random()*0.2, 0.2+Math.random()*0.4, 0.1+Math.random()*0.2);
	}
	
	public NetStructureConfigurationGenotype(NeuronNetFitnessEstimator estimator, int inputLayerSize, int firstHiddenLayerSize, double firstHiddenLayerMomentum, double firstHiddenLayerLearningRate, int secondHiddenLayerSize, double secondHiddenLayerMomentum, double secondHiddenLayerLearningRate, double outputLayerMomentum, double outputLayerLearningRate) {
		this.estimator = estimator;
		this.genes = new ArrayList<Gene>();
		
		genes.add(new SingleNaturalNumberGene(inputLayerSize, 100));
		
		genes.add(new SingleNaturalNumberGene(firstHiddenLayerSize, 100));
		genes.add(new SinglePositiveNumberGene(firstHiddenLayerMomentum));
		genes.add(new SinglePositiveNumberGene(firstHiddenLayerLearningRate));
		
		genes.add(new SingleNaturalNumberGene(secondHiddenLayerSize, 100));
		genes.add(new SinglePositiveNumberGene(secondHiddenLayerMomentum));
		genes.add(new SinglePositiveNumberGene(secondHiddenLayerLearningRate));
		
		genes.add(new SinglePositiveNumberGene(outputLayerMomentum));
		genes.add(new SinglePositiveNumberGene(outputLayerLearningRate));
	}
	
	@Override
	public double computeFitness() {
		return estimator.estimate(buildNeuronNet());
	}

	public NeuronNet buildNeuronNet() {
		
		SingleNaturalNumberGene inputLayerSizeGene = (SingleNaturalNumberGene) genes.get(0);
		
		SingleNaturalNumberGene firstHiddenLayerSizeGene = (SingleNaturalNumberGene) genes.get(1);
		SinglePositiveNumberGene firstHiddenLayerMomentumGene = (SinglePositiveNumberGene) genes.get(2);
		SinglePositiveNumberGene firstHiddenLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(3);

		SingleNaturalNumberGene secondHiddenLayerSizeGene = (SingleNaturalNumberGene) genes.get(4);
		SinglePositiveNumberGene secondHiddenLayerMomentumGene = (SinglePositiveNumberGene) genes.get(5);
		SinglePositiveNumberGene secondHiddenLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(6);

		SinglePositiveNumberGene outputLayerMomentumGene = (SinglePositiveNumberGene) genes.get(7);
		SinglePositiveNumberGene outputLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(8);
		
		return new NeuronNet(
			new NetStructureConfiguration(
				inputLayerSizeGene.getValue(),
				new LayerStructureConfiguration(
					outputLayerMomentumGene.getValue(), outputLayerLearningRateGene.getValue(), 1, DifferentiableFunction.SIGMOID_ONE
				),
				new LayerStructureConfiguration(
					firstHiddenLayerMomentumGene.getValue(), firstHiddenLayerLearningRateGene.getValue(), firstHiddenLayerSizeGene.getValue(), DifferentiableFunction.SIGMOID_ONE
				),
				new LayerStructureConfiguration(
					secondHiddenLayerMomentumGene.getValue(), secondHiddenLayerLearningRateGene.getValue(), secondHiddenLayerSizeGene.getValue(), DifferentiableFunction.SIGMOID_ONE
				)
			)
		);
	}

	@Override
	public NetStructureConfigurationGenotype copied() {
		SingleNaturalNumberGene inputLayerSizeGene = (SingleNaturalNumberGene) genes.get(0);
		
		SingleNaturalNumberGene firstHiddenLayerSizeGene = (SingleNaturalNumberGene) genes.get(1);
		SinglePositiveNumberGene firstHiddenLayerMomentumGene = (SinglePositiveNumberGene) genes.get(2);
		SinglePositiveNumberGene firstHiddenLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(3);

		SingleNaturalNumberGene secondHiddenLayerSizeGene = (SingleNaturalNumberGene) genes.get(4);
		SinglePositiveNumberGene secondHiddenLayerMomentumGene = (SinglePositiveNumberGene) genes.get(5);
		SinglePositiveNumberGene secondHiddenLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(6);

		SinglePositiveNumberGene outputLayerMomentumGene = (SinglePositiveNumberGene) genes.get(7);
		SinglePositiveNumberGene outputLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(8);
		return new NetStructureConfigurationGenotype(estimator, inputLayerSizeGene.getValue(), firstHiddenLayerSizeGene.getValue(), firstHiddenLayerMomentumGene.getValue(), firstHiddenLayerLearningRateGene.getValue(), secondHiddenLayerSizeGene.getValue(), secondHiddenLayerMomentumGene.getValue(), secondHiddenLayerLearningRateGene.getValue(), outputLayerMomentumGene.getValue(), outputLayerLearningRateGene.getValue());
	}

	@Override
	public List<Gene> getGenes() {
		return genes;
	}

	@Override
	public String toString() {
		SingleNaturalNumberGene inputLayerSizeGene = (SingleNaturalNumberGene) genes.get(0);
		
		SingleNaturalNumberGene firstHiddenLayerSizeGene = (SingleNaturalNumberGene) genes.get(1);
		SinglePositiveNumberGene firstHiddenLayerMomentumGene = (SinglePositiveNumberGene) genes.get(2);
		SinglePositiveNumberGene firstHiddenLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(3);

		SingleNaturalNumberGene secondHiddenLayerSizeGene = (SingleNaturalNumberGene) genes.get(4);
		SinglePositiveNumberGene secondHiddenLayerMomentumGene = (SinglePositiveNumberGene) genes.get(5);
		SinglePositiveNumberGene secondHiddenLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(6);

		SinglePositiveNumberGene outputLayerMomentumGene = (SinglePositiveNumberGene) genes.get(7);
		SinglePositiveNumberGene outputLayerLearningRateGene = (SinglePositiveNumberGene) genes.get(8);
		
		return inputLayerSizeGene.getValue()+" inputs, "+
		"First: s="+firstHiddenLayerSizeGene.getValue()+", m="+firstHiddenLayerMomentumGene.getValue()+", lr="+firstHiddenLayerLearningRateGene.getValue()+
		"Second: s="+secondHiddenLayerSizeGene.getValue()+", m="+secondHiddenLayerMomentumGene.getValue()+", lr="+secondHiddenLayerLearningRateGene.getValue()+
		"Output: m="+outputLayerMomentumGene.getValue()+", lr="+outputLayerLearningRateGene.getValue();
	}
}
