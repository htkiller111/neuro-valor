package pl.edu.agh.miss.neuroValor;

import java.util.ArrayList;
import java.util.List;

import pl.edu.agh.miss.neuroValor.functions.DifferentiableFunction;
import pl.edu.agh.miss.neuroValor.nodes.InnerNeuron;
import pl.edu.agh.miss.neuroValor.nodes.Neuron;
import pl.edu.agh.miss.neuroValor.nodes.OutputProducer;
import pl.edu.agh.miss.neuroValor.nodes.Synapse;
import pl.edu.agh.miss.neuroValor.nodes.ValueHolder;

public class NeuronNet {

	private List<List<InnerNeuron>> innerLayers;
	private List<ValueHolder> inputLayer;
	private ArrayList<Neuron> outputLayer;
	private double learningRate;

	public NeuronNet(int inputs, int outputs, int... inners) {
		this(0.2, DifferentiableFunction.SIGMOID_ONE, DifferentiableFunction.SIGMOID_ONE, inputs, outputs, inners);
	}
	
	public NeuronNet(double learningRate, DifferentiableFunction innerActivation, DifferentiableFunction outputActivation, int inputs, int outputs, int... inners) {
		
		this.setLearningRate(learningRate);
		
		inputLayer = new ArrayList<ValueHolder>();
		for (int i=0; i<inputs; ++i) {
			inputLayer.add(new ValueHolder());
		}
		
		List<? extends OutputProducer> synapsesFrom = inputLayer;
		
		innerLayers = new ArrayList<List<InnerNeuron>>();
		for (int i=0; i<inners.length; ++i) {
			List<InnerNeuron> il = new ArrayList<InnerNeuron>();
			innerLayers.add(il);
			
			for (int j=0; j<inners[i]; ++j) {
				List<Synapse> ss = new ArrayList<Synapse>();
				for (OutputProducer sf: synapsesFrom) {
					ss.add(new Synapse(sf));
				}
				il.add(new InnerNeuron(innerActivation, ss));
			}
			
			synapsesFrom = il;
		}
		
		outputLayer = new ArrayList<Neuron>();
		for (int i=0; i<outputs; ++i) {
			List<Synapse> ss = new ArrayList<Synapse>();
			for (OutputProducer sf: synapsesFrom) {
				ss.add(new Synapse(sf));
			}
			outputLayer.add(new Neuron(outputActivation, ss));
		}
	}

	public double[] compute(double... args) {
		
		setInputValues(args);
		
		for (List<InnerNeuron> layer: innerLayers) {
			cacheCurrentOutputsIn(layer);
		}
		
		return computeCurrentOutputs();
	}

	private double[] computeCurrentOutputs() {
		double[] ret = new double[outputLayer.size()];
		int i = -1;
		for (OutputProducer op: outputLayer) {
			ret[++i] = op.getOutput();
		}
		return ret;
	}

	private void setInputValues(double[] ds) {
		int i=-1;
		for (ValueHolder vh: inputLayer) {
			vh.setValue(ds[++i]);
		}
	}

	private void cacheCurrentOutputsIn(List<InnerNeuron> layer) {
		for (InnerNeuron in: layer) {
			in.cacheCurrentOutput();
		}
	}

	public void learnOn(double[] expected, double... args) {
		
		double[] computed = compute(args);
		
		double[][][] changes = new double[innerLayers.size()+1][][];
		double[][] dfis = new double[innerLayers.size()+1][];
		
		changes[innerLayers.size()] = new double[outputLayer.size()][];
		dfis[innerLayers.size()] = new double[outputLayer.size()];
		double[] lastErrors = new double[outputLayer.size()];
		for (int i=0; i<outputLayer.size(); ++i) {
			double e = outputLayer.get(i).getActivation().deriveComputation(computed[i]) * (expected[i] - computed[i]);
			lastErrors[i] = e;
			double dfi = learningRate * e;
			dfis[innerLayers.size()][i] = dfi;
			changes[innerLayers.size()][i] = new double[outputLayer.get(i).getSynapses().size()];
			for (int j=0; j<changes[innerLayers.size()][i].length; ++j) {
				changes[innerLayers.size()][i][j] = dfi*outputLayer.get(i).getSynapses().get(j).getFrom().getOutput();
			}
		}
		List<? extends Neuron> lastLayer = outputLayer;
		 
		for (int i=innerLayers.size()-1; i>=0; --i) {
			List<InnerNeuron> innerLayer = innerLayers.get(i);
			double[] errors = new double[innerLayer.size()];
			changes[i] = new double[innerLayer.size()][];
			dfis[i] = new double[innerLayer.size()];
			for (int j=0; j<innerLayer.size(); ++j) {
				InnerNeuron n = innerLayer.get(j);
				double g = 0.0;
				for (int k=0; k<lastLayer.size(); ++k) {
					g += lastErrors[k]*lastLayer.get(k).getSynapses().get(j).getWeight();
				}
				double e = n.getActivation().deriveComputation(n.getOutput()) * g;
				errors[j] = e;
				double dfi = learningRate * e;
				dfis[i][j] = dfi;
				changes[i][j] = new double[n.getSynapses().size()];
				for (int k=0; k<changes[i][j].length; ++k) {
					changes[i][j][k] = dfi*n.getSynapses().get(k).getFrom().getOutput();
				}
			}
			lastErrors = errors;
			lastLayer = innerLayer;
		}
		
		for (int i=0; i<innerLayers.size(); ++i) {
			List<InnerNeuron> innerLayer = innerLayers.get(i);
			for (int j=0; j<innerLayer.size(); ++j) {
				Neuron n = innerLayer.get(j);
				n.setThreshold(n.getThreshold()+dfis[i][j]);
				for (int k=0; k<n.getSynapses().size(); ++k) {
					Synapse s = n.getSynapses().get(k);
					s.setWeight(s.getWeight()+changes[i][j][k]);
				}
			}
		}
		
		for (int j=0; j<outputLayer.size(); ++j) {
			Neuron n = outputLayer.get(j);
			n.setThreshold(n.getThreshold()+dfis[innerLayers.size()][j]);
			for (int k=0; k<n.getSynapses().size(); ++k) {
				Synapse s = n.getSynapses().get(k);
				s.setWeight(s.getWeight()+changes[innerLayers.size()][j][k]);
			}
		}
		
//		double[] previousDeltas = new double[outputLayer.size()];
//		List<? extends Neuron> previousLayer = outputLayer;
//		for (int i=0; i<outputLayer.size(); ++i) {
//			previousDeltas[i] = computed[i] - expected[i];
//		}
//		
//		double[][] deltasCache = new double[innerLayers.size()+1][];
//		deltasCache[innerLayers.size()] = previousDeltas;
//		
//		for (int b=innerLayers.size()-1; b>=0; --b) {
//			List<InnerNeuron> layer = innerLayers.get(b);
//			double[] deltas = new double[layer.size()];
//			for (int i=0; i<layer.size(); ++i) {
//				double delta = 0.0;
//				for (int j=0; j<previousDeltas.length; ++j) {
//					delta += previousLayer.get(j).getSynapses().get(i).getWeight() * previousDeltas[j];
//				}
//				deltas[i] = delta;
//			}
//			
//			deltasCache[b] = deltas;
//			
//			previousDeltas = deltas;
//			previousLayer = layer;
//		}
//		
//		for (int i=0; i<innerLayers.size(); ++i) {
//			double[] deltas = deltasCache[i];
//			
//			List<InnerNeuron> innerLayer = innerLayers.get(i);
//			for (int j=0; j<innerLayer.size(); ++j) {
//				InnerNeuron in = innerLayer.get(j);
//				for (Synapse s: in.getSynapses()) {
//					s.setWeight(s.getWeight()+learningRate*deltas[j]*in.getActivation().deriveComputation(in.getOutput())*s.getFrom().getOutput());
//				}
//			}
//		}
//		
//		double[] deltas = deltasCache[innerLayers.size()];
//		for (int i=0; i<outputLayer.size(); ++i) {
//			Neuron n = outputLayer.get(i);
//			for (Synapse s: n.getSynapses()) {
//				s.setWeight(s.getWeight()+learningRate*deltas[i]*n.getActivation().deriveComputation(computed[i])*s.getFrom().getOutput());
//			}
//		}
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public double getLearningRate() {
		return learningRate;
	}

}
