package pl.edu.agh.miss.neuroValor;

import java.util.ArrayList;
import java.util.List;

import pl.edu.agh.miss.neuroValor.helpers.LayerStructureConfiguration;
import pl.edu.agh.miss.neuroValor.helpers.NetStructureConfiguration;
import pl.edu.agh.miss.neuroValor.nodes.InnerNeuron;
import pl.edu.agh.miss.neuroValor.nodes.Neuron;
import pl.edu.agh.miss.neuroValor.nodes.OutputProducer;
import pl.edu.agh.miss.neuroValor.nodes.Synapse;
import pl.edu.agh.miss.neuroValor.nodes.ValueHolder;
import pl.edu.agh.miss.neuroValor.tools.Tools;

public class NeuronNet {

	private List<List<InnerNeuron>> innerLayers;
	private List<ValueHolder> inputLayer;
	private ArrayList<Neuron> outputLayer;

	public NeuronNet(NetStructureConfiguration c) {
		
		inputLayer = new ArrayList<ValueHolder>();
		for (int i=0; i<c.getInputs(); ++i) {
			inputLayer.add(new ValueHolder());
		}
		
		List<? extends OutputProducer> synapsesFrom = inputLayer;
		
		innerLayers = new ArrayList<List<InnerNeuron>>();
		for (LayerStructureConfiguration i: c.getInners()) {
			List<InnerNeuron> il = new ArrayList<InnerNeuron>();
			innerLayers.add(il);
			
			for (int j=0; j<i.getNeuronCount(); ++j) {
				List<Synapse> ss = new ArrayList<Synapse>();
				for (OutputProducer sf: synapsesFrom) {
					ss.add(new Synapse(sf));
				}
				il.add(new InnerNeuron(i.getActivation(), ss, i.getMomentum(), i.getLearningRate()));
			}
			
			synapsesFrom = il;
		}
		
		outputLayer = new ArrayList<Neuron>();
		for (int j=0; j<c.getOutputs().getNeuronCount(); ++j) {
			List<Synapse> ss = new ArrayList<Synapse>();
			for (OutputProducer sf: synapsesFrom) {
				ss.add(new Synapse(sf));
			}
			outputLayer.add(new Neuron(c.getOutputs().getActivation(), ss, c.getOutputs().getMomentum(), c.getOutputs().getLearningRate()));
		}
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
	
	private void setInputValues(List<Double> ds) {
		int i=-1;
		for (ValueHolder vh: inputLayer) {
			vh.setValue(ds.get(++i));
		}
	}

	private void cacheCurrentOutputsIn(List<InnerNeuron> layer) {
		for (InnerNeuron in: layer) {
			in.cacheCurrentOutput();
		}
	}

	public void learnOn(List<Double> args, List<Double> expected) {
		double[] computed = compute(args);
		
		double[][][] changes = new double[innerLayers.size()+1][][];
		double[][] dfis = new double[innerLayers.size()+1][];
		
		changes[innerLayers.size()] = new double[outputLayer.size()][];
		dfis[innerLayers.size()] = new double[outputLayer.size()];
		double[] lastErrors = new double[outputLayer.size()];
		for (int i=0; i<outputLayer.size(); ++i) {
			double e = outputLayer.get(i).getActivation().deriveComputation(computed[i]) * (expected.get(i) - computed[i]);
			lastErrors[i] = e;
			double dfi = outputLayer.get(i).getLearningRate() * e;
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
				double dfi = n.getLearningRate() * e;
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
					s.changeWeight(changes[i][j][k], n.getMomentum());
				}
			}
		}
		
		for (int j=0; j<outputLayer.size(); ++j) {
			Neuron n = outputLayer.get(j);
			n.setThreshold(n.getThreshold()+dfis[innerLayers.size()][j]);
			for (int k=0; k<n.getSynapses().size(); ++k) {
				Synapse s = n.getSynapses().get(k);
				s.changeWeight(changes[innerLayers.size()][j][k], n.getMomentum());
			}
		}
	}
	
	public void learnOn(double[] args, double[] expected) {
		learnOn(Tools.asList(args), Tools.asList(expected));
	}

	public int getInputCount() {
		return inputLayer.size();
	}

	@Override
	public String toString() {
		String inners = " -> ";
		for (List<InnerNeuron> i: innerLayers) {
			inners += i.size()+" -> ";
		}
		return inputLayer.size()+inners+outputLayer.size();
	}

	public double[] compute(double... args) {
		setInputValues(args);
		
		for (List<InnerNeuron> layer: innerLayers) {
			cacheCurrentOutputsIn(layer);
		}
		
		return computeCurrentOutputs();
	}
	
	public double[] compute(List<Double> args) {
		setInputValues(args);
		
		for (List<InnerNeuron> layer: innerLayers) {
			cacheCurrentOutputsIn(layer);
		}
		
		return computeCurrentOutputs();
	}
}
