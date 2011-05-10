package pl.edu.agh.miss.neuroValor.helpers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class NetStructureConfiguration {

	private int inputs;
	private LayerStructureConfiguration outputs;
	private Collection<LayerStructureConfiguration> inners;

	public NetStructureConfiguration() {
	}
	
	public NetStructureConfiguration(int inputs, LayerStructureConfiguration outputs, LayerStructureConfiguration... inners) {
		this.inputs = inputs;
		this.outputs = outputs;
		this.inners = new ArrayList<LayerStructureConfiguration>(Arrays.asList(inners));
	}

	public void setInputs(int inputs) {
		this.inputs = inputs;
	}

	public int getInputs() {
		return inputs;
	}

	public void setOutputs(LayerStructureConfiguration outputs) {
		this.outputs = outputs;
	}

	public LayerStructureConfiguration getOutputs() {
		return outputs;
	}

	public void setInners(Collection<LayerStructureConfiguration> inners) {
		this.inners = inners;
	}

	public Collection<LayerStructureConfiguration> getInners() {
		return inners;
	}

}
