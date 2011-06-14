package pl.edu.agh.miss.neuroValor.helper;

import java.util.List;

public class DataManager<T> {

	private final List<T> data;
	private final int trainingDataLength;
	private final int testDataLength;
	private int startIndex;
	private int shift;

	public DataManager(List<T> data, int trainingDataLength,
			int testDataLength, int startIndex, int shift) {
		this.data = data;
		this.trainingDataLength = trainingDataLength;
		this.testDataLength = testDataLength;
		this.startIndex = startIndex;
		this.shift = shift;
	}

	public List<T> getTrainingData() throws IntervalRangeException {
		try {
			List<T> trainingData = data.subList(startIndex, startIndex
					+ trainingDataLength);
			return trainingData;
		} catch (IndexOutOfBoundsException e) {
			throw new IntervalRangeException();
		}
	}

	public List<T> getTestData() throws IntervalRangeException {
		try {
			List<T> testData = data.subList(startIndex + trainingDataLength,
					startIndex + trainingDataLength + testDataLength);
			return testData;
		} catch (IndexOutOfBoundsException e) {
			throw new IntervalRangeException();
		}
	}

	public void doShift() {
		startIndex = startIndex + shift;
	}

}
