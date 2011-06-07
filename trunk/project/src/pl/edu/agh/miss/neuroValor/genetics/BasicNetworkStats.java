package pl.edu.agh.miss.neuroValor.genetics;

public class BasicNetworkStats {

	private final int okCount;
	private final int wrongCount;

	public BasicNetworkStats(int okCount, int wrongCount, int unknownCount) {
		this.wrongCount = wrongCount;
		this.okCount = okCount;
	}

	public int getOkCount() {
		return okCount;
	}

	public int getWrongCount() {
		return wrongCount;
	}

}
