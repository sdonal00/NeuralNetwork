import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class PS5 {

	public static ArrayList<double[]> Xtemp;
	public static double X[][];
	public static double h[][];
	public static double w1[][];
	public static double w1D[][];
	public static double w2[][];
	public static double w2D[][];
	public static double grad1[][];
	public static double grad2[][];
	public static double yhat[][];
	public static double loss[];
	public static double accuracy[];
	public static int y[][];
	public static final int EPOCH = 700;
	public static int epochCount = 0;

	public static void main(String[] args) throws IOException {

		File xinput = new File("xdata.txt");
		BufferedReader br = new BufferedReader(new FileReader(xinput));
		int imgCount = 0;

		Xtemp = new ArrayList<>();

		while (br.ready()) {

			String str = br.readLine();
			String[] strs = str.split(",");
			double[] c = new double[785];

			for (int i = 1; i < strs.length + 1; i++) {
				c[i] = Double.parseDouble(strs[i - 1]);
			}
			Xtemp.add(c);

			imgCount++;
		}

		br.close();

		X = new double[imgCount][784 + 1];
		h = new double[imgCount][30];
		yhat = new double[imgCount][10];
		loss = new double[EPOCH];
		accuracy = new double[EPOCH];
		y = new int[imgCount][10];
		w1 = new double[30][785];
		w2 = new double[10][31];
		grad1 = new double[h[0].length][X[0].length];
		grad2 = new double[y[0].length][h[0].length];

		File yinput = new File("ydata.txt");
		br = new BufferedReader(new FileReader(yinput));
		int count = 0;

		while (br.ready()) {

			int[] k = new int[10];
			int num = Integer.parseInt(br.readLine());

			if (num != 0) {
				k[num - 1] = 1;
			} else {
				k[num] = 1;
			}

			y[count] = k;
			count++;
		}

		for (int i = 1; i < Xtemp.size(); i++) {

			X[i] = Xtemp.get(i);
			X[i][0] = 1;
		}

		br.close();
		br = new BufferedReader(new FileReader("w1.txt"));
		int wCount = 0;
		while (br.ready()) {
			String str = br.readLine();
			String[] strs = str.split(",");
			for (int i = 0; i < strs.length; i++) {
				w1[wCount][i] = Double.parseDouble(strs[i]);
			}
			wCount++;
		}

		br.close();
		br = new BufferedReader(new FileReader("w2.txt"));
		wCount = 0;
		while (br.ready()) {
			String str = br.readLine();
			String[] strs = str.split(",");
			for (int i = 0; i < strs.length; i++) {
				w2[wCount][i] = Double.parseDouble(strs[i]);
			}
			wCount++;
		}
		br.close();

		train();

		BufferedWriter bw = new BufferedWriter(new FileWriter("w1out.txt"));

		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[0].length; j++) {
				bw.write(String.format("%.4f", w1[i][j]) + ",");
			}
			bw.write("\n");
		}
		bw.close();
		bw = new BufferedWriter(new FileWriter("w2out.txt"));

		for (int i = 0; i < w2.length; i++) {
			for (int j = 0; j < w2[0].length; j++) {
				bw.write(String.format("%.4f", w2[i][j]) + ",");
			}
			bw.write("\n");
		}
		bw.close();
		bw = new BufferedWriter(new FileWriter("loss.txt"));
		for (int i = 0; i < loss.length; i++) {
			bw.write(i + " " + Double.toString(loss[i]) + "\n");

		}
		bw.close();

		bw = new BufferedWriter(new FileWriter("accuracy.txt"));
		for (int i = 0; i < accuracy.length; i++) {
			bw.write(i + " " + String.format("%.5s", Double.toString(accuracy[i] * 100)) + "\n");
		}
		bw.close();

		bw = new BufferedWriter(new FileWriter("predicted.txt"));

		for (int i = 0; i < yhat.length; i++) {
			int p = getMax(yhat[i]);
			bw.write(p + " \n");
		}
		bw.close();

		System.out.printf("\nRECORDS: %.5s\n", imgCount);
		System.out.printf("FEATURES: %.5s\n", X[0].length - 1);
		System.out.printf("EPOCH COUNT: %.5s\n\n", epochCount);

		double[][] X2 = new double[10][785];

		for (int i = 0; i < 10; i++) {
			X2[i] = X[i];
		}

		predict(X2);
		for (int i = 0; i < 10; i++) {

			int yhatv = getMax(yhat[i]);
			int yv = getMax(y[i]);
			System.out.printf("PREDICTED %-4s ACTUAL: %-4s", yhatv, yv);

			if (yhatv == yv) {
				System.out.println(" CORRECT");
			} else {
				System.out.println(" INCORRECT");
			}
		}

	}

	public static void loss() {

		for (int i = 0; i < y.length; i++) {
			for (int j = 0; j < y[0].length; j++) {
				loss[epochCount] += ((-y[i][j]) * Math.log(yhat[i][j]))
						- ((1.0 - y[i][j]) * Math.log(1.0 - yhat[i][j]));

			}
		}
		loss[epochCount] /= y.length;
		loss[epochCount] += meanSumOfWeights();

	}

	public static void train() {

		for (int i = 0; i < EPOCH; i++) {
			System.out.println("EPOCH COUNT: " + (i + 1));
			h = sigmoid(m(X, transpose(w1)));
			h = addFirstColumn(h);
			yhat = sigmoid(m(h, transpose(w2)));
			accuracy[epochCount] = accuracy();
			loss();
			System.out.printf("ACCURACY: %.2f%%\n", (accuracy() * 100));
			backProp();
			epochCount++;
		}
	}

	public static double[][] sigmoid(double[][] k) {

		double[][] o = new double[k.length][k[0].length];

		for (int i = 0; i < k.length; i++) {
			for (int j = 0; j < k[0].length; j++) {
				o[i][j] = (double) (1.0 / (1.0 + Math.exp(-k[i][j])));

			}
		}

		return o;
	}

	public static double[][] derivS(double[][] k) {

		double[][] o = new double[k.length][k[0].length];
		k = sigmoid(k);
		for (int i = 0; i < k.length; i++) {
			for (int j = 0; j < k[0].length; j++) {
				o[i][j] = (k[i][j]) * (1.0 - (k[i][j]));
			}
		}

		return o;
	}

	public static double[][] deriv(double[][] k) {

		double[][] o = new double[k.length][k[0].length];

		for (int i = 0; i < k.length; i++) {
			for (int j = 0; j < k[0].length; j++) {
				o[i][j] = (k[i][j]) * (1.0 - (k[i][j]));
			}
		}

		return o;
	}

	public static double meanSumOfWeights() {

		double o = 0;

		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[0].length; j++) {
				o += w1[i][j] * w1[i][j];
			}
		}
		for (int i = 0; i < w2.length; i++) {
			for (int j = 0; j < w2[0].length; j++) {
				o += w2[i][j] * w2[i][j];
			}
		}

		o *= 3.0 / (2.0 * y.length);

		return o;
	}

	public static void backProp() {

		double delta1[][];
		double delta2[][] = new double[y.length][y[0].length];

		for (int j = 0; j < y.length; j++) {
			for (int k = 0; k < y[0].length; k++) {
				delta2[j][k] = y[j][k] - yhat[j][k];
			}
		}

		double[][] w1Temp = new double[w1.length][w1[0].length];
		double[][] w2Temp = new double[w2.length][w2[0].length];

		for (int i = 0; i < w1.length; i++) {
			for (int j = 1; j < w1[0].length; j++) {
				w1Temp[i][j] = w1[i][j];
			}
			w1Temp[i][0] = 0;
		}

		for (int i = 0; i < w2.length; i++) {
			for (int j = 1; j < w2[0].length; j++) {
				w2Temp[i][j] = w2[i][j];
			}
			w2Temp[i][0] = 0;
		}

		delta1 = ham(m(delta2, w2Temp), derivS(m(X, transpose(w1Temp))));

		grad2 = m(transpose(delta2), h);
		grad1 = m(transpose(delta1), X);

		for (int i = 0; i < grad1.length; i++) {
			for (int j = 0; j < grad1[0].length; j++) {
				grad1[i][j] /= X.length;
			}
		}

		for (int i = 0; i < grad2.length; i++) {
			for (int j = 0; j < grad2[0].length; j++) {
				grad2[i][j] /= X.length;
			}
		}

		for (int i = 0; i < w1.length; i++) {
			for (int j = 0; j < w1[0].length; j++) {
				w1[i][j] = w1[i][j] + (grad1[i][j] / 4);
			}
		}
		for (int i = 0; i < w2.length; i++) {
			for (int j = 0; j < w2[0].length; j++) {
				w2[i][j] = w2[i][j] + (grad2[i][j] / 4);
			}

		}
	}

	public static double[][] transpose(double m[][]) {
		double[][] temp = new double[m[0].length][m.length];

		for (int i = 0; i < m[0].length; i++)
			for (int j = 0; j < m.length; j++)
				temp[i][j] = m[j][i];

		return temp;
	}

	public static double[][] ham(double[][] p, double[][] o) {

		double[][] z = new double[p.length][p[0].length];

		for (int i = 0; i < o.length; i++) {
			for (int j = 0; j < o[0].length; j++) {
				z[i][j] += p[i][j] * o[i][j];
			}
		}

		return z;
	}

	public static double[][] m(double[][] A, double[][] B) {

		int rows1 = A.length;
		int cols1 = A[0].length;
		int rows2 = B.length;
		int cols2 = B[0].length;

		if (cols1 != rows2) {
			System.err.println("Error with multiplication!  Check the dimensions.");
			throw new IllegalArgumentException();
		}

		double[][] C = new double[rows1][cols2];
		for (int i = 0; i < rows1; i++) {
			for (int j = 0; j < cols2; j++) {
				C[i][j] = 0.00000;
			}
		}

		for (int i = 0; i < rows1; i++) {
			for (int j = 0; j < cols2; j++) {
				for (int k = 0; k < cols1; k++) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}

		return C;
	}

	public static int getMax(int[] z) {

		int loc = -2;

		for (int i = 0; i < z.length; i++) {

			if (loc == -2 || z[loc] < z[i]) {
				loc = i;
			}
		}

		loc++;
		return loc;
	}

	public static int getMax(double[] z) {

		int loc = -2;

		for (int i = 0; i < z.length; i++) {

			if (loc == -2 || z[loc] < z[i]) {
				loc = i;
			}
		}

		loc++;
		return loc;
	}

	public static double accuracy() {
		double accCount = 0;
		for (int i = 0; i < yhat.length; i++) {
			if (getMax(yhat[i]) == getMax(y[i])) {
				accCount++;
			}
		}
		return accCount / yhat.length;
	}

	public static void predict(double[][] o) {

		h = sigmoid(m(o, transpose(w1)));
		h = addFirstColumn(h);
		yhat = sigmoid(m(h, transpose(w2)));

	}

	public static double[][] addFirstColumn(double m[][]) {
		double[][] temp = new double[m.length][m[0].length + 1];

		for (int i = 0; i < temp.length; i++) {
			for (int k = 0; k < m[i].length; k++) {
				temp[i][k + 1] = m[i][k];
			}
			temp[i][0] = 1;
		}
		return temp;
	}

	public static double[][] dropFirstColumn(double m[][]) {
		double[][] temp = new double[m.length][m[0].length - 1];

		for (int i = 0; i < temp.length; i++) {
			for (int k = 0; k < temp[i].length; k++) {
				temp[i][k] = m[i][k + 1];
			}
		}
		return temp;
	}

}
