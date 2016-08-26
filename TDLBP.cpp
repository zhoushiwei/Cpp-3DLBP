#include "TDLBP.h"

int dec2bin(int number){
	// Converts decimal number to binary

	int bin = 0;
	int i = 1;


	while (number > 0){
		bin += (number % 2) * i;
		number = number / 2;
		i *= 10;
	}

	return bin;
}

void histogramCount(Mat &LBP, vector<int>& histogram, int x1, int x2, int y1, int y2){
	// Gets the histogram of the LBP values of a region of the image.

	int bins = 14;
	double division = (double) 256 / bins; // 0 - 255 possible intensity values;
	int temp; // debugging purposes

	for (int i = x1; i <= x2; i = i + 1){
		for (int j = y1; j <= y2; j++){

			
			temp = (float) floor((LBP.at<float>(i, j)) / division);
			histogram[temp] = histogram[temp] + 1;
		}
	}

}

int bin2dec(int number){
	// Converts binary to decimal

	int dec = 0;
	int rem;
	int i = 1;
	while (number > 0){
		rem = number % 10;
		dec = dec + rem * i;
		i = i * 2;
		number = number / 10;
	}

	return dec;
}




// Calculates the 3DLBP in the depth image. Remember that the image should be between 0 and 255 pixel intensitys.
Mat calculate3DLBP(Mat depth_image){

	// Initialize Matrices that will store the 4 LBP code numbers.
	Mat LBP1(depth_image.rows, depth_image.cols, CV_32F);
	Mat LBP2(depth_image.rows, depth_image.cols, CV_32F);
	Mat LBP3(depth_image.rows, depth_image.cols, CV_32F);
	Mat LBP4(depth_image.rows, depth_image.cols, CV_32F);

	// LBP Indexes to be visited
	vector<int> index_x, index_y;
	index_x.reserve(8);
	index_y.reserve(8);

	int vx [] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int vy [] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	index_x.insert(index_x.begin(), &vx[0], &vx[8]);
	index_y.insert(index_y.begin(), &vy[0], &vy[8]);

	// 8 Neighbors values
	vector<double> neigh;
	neigh.reserve(8);

	// Values of Neighbor - central pixel
	vector<double> subtraction;
	subtraction.reserve(8);

	//  LBP values
	int i1 = 0;
	int i2 = 0;
	int i3 = 0;
	int i4 = 0;

	// Variables needed
	double subtract;
	int temporary_bin = 0;

	// For all pixels in image performs 3DLBP
	for (int i = 0; i < depth_image.rows; i++){
		for (int j = 0; j < depth_image.cols; j++){

			// 8 Neighbors visited
			for (int k = 0; k < 8; k++){
				// Tests if goes out of image borders
				if (i + index_x[k] < 0 || i + index_x[k] >= depth_image.rows || j + index_y[k] < 0 || j + index_y[k] >= depth_image.cols){
					neigh.push_back(0);
				}
				else{
					neigh.push_back((double) depth_image.at<uchar>(i + index_x[k], j + index_y[k]));
				}
				
				subtract = (neigh[k] - (double) depth_image.at<uchar>(i,j));

				if (subtract > 7){
					subtraction.push_back(7);
				}

				else if (subtract < -7){
					subtraction.push_back(-7);
				}

				else{
					subtraction.push_back(subtract);
				}

				int i11 = (subtraction[k] >= 0);

				i1 = i1 + (i11 * pow(10, static_cast<double>(7 - k)));

				temporary_bin = dec2bin(int(abs(subtraction[k])));

				int i22 = temporary_bin / 100 % 10;
				int i33 = temporary_bin / 10 % 10;
				int i44 = temporary_bin / 1 % 10;

				i2 = i2 + (i22 * pow(10, static_cast<double>(7 - k)));
				i3 = i3 + (i33 * pow(10, static_cast<double>(7 - k)));
				i4 = i4 + (i44 * pow(10, static_cast<double>(7 - k)));

			}

			i1 = bin2dec(i1);
			i2 = bin2dec(i2);
			i3 = bin2dec(i3);
			i4 = bin2dec(i4);

			subtraction.clear();
			subtraction.reserve(8);
			neigh.clear();
			neigh.reserve(8);

			LBP1.at<float>(i, j) = (float) i1;
			LBP2.at<float>(i, j) = (float) i2;
			LBP3.at<float>(i, j) = (float) i3;
			LBP4.at<float>(i, j) = (float) i4;

			// Reset Parameters;
			i1 = 0;
			i2 = 0;
			i3 = 0;
			i4 = 0;
		}
	}

	// Now we compile all info in a histogram

	// Image will be divided in 8 x 8 total blocks (64 blocks)
	int numWindows_x = 8;
	int numWindows_y = 8;

	// Adapt the size of the blocks to the depth image size.
	int step_x = floor(static_cast<double>(depth_image.cols / numWindows_x));
	int step_y = floor(static_cast<double>(depth_image.rows / numWindows_y));

	// Variable needed for histogram count
	int count = 0;


	// Histograms will have 14 bins.
	vector<int> histogram(14, 0);
	vector<int> histogram2(14, 0);
	vector<int> histogram3(14, 0);
	vector<int> histogram4(14, 0);

	// Correspond to the end and start of the blocks
	int x1, x2, y1, y2;

	// 64 blocks each one having 14 bins. They will all be concatenated in the Final Descriptor
	Mat FinalDescriptor = Mat::zeros(Size(14 * 64 * 4, 1),CV_32FC1); 

	for (int n = 0; n < numWindows_y; n++){
		for (int m = 0; m < numWindows_x; m++){

			// Calculating start and end of the evaluated regions
			x1 = n * step_y;
			x2 = (n + 1)*step_y - 1;

			y1 = m * step_x;
			y2 = (m + 1)*step_x - 1;

			// Histogram count and Fill in the Final Frescriptor for the 4 LBP codes
			histogramCount(LBP1, histogram, x1, x2, y1, y2);
			for (int o = 0; o < 14; o++){
				FinalDescriptor.at<float>(count) = histogram[o];
				histogram[o] = 0;
				count++;
			}


			histogramCount(LBP2, histogram2, x1, x2, y1, y2);
			for (int o = 14; o < 28; o++){
				FinalDescriptor.at<float>(count) = histogram2[o - 14];
				histogram2[o - 14] = 0;
				count++;
			}


			histogramCount(LBP3, histogram3, x1, x2, y1, y2);
			for (int o = 28; o < 42; o++){
				FinalDescriptor.at<float>(count) = histogram3[o - 28];
				histogram3[o - 28] = 0;
				count++;
			}


			histogramCount(LBP4, histogram4, x1, x2, y1, y2);
			for (int o = 42; o < 56; o++){
				FinalDescriptor.at<float>(count) = histogram4[o - 42];
				histogram4[o - 42] = 0;
				count++;
			}

		}
	}

	return FinalDescriptor;

}
