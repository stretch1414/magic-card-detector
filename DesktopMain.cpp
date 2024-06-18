#include <map>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <direct.h>
#include <fstream>

#include <opencv2/core/core.hpp>           // cv::Mat
#include <opencv2/highgui/highgui.hpp>     // cv::imread()
#include <opencv2/imgproc/imgproc.hpp>     // 
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;

class CardDatabase
{
	public:
		string name;
		string manaCost;
		string cmc;
		string colors;
		string type;
		string supertypes;
		string types;
		string subtypes;
		string setName;
		string rarity;
		string text;
		string flavorText;
		string artist;
		string setNumber;
		string power;
		string toughness;
		string loyalty;
};


void main (int argc, char ** argv)
{
	// Build data set descriptors
	// Read in from the camera
	// Store each frame to a Mat variable
	// Analyze each frame for the descriptor points (SURF)
	// Match on a certain threshold
	// Compare against the compiled data set
	// Output the matched results (and hope they are correct)

	// Build data set descriptors
	int numCards = 5;
	const string imageSource = "Images\\000";
	string dataSource = "Images\\000";

	//Storing image data in array for indexing later
	Mat imgArr[10], test;
	vector<KeyPoint> keypointsArr[10];
	Mat descriptorArr[10];
	SurfFeatureDetector detector(500);
	SurfDescriptorExtractor extractor;

	CardDatabase dataArr[10];
	ifstream dataFile;
	string line;

	for (int i = 0; i <= numCards; i++)
	{
		stringstream ss1, ss2;
		ss1 << imageSource << i+1 << ".jpg";
		imgArr[i] = imread(ss1.str(), CV_LOAD_IMAGE_COLOR);
		// Detecting keypoints
		detector.detect(imgArr[i], keypointsArr[i]);
		// Computing descriptors
		extractor.compute(imgArr[i], keypointsArr[i], descriptorArr[i]);

		// Open the data file corresponding to the image
		ss2 << dataSource << i+1 << ".txt";
		dataFile.open(ss2.str());

		// Retrieve and store the data into the array of classes
		getline(dataFile, dataArr[i].name);
		getline(dataFile, dataArr[i].manaCost);
		getline(dataFile, dataArr[i].cmc);
		getline(dataFile, dataArr[i].colors);
		getline(dataFile, dataArr[i].type);
		getline(dataFile, dataArr[i].supertypes);
		getline(dataFile, dataArr[i].types);
		getline(dataFile, dataArr[i].subtypes);
		getline(dataFile, dataArr[i].setName);
		getline(dataFile, dataArr[i].rarity);
		getline(dataFile, dataArr[i].text);
		getline(dataFile, dataArr[i].flavorText);
		getline(dataFile, dataArr[i].artist);
		getline(dataFile, dataArr[i].setNumber);
		getline(dataFile, dataArr[i].power);
		getline(dataFile, dataArr[i].toughness);
		getline(dataFile, dataArr[i].loyalty);

		dataFile.close();
	}


	VideoCapture inputVideo;
	inputVideo.open(0);
	if (!inputVideo.isOpened())
	{
		cout << "Could not open the input video" << endl;
		return;
	}
	Mat src;
	vector<KeyPoint> keypoints2;
	Mat descriptors2;
	FlannBasedMatcher matcher;
	vector<DMatch> matches;

	namedWindow("testing");
	int flag = true;
	// Show the image captured in the window and repeat
	while (flag)
	{
		inputVideo >> src;
		// Check if at end
		if (src.empty()) break;
		waitKey(30);
		imshow("testing", src);
		// Detecting keypoints
		detector.detect(src, keypoints2);
		// Computing descriptors
		extractor.compute(src, keypoints2, descriptors2);
		// Check against descriptor set
		for (int i = 0; i <= numCards; i++)
		{
			if (descriptors2.empty()) break;
			// Matching descriptors
			matcher.match(descriptorArr[i], descriptors2, matches);
			//-- Quick calculation of max and min distances between keypoints
			double max_dist = 0, min_dist = 100;
			for (int j = 0; j < descriptorArr[i].rows; j++)
			{
				double dist = matches[j].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			std::vector< DMatch > good_matches;
			for (int j = 0; j < descriptorArr[i].rows; j++)
			{
				if (matches[j].distance <= max(2 * min_dist, .02))
				{
					good_matches.push_back(matches[j]);
				}
			}
			// drawing the results
			namedWindow("matches2", 1);
			Mat img1_matches, img2_matches;
			drawMatches(imgArr[i], keypointsArr[i], src, keypoints2,
				good_matches, img2_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			std::vector<Point2f> obj;
			std::vector<Point2f> scene;
			for (size_t k = 0; k < good_matches.size(); k++)
			{
				//-- Get the keypoints from the good matches
				obj.push_back(keypointsArr[i][good_matches[k].queryIdx].pt);
				scene.push_back(keypoints2[good_matches[k].trainIdx].pt);
			}
			// Check for any matches at all (homography requires at least 4)
			if (obj.size() > 3)
			{
				Mat H = findHomography(obj, scene, CV_RANSAC);
				//-- Get the corners from the image_1 ( the object to be "detected" )
				std::vector<Point2f> obj_corners(4);
				obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(imgArr[i].cols, 0);
				obj_corners[2] = cvPoint(imgArr[i].cols, imgArr[i].rows); obj_corners[3] = cvPoint(0, imgArr[i].rows);
				std::vector<Point2f> scene_corners(4);
				perspectiveTransform(obj_corners, scene_corners, H);
				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				cv::line(img2_matches, scene_corners[0] + Point2f(imgArr[i].cols, 0), scene_corners[1] + Point2f(imgArr[i].cols, 0), Scalar(0, 255, 0), 4);
				cv::line(img2_matches, scene_corners[1] + Point2f(imgArr[i].cols, 0), scene_corners[2] + Point2f(imgArr[i].cols, 0), Scalar(0, 255, 0), 4);
				cv::line(img2_matches, scene_corners[2] + Point2f(imgArr[i].cols, 0), scene_corners[3] + Point2f(imgArr[i].cols, 0), Scalar(0, 255, 0), 4);
				cv::line(img2_matches, scene_corners[3] + Point2f(imgArr[i].cols, 0), scene_corners[0] + Point2f(imgArr[i].cols, 0), Scalar(0, 255, 0), 4);
				// Check a threshold on the matches to see if it is close enough
				if (good_matches.size() <= 40)
				{
					cout << dataArr[i].name << endl;
					cout << dataArr[i].manaCost << endl;
					cout << dataArr[i].cmc << endl;
					cout << dataArr[i].colors << endl;
					cout << dataArr[i].type << endl;
					cout << dataArr[i].supertypes << endl;
					cout << dataArr[i].types << endl;
					cout << dataArr[i].subtypes << endl;
					cout << dataArr[i].setName << endl;
					cout << dataArr[i].rarity << endl;
					cout << dataArr[i].text << endl;
					cout << dataArr[i].flavorText << endl;
					cout << dataArr[i].artist << endl;
					cout << dataArr[i].setNumber << endl;
					cout << dataArr[i].power << endl;
					cout << dataArr[i].toughness << endl;
					cout << dataArr[i].loyalty << endl;
				}
			}
			imshow("matches2", img2_matches);
			waitKey(1000);	// This is just for demonstration purposes
		}
	}
	cout << "Finished" << endl;
	waitKey(0);
	destroyAllWindows();
}
