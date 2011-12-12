/**
 * OpenTLDC is an algorithm for tracking of unknown objects
 * in unconstrained video streams. It is based on TLD,
 * published by Zdenek Kalal
 * (see http://info.ee.surrey.ac.uk/Personal/Z.Kalal/tld.html).
 *
 * Copyright (C) 2011 Sascha Schrader, Stefan Brending, Adrian Block
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "tld.h"
#include "../utils/utility.h"
#include "../img/img.h"
#include "../bbox/bbox.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <string>

/**
 * Is used to show results on window. Current bounding box,
 * positives and negatives, current target.
 *
 * @param i initial sign
 * @param index
 * @param tld learned structures
 * @param fps number of frames per second
 */
void tldDisplay(int i, unsigned long index, TldStruct& tld, double fps) {

	IplImage* inputClone = cvCloneImage(tld.currentImg.input);

	if (i == 0) {

		// draw bounding box
		bb_draw_add_color(inputClone, tld.currentBB);

		tld.handle = cvCreateImage(cvGetSize(inputClone), img_get_colored()->depth, 3);

		cvCvtColor(inputClone, tld.handle, CV_GRAY2BGR);

		cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);

		cvShowImage("Result", tld.handle);
		if (waitKey(10) >= 0)
			std::cout << "key pressed" << std::endl;
	} else {

		// show positive patches
		if (tld.plot->pex == 1)
			inputClone = embedPex(inputClone, tld);

		// show negative patches
		if (tld.plot->nex == 1)
			inputClone = embedNex(inputClone, tld);

		CvFont font;
		fps = 1 / fps;
		std::ostringstream fpsc;
		fpsc << fps;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);


		tld.handle = cvCreateImage(cvGetSize(inputClone), img_get_colored()->depth, 3);
		cvCvtColor(inputClone, tld.handle, CV_GRAY2BGR); // colored output image

		// put fps
		cvPutText(tld.handle, (fpsc.str()).c_str(), cvPoint(
				40, 40), &font, cvScalar(0, 0, 255, 0));

		unsigned char size = 100;

		// show current target (not tested)
		if (tld.plot->target == 1) {

			Eigen::Vector2d vecin;
			vecin << 4, 4;
			Eigen::Vector4d bb = bb_rescalerel(tld.currentBB,
					vecin);

			IplImage* patch = img_patch(tld.handle, bb);
			IplImage* dest = cvCreateImage(cvSize((int) size, (int) size),
					patch->depth, patch->nChannels);

			cvResize(patch, dest);

			for (unsigned int y = 0; y < size; y++)
				for (unsigned int x = 0; x < size; x++) {
					((uchar*) (tld.handle->imageData + tld.handle->widthStep
							* (y)))[x] = ((uchar*) (dest->imageData
							+ dest->widthStep * (y)))[x];
				}

			cvReleaseImage(&patch);
			cvReleaseImage(&dest);
		}

		// Replace
		if (tld.plot->replace == 1) {

			Eigen::Vector4d bb;
			bb(0) = floor(tld.currentBB(0) + 0.5);
			bb(1) = floor(tld.currentBB(1) + 0.5);
			bb(2) = floor(tld.currentBB(2) + 0.5);
			bb(3) = floor(tld.currentBB(3) + 0.5);

			Eigen::Vector2i imsize;
			imsize << tld.currentImg.input->width, tld.currentImg.input->height;

			if (bb_isin(bb, imsize)) {
				unsigned int width = (int) (bb(2) - bb(0)) + 1;
				unsigned int height = (int) (bb(3) - bb(1)) + 1;

				IplImage* patch = cvCreateImage(cvSize(width, height),
						tld.target->depth, tld.target->nChannels);

				cvResize(tld.target, patch);

				for (unsigned int y = bb(1); y <= bb(3); y++)
					for (unsigned int x = bb(0); x <= bb(2); x++) {
						((uchar*) (tld.handle->imageData
								+ tld.handle->widthStep * (y)))[x]
								= ((uchar*) (patch->imageData
										+ patch->widthStep * (y - int(bb(1)))))[x
										- int(bb(0))];
					}

				cvReleaseImage(&patch);
			}
		}


		// Draw Track
		unsigned int linewidth = 2;
		if (tld.currentValid == 1)
			linewidth = 4;
		cv::Scalar color = cv::Scalar(0, 255, 255);

		// grab current bounding box
		Eigen::Vector4d bb = tld.currentBB;
		Eigen::Vector2d vecin;
		vecin << 1.2, 1.2;

		if (!isnan(bb(0))) {
			switch (tld.plot->drawoutput) {
			case 1:
				bb = bb_rescalerel(bb_square(bb), vecin); // scale 1.2
				bb_draw(tld.handle, bb, color, linewidth);
				break;
			case 3:
				bb_draw(tld.handle, bb, color, linewidth);
			}
		}

		cvShowImage("Result", tld.handle);
		if (waitKey(10) >= 0)
			std::cout << "key pressed" << std::endl;

	}

	cvReleaseImage(&inputClone);

}

/**
 * Puts all positive patches on IplImage.
 *
 * @param img output image
 * @param tld learned structures
 * @return output image
 */
IplImage* embedPex(IplImage* img, TldStruct& tld) {


	double rescale = tld.plot->patch_rescale;

	// measure number of possible rows of patches
	int nrow = floor(tld.imgsize.m / (rescale * tld.model->patchsize.x));

	// measure number of possible columns of patches
	int ncol = floor(tld.imgsize.n / (rescale * tld.model->patchsize.y));

	// get prepared Eigen Matrix
	Eigen::MatrixXd pex;
	if (tld.npex > nrow * ncol) { // max nrow * ncol
		pex = mat2img(tld.pex.leftCols(nrow * ncol), nrow * ncol, nrow);
	} else {
		pex = mat2img(tld.pex, tld.npex, nrow);
	}

	int pH = pex.rows();
	int pW = pex.cols();

	// include in output image
	for (int y = 0; y < pH; y++)
		for (int x = img->width - pW; x < img->width; x++) {
			((uchar*) (img->imageData + img->widthStep * (y)))[x] = 255 * pex(
					y, x - (img->width - pW));

		}

	return img;

}

/**
 * Puts all negative patches on IplImage.
 *
 * @param img output image
 * @param tld learned structures
 * @return output image
 */
IplImage* embedNex(IplImage* img, TldStruct& tld) {

	double rescale = tld.plot->patch_rescale;

	// measure number of possible rows of patches
	int nrow = floor(tld.imgsize.m / (rescale * tld.model->patchsize.x));

	// measure number of possible columns of patches
	int ncol = floor(tld.imgsize.n / (rescale * tld.model->patchsize.y));

	// get prepared Eigen Matrix
	Eigen::MatrixXd nex;
	if (tld.nnex > nrow * ncol) { // max nrow * ncol
		nex = mat2img(tld.nex.leftCols(nrow * ncol), nrow * ncol, nrow);
	} else {
		nex = mat2img(tld.nex, tld.nnex, nrow);
	}

	int pH = nex.rows();
	int pW = nex.cols();

	// include in output image
	for (int y = 0; y < pH; y++)
		for (int x = 0; x < pW; x++) {
			((uchar*) (img->imageData + img->widthStep * (y)))[x]
					= 255 * nex(y, x);
		}
	return img;

}
