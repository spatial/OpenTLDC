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

#include "bbox.h"
#include <cmath>
#include "../utils/median.h"

/*Predicts where the tracked bounding box will be in the next frame*/
Eigen::Vector4d bb_predict(Eigen::VectorXd const & bb0,
		Eigen::MatrixXd const & pt0, Eigen::MatrixXd const & pt1) {

	Eigen::MatrixXd of(pt1.rows(), pt1.cols());
	of = pt1.array() - pt0.array();
	double dx = median(of.row(0));
	double dy = median(of.row(1));
	unsigned int matCols = pt0.cols(), pos = 0, len = (matCols * (matCols - 1))
			/ 2;
	Eigen::RowVectorXd d1(len);
	Eigen::RowVectorXd d2(len);
	//calculate euclidean distance
	for (unsigned int h = 0; h < matCols - 1; h++)
		for (unsigned int j = h + 1; j < matCols; j++, pos++) {
			d1(pos) = sqrt(((pt0(0, h) - pt0(0, j)) * (pt0(0, h) - pt0(0, j)))
					+ ((pt0(1, h) - pt0(1, j)) * (pt0(1, h) - pt0(1, j))));

			d2(pos) = sqrt(((pt1(0, h) - pt1(0, j)) * (pt1(0, h) - pt1(0, j)))
					+ ((pt1(1, h) - pt1(1, j)) * (pt1(1, h) - pt1(1, j))));
		}

	Eigen::RowVectorXd ds(len);
	//calculate mean value
	ds = d2.array() / d1.array();
	double s = median(ds);
	double s1 = (0.5 * (s - 1) * bb_width(bb0));
	double s2 = (0.5 * (s - 1) * bb_height(bb0));
	Eigen::Vector4d bb1;
	//
	bb1(0) = bb0(0) - s1 + dx;
	bb1(1) = bb0(1) - s2 + dy;
	bb1(2) = bb0(2) + s1 + dx;
	bb1(3) = bb0(3) + s2 + dy;
	return bb1;
}
