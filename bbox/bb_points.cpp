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

//Generates row x col points on bounding box
Eigen::Matrix<double, 2, 150> bb_points(Eigen::VectorXd const & bb, int row,
		int col, int margin, int & size) {

	Eigen::VectorXd bbin = bb;
	bbin(0) += margin;
	bbin(1) += margin;
	bbin(2) -= margin;
	bbin(3) -= margin;
	//Case : 1 row & 1 row
	if (row == 1 && col == 1) {
		Eigen::MatrixXd pt(2, 150);
		pt.col(0) = bb_center(bbin);
		size = 1;
		return pt;
	}
	//Case : 1 row & >1 cols
	if (row == 1 && col > 1) {
		Eigen::Vector2d c = bb_center(bbin);
		double stepW = (bbin(2) - bbin(0)) / (col - 1);
		std::vector<double> tupA;
		for (unsigned int i = 0; bbin(0) + i * stepW < bbin(2); i++)
			tupA.push_back(bbin(0) + i * stepW);

		tupA.push_back(bbin(2));
		Eigen::MatrixXd pt(2, 150);
		size = tupA.size();
		for (unsigned int i = 0; i < tupA.size(); i++) {
			pt(0, i) = tupA[i];
			pt(1, i) = c(1);
		}
		return pt;
	}
	//Case: 1 row & >1 cols
	if (row > 1 && col == 1) {
		Eigen::Vector2d c = bb_center(bbin);
		double stepH = (bbin(3) - bbin(1)) / (row - 1);
		std::vector<double> tupB;
		for (unsigned int i = 0; bbin(1) + i * stepH < bbin(3); i++)
			tupB.push_back(bbin(1) + i * stepH);

		tupB.push_back(bbin(3));
		Eigen::MatrixXd pt(2, 150);
		size = tupB.size();
		for (unsigned int i = 0; i < tupB.size(); i++) {
			pt(0, i) = c(0);
			pt(1, i) = tupB[i];
		}
		return pt;
	}
	//Case: >1 rows & >1 cols
	double stepW = (bbin(2) - bbin(0)) / (col - 1);
	double stepH = (bbin(3) - bbin(1)) / (row - 1);
	std::vector<double> tupA;
	for (unsigned int i = 0; bbin(0) + i * stepW < bbin(2); i++) {
		tupA.push_back(bbin(0) + i * stepW);
	}
	tupA.push_back(bbin(2));
	std::vector<double> tupB;
	for (unsigned int i = 0; bbin(1) + i * stepH < bbin(3); i++) {
		tupB.push_back(bbin(1) + i * stepH);
	}
	tupB.push_back(bbin(3));
	Eigen::MatrixXd pt(2, 150);
	size = tupA.size() * tupB.size();
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < tupB.size(); i++)
		for (unsigned int p = 0; p < tupA.size(); p++) {
			pt(0, cnt) = tupA[p];
			pt(1, cnt) = tupB[i];
			cnt++;
		}
	return pt;
}
