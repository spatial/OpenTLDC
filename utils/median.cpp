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

#include <list>
#include <Eigen/Core>
#include <algorithm>
#include <vector>
/*Calculates the median value of a given vector*/
double median(Eigen::RowVectorXd vec) {
	double median = 0;
	unsigned int len = vec.cols();
	unsigned int middle = len / 2;
	//filter NaN values
	std::vector<double> srtvec(len);
	for (unsigned int i = 0; i < len; i++){
		if (!isnan(vec(i)))
			srtvec[i] = vec(i);
		else
			srtvec[i] = 0;
	}
	//sort vector
	std::sort(srtvec.begin(), srtvec.end());
	if (len % 2 == 0) { //even length
		if (len > 0)
			median = (srtvec[middle] + srtvec[middle - 1]) / 2;
	} else { //odd length
		median = srtvec[middle];
	}
	return median;
}

