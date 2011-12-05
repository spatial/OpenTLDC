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
#include "../mex/mex.h"
#include "../utils/utility.h"
#include <limits>
#include <list>

Eigen::MatrixXd bb_cluster_confidence(Eigen::MatrixXd const & iBB,
		Eigen::VectorXd const & iConf) {
	double SPACE_THR = 0.5;
	Eigen::VectorXd T;
	Eigen::VectorXd Tbak;
	unsigned int iBBcols = iBB.cols();
	Eigen::MatrixXd bdist;
	//Calculates the index of the bb that fits the best
	switch (iBBcols) {
	//0 cols, set indices to 1
	case 0:
		T = Eigen::VectorXd::Zero(1);
		break;
	//1 col, set index to 0;
	case 1:
		T.resize(1);
		T(0) = 0;
		break;
	//2 cols, set indices to zero; if above treshhold to 1
	case 2:
		T = Eigen::VectorXd::Zero(2);
		bdist = bb_distance(iBB);
		if (bdist(0, 0) > SPACE_THR) {
			T(1) = 1;
		}
		break;
	//workaround for clustering.
	default:
		Eigen::Vector4d meanBB = iBB.rowwise().mean();
		int maxIndex = 0;
		double maxDist = 10;
		for (int penis = 0; penis < iBB.cols(); penis++) {
			Eigen::MatrixXd bd = bb_distance(iBB.col(penis), meanBB);
			//save the shortest distance
			if (bd(0, 0) < maxDist) {
				maxIndex = penis;
				maxDist = bd(0, 0);
			}
		}
		//set the indices to the index of the bounding box with the
		//shortest distance
		T = Eigen::VectorXd::Constant(iBB.cols(), maxIndex);
		break;
	}
	Eigen::VectorXd idx_cluster;
	idx_cluster.resize(0);
	bool breaker;
	//filter indices that occur twice
	for (int p = 0; p < T.size(); p++) {
		breaker = false;
		for (int j = 0; j < idx_cluster.size(); j++)
			if (idx_cluster(j) == T(p)) {
				breaker = true;
				break;
			}
		if (breaker)
			continue;
		Eigen::VectorXd unibak(idx_cluster.size());
		unibak = idx_cluster;
		idx_cluster.resize(unibak.size() + 1);
		idx_cluster << unibak, T(p);
	}
	int num_clusters = idx_cluster.size();

	Eigen::MatrixXd oBB = Eigen::MatrixXd::Constant(4, num_clusters,
			std::numeric_limits<double>::quiet_NaN());
	Eigen::MatrixXd oConf = Eigen::MatrixXd::Constant(4, num_clusters,
			std::numeric_limits<double>::quiet_NaN());
	Eigen::MatrixXd oSize = Eigen::MatrixXd::Constant(4, num_clusters,
			std::numeric_limits<double>::quiet_NaN());

	for (int k = 0; k < num_clusters; k++) {
		std::vector<int> idx;
		for (int u = 0; u < T.size(); u++)
			if (T(u) == idx_cluster(k))
				idx.push_back(u);

		Eigen::MatrixXd iBBidx(4, idx.size());

		for (unsigned int f = 0; f < idx.size(); f++) {
			iBBidx.col(f) = iBB.block(0, idx[f], 4, 1);
		}
		oBB.col(k) = iBBidx.rowwise().mean();
		Eigen::VectorXd iConfidx(idx.size());
		for (unsigned int f = 0; f < idx.size(); f++)
			iConfidx(f) = iConf(idx[f]);
		//save information how valid a certain bounding box is
		oConf(0, k) = iConfidx.mean();
		oSize(0, k) = idx.size();
	}
	Eigen::MatrixXd ret(4, 3 * num_clusters);
	ret << oBB, oConf, oSize;
	return ret;
}

