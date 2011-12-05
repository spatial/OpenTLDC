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

/**
 * Generates random features for training set.
 *
 * @param tld learning structure
 * @param nTREES number of columns in features
 * @param nFEAT number of values per tree
 */
void tldGenerateFeatures(TldStruct& tld, unsigned int nTREES,
		unsigned int nFEAT) {

	const double SHIFT = 0.2;
	const double SCA = 1;

	const unsigned int N = 6;
	std::vector<double> a(N), b(N);

	// all values between zero and one
	for (unsigned int i = 0; i < N; i++)
		a[i] = (i * SHIFT);

	for (unsigned int i = 0; i < N; i++)
		b[i] = (i * SHIFT);

	Eigen::Matrix4Xd x(4, 8 * N * N);
	Eigen::Matrix4Xd x1(4, N * N);

	unsigned int column = 0;

	// all possible tuples
	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++) {
			x1.col(column) << a[i], b[j], a[i], b[j];
			column++;
		}

	Eigen::Matrix4Xd x2(4, 2 * N * N);
	x2 << x1, x1.array() + (SHIFT / 2);

	unsigned int len = x2.cols();

	// add / sub random values between zero and 1
	Eigen::Matrix4Xd r(4, len);
	r = x2;
	for (unsigned int i = 0; i < len; i++)
		r(2, i) += ((SCA * uniform()) + SHIFT);

	Eigen::Matrix4Xd l(4, len);
	l = x2;
	for (unsigned int i = 0; i < len; i++)
		l(2, i) -= ((SCA * uniform()) + SHIFT);

	Eigen::Matrix4Xd t(4, len);
	t = x2;
	for (unsigned int i = 0; i < len; i++)
		t(3, i) -= ((SCA * uniform()) + SHIFT);

	Eigen::Matrix4Xd bo(4, len);
	bo = x2;
	for (unsigned int i = 0; i < len; i++)
		bo(3, i) += ((SCA * uniform()) + SHIFT);

	// right, left, top, bottom border
	x << r, l, t, bo;

	std::vector<int> lo;
	int cnt = 0;

	// find coefficients that are less than 1 and greater than 0 in rows 0 and 1
	for (unsigned int i = 0; i < 8 * N * N; i++) {
		if (x(0, i) < 1 && x(1, i) < 1 && x(0, i) > 0 && x(1, i) > 0) {
			lo.push_back(1);
			cnt++;
		} else
			lo.push_back(0);

	}

	Eigen::Matrix4Xd y(4, cnt);
	unsigned int p = 0;

	// values greater than 1 will be set to 1, less than 0 to 0
	for (unsigned int i = 0; i < 8 * N * N; i++) {
		if (lo[i] == 1) {
			y.col(p) = x.col(i);
			if (y(2, p) > 1)
				y(2, p) = 1;
			else if (y(2, p) < 0)
				y(2, p) = 0;
			if (y(3, p) > 1)
				y(3, p) = 1;
			else if (y(3, p) < 0)
				y(3, p) = 0;

			p++;
		}
	}

	// random permutation
	y = permutate_cols(y);

	// cut first elements with length of nFEAT times nTREES
	y = y.block(0, 0, 4, (nFEAT * nTREES));

	Eigen::MatrixXd z(4 * nFEAT, nTREES); // 52 x 10

	// reshape
	for (unsigned int i = 0; i < nTREES; i++)
		for (p = 0; p < nFEAT; p++)
			z.block(p * 4, i, 4, 1) = y.block(0, i * nFEAT + p, 4, 1);

	tld.features = z;

}

