#include <vector>
#include <boost/pending/disjoint_sets.hpp>
#include "connected_components.h"

double connected_components(
	size_t numNodes,
	const double* mst,
	double threshold,
	uint64_t* components) {

	// disjoint sets datastructure to keep track of cluster merging
	std::vector<size_t> rank(numNodes);
	std::vector<uint64_t> parent(numNodes);
	boost::disjoint_sets<size_t*, uint64_t*> clusters(&rank[0], &parent[0]);

	for (size_t i = 0; i < numNodes; i++) {

		// initially, every node is in its own cluster
		clusters.make_set(i);
	}

	// merge edges

	for (size_t i = 0; i < numNodes - 1; i++) {

		uint64_t u = mst[i*3];
		uint64_t v = mst[i*3 + 1];
		double dist = mst[i*3 + 2];

		if (dist > threshold)
			break;

		uint64_t clusterU = clusters.find_set(u);
		uint64_t clusterV = clusters.find_set(v);

		assert(clusterU != clusterV);

		// link and make sure clusterU is the new root
		clusters.link(clusterU, clusterV);
		if (clusters.find_set(clusterU) == clusterV)
			std::swap(clusterU, clusterV);
	}

	// label components array

	for (size_t i = 0; i < numNodes; i++) {

		components[i] = clusters.find_set(i);
	}
}
