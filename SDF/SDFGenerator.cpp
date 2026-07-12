/**
 * @file SDFGenerator.cpp
 * @brief Implementation file for the SDFGenerator class.
 * It relies on the declarations in SDFGenerator.hpp.
 */

#include "SDFGenerator.hpp"

#include <fstream>
#include <limits>

void SDFGenerator::computeCord(const datFile& airfoil) {
    double max_x = std::numeric_limits<double>::lowest();
    double min_x = std::numeric_limits<double>::max();
    double max_z = 0.0;
    double min_z = 0.0;

    // Iterate through the boundaary points to find the maximum and minimum x coordinate and their corresponding z coordinates.
    for (const auto& point : airfoil.boundary) {
        if (point.x > max_x) {
            max_x = point.x;
            max_z = point.z;
        }
        if (point.x < min_x) {
            min_x = point.x;
            min_z = point.z;
        }
    }
    
    // Compute the chord size as the distance between the points with maximum and minimum x coordinates.
    // This is equivalent to sqrt((max_x-min_x)^2 + (max_z_at_max_x-min_z_at_min_x)^2).
    cord_size = std::hypot(max_x - min_x, max_z - min_z);
    if (cord_size == 0.0) {
        cord_size = 1.0;
    }
}


void SDFGenerator::normalizePoint(datFile& airfoil) {
    // Divide each x and z coordinate of the airfoil boundary points by the chord size to normalize them.
    for (auto& point : airfoil.boundary) {
        point.x /= cord_size;
        point.z /= cord_size;
    }
}


double SDFGenerator::crossProduct(Point p, Point a, Point b) const {
    return (a.x - p.x) * (b.z - p.z) - (b.x - p.x) * (a.z - p.z);
}


double SDFGenerator::computeDistance(Point p, Point a, Point b) const {
        double norm2 = std::pow(a.x - b.x, 2) + std::pow(a.z - b.z, 2);
        // If norm2 = 0 than a and b are the same point, so we return the distance from p to a (or b, since they are the same).
        if (norm2 == 0.0) return std::hypot(p.x - a.x, p.z - a.z);

        // Compute the projection of point p onto the line defined by points a and b, and clamp it to the segment [a,b].
        double t = ((p.x - a.x) * (b.x - a.x) + (p.z - a.z) * (b.z - a.z)) / norm2;
        t = std::max(0.0, std::min(1.0,t)); // clamping between [0,1]
        Point projection = {a.x + t * (b.x - a.x), a.z + t * (b.z - a.z)}; // projection along the segment [a,b]
        
        // Return the distance from point p to the projection on the segment (a.k.a. the shortest one).
        double distance = std::hypot(p.x - projection.x, p.z - projection.z);
        return distance;
    }


double SDFGenerator::getMinDistance(Point p) const {
    double min_dist = std::numeric_limits<double>::max();
    int n = static_cast<int>(airfoil.boundary.size());

    // Iterate through each edge of the airfoil boundary and compute the distance between point p and the edge defined by consecutive boundary points.
    for (int i = 0; i < n; ++i) {
        double dist = computeDistance(
            p,
            airfoil.boundary[i],
            airfoil.boundary[(i + 1) % n]
        );
        min_dist = std::min(min_dist, dist);
    }
    return min_dist;
}


bool SDFGenerator::windingAlgorithm(Point p) const {
    int wn = 0;
    int n = static_cast<int>(airfoil.boundary.size());
    if (n < 3) return false; // No closed shape can be 2 

    // Iterate through each edge of the airfoil boundary and count the number of times the edges wind around the point p.
    for (int i = 0; i < n; ++i) {
        Point v1 = airfoil.boundary[i];
        Point v2 = airfoil.boundary[(i + 1) % n];

        if (v1.z <= p.z) {
            // An upward crossing: edge starts below or at point's z-level and ends above it.
            // In this case, the point must be to the left of the edge (v1->v2) (for this reason the cross product need to be positive)
            if (v2.z > p.z && crossProduct(v1, v2, p) > 0) {
                ++wn; 
            }
        } else {
            // A downward crossing: edge starts above point's z-level and ends below or at it.
            // In this case, the point must be to the right of the edge (v1->v2) (for this reason the cross product need to be negative)
            if (v2.z <= p.z && crossProduct(v1, v2, p) < 0) {
                --wn; 
            }
        }
    }
    // The point is inside if and only if the winding number is non-zero.
    return wn != 0;
}
 

SDFGenerator::SDFGenerator(const std::vector<Point>& boundary) : airfoil{boundary} {
    computeCord(airfoil);
    normalizePoint(airfoil);
}


void SDFGenerator::mappingToMatrix(const std::vector<double>& sdf, const std::string& filename) {
    const int expected_size = GRID_SIZE * GRID_SIZE;
    if (static_cast<int>(sdf.size()) != expected_size) {
        std::cerr << "Invalid SDF size: expected " << expected_size
                  << ", got " << sdf.size() << std::endl;
        return;
    }

    // Open the output file for writing the SDF matrix.
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening output file " << filename << std::endl;
        return;
    }

    // Write the SDF values to the file in a matrix format, with each row corresponding to a line in the file.
    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            if (col > 0) {
                outfile << ' ';
            }
            outfile << sdf[row * GRID_SIZE + col];
        }
        outfile << '\n';
    }
}


std::vector<double> SDFGenerator::generateTensorMPI() {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the unique id of the current process
        MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

        int total_points = GRID_SIZE * GRID_SIZE;
        
        // Calculate data distribution for MPI
        int points_per_rank = total_points / size;
        int remainder = total_points % size;
        
        // Distribute remainder among the first few ranks
        int local_count = points_per_rank + (rank < remainder ? 1 : 0);
        int local_offset = rank * points_per_rank + std::min(rank, remainder);

        // Allocate memory for the SDF values that this rank will compute.
        std::vector<double> local_sdf(local_count);

        // Calculate the spatial step size for the grid in x and z directions.
        // There are GRID_SIZE points, meaning GRID_SIZE-1 intervals.
        double dx = (X_MAX - X_MIN) / (GRID_SIZE - 1);
        double dz = (Z_MAX - Z_MIN) / (GRID_SIZE - 1);

        // Each rank iterates through its assigned subset of grid points.
        for (int i = 0; i < local_count; ++i) {
            int global_idx = local_offset + i;
            
            // Map 1D index back to 2D grid coordinates (row, col)
            int row = global_idx / GRID_SIZE;
            int col = global_idx % GRID_SIZE;

            // Convert grid coordinates (row, col) to spatial coordinates (x, z).
            Point p = {
                X_MIN + col * dx,
                Z_MAX - row * dz // Assuming Z axis goes down visually in the tensor
            };

            // Compute the minimum distance from point 'p' to the airfoil boundary.
            double dist = getMinDistance(p);
            if (windingAlgorithm(p)) {
                dist = -dist; // Negative distance for points inside the airfoil.
            }
            local_sdf[i] = dist;
        }

        // Prepare for gathering all local SDF chunks into a global SDF vector.
        std::vector<double> global_sdf(total_points);
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size); // Offsets for each rank's data in the global array.

        // Calculate the receive counts and displacements for all ranks.
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = points_per_rank + (i < remainder ? 1 : 0);
            displs[i] = i * points_per_rank + std::min(i, remainder);
        }

        // Stitch the local chunks together so every rank has the full SDF image.
        MPI_Allgatherv(local_sdf.data(),   // Pointer to the local SDF data.
                        local_count,       // Number of elements in the local SDF chunk.
                        MPI_DOUBLE,        // Data type of elements.
                        global_sdf.data(), // Array of counts for each process's received data.
                        recvcounts.data(), // Array of displacements for each process's data.
                        displs.data(),     // Offsets of each rank's chunk in the global array.
                        MPI_DOUBLE,        // Data type of elements to receive.
                        MPI_COMM_WORLD);   // MPI communicator for the operation.

        return global_sdf;
    }