/**
 * @file main.cpp
 * @brief Main executable for generating Signed Distance Functions (SDF) for airfoil shapes.
 */

#include "SDFGenerator.hpp" // Include the SDFGenerator class definition.

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace {

/** @brief Checks if a pair of double values represent valid airfoil coordinate values.
 * @param x_val The x-coordinate of the point.
 * @param z_val The z-coordinate of the point.
 * @return True if the point is within the valid range, false otherwise.
 */
bool isCoordinateLine(double x_val, double z_val) {
    return x_val >= -0.5 && x_val <= 1.5 && z_val >= -1.0 && z_val <= 1.0;
}

/**
 * @brief Reconstructs a continuous airfoil boundary from a list of raw points.
 * It removes duplicate consecutive points and attempts to stitch the points into
 * a single closed loop, which is necessary for accurate SDF generation.
 * @param raw_points A vector of Point structures representing the raw coordinates read from a file.
 * @return A vector of Point structures forming a closed, clean airfoil boundary.
 */
std::vector<Point> buildBoundary(const std::vector<Point>& raw_points) {
    std::vector<Point> boundary;
    boundary.reserve(raw_points.size());

    // Iterates through the raw points and adds them to the boundary vector, skipping consecutive duplicates.
    for (const auto& point : raw_points) {
        if (boundary.empty() || point.x != boundary.back().x || point.z != boundary.back().z) {
            boundary.push_back(point);
        }
    }

    // If the first and last points are the same, remove the last point to avoid duplication.
    if (boundary.size() >= 2) {
        const Point& first = boundary.front();
        const Point& last = boundary.back();
        if (first.x == last.x && first.z == last.z) {
            boundary.pop_back(); // Remove the redundant last point.
        }
    }

    // If after cleaning, we don't have enough points to form a polygon (at least 3), return early.
    if (boundary.size() < 3) {
        return boundary;
    }

    // It looks for the largest "drop" in x-coordinate to identify a potential split point,
    // which often signifies the transition between surfaces (e.g., at the leading or trailing edge).
    size_t split_idx = 0;
    double largest_drop = 0.0;

    // Iterate through consecutive points to find the largest drop in x.
    for (size_t i = 1; i < boundary.size(); ++i) {
        const double delta_x = boundary[i].x - boundary[i - 1].x;
        if (delta_x < largest_drop) {
            largest_drop = delta_x;
            split_idx = i;
        }
    }

    // If a significant drop (heuristic: less than -0.5) was found, and it's not at the very beginning or end, attempt to stitch the boundary into a single loop.
    if (largest_drop < -0.5 && split_idx > 0 && split_idx < boundary.size()) {
        std::vector<Point> stitched;
        stitched.reserve(boundary.size() - 1);

        stitched.insert(stitched.end(), boundary.begin(), boundary.begin() + split_idx);

        // Add points from the end back towards the split index (part of the second surface, in reverse)
        // in order to connect the two surfaces into a single closed loop.
        for (size_t i = boundary.size() - 1; i > split_idx; --i) {
            stitched.push_back(boundary[i]);
        }

        // If the stitched boundary is valid (>= 3 points), return it.
        if (stitched.size() >= 3) {
            return stitched;
        }
    }

    // If no significant drop was detected or stitching failed, return the cleaned, but potentially not "stitched", boundary.
    return boundary;
}

} // namespace

/**
 * @brief Main function of the SDF generation executable.
 * 
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @return 0 on successful execution, non-zero on error.
 */
int main(int argc, char** argv) {
    // Initialize MPI environment for parallel processing.
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process.
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes.

    if (rank == 0) {
        std::cout << "Starting SDF generation..." << std::endl;
    }

    // Define the directory where airfoil data files are located.
    std::vector<std::string> dat_files;
    std::string data_dir = "data";
    
    // Check if the specified data directory exists and is actually a directory.
    if (fs::exists(data_dir) && fs::is_directory(data_dir)) {
        // Iterate through all entries in the data directory.
        for (const auto& entry : fs::directory_iterator(data_dir)) {
            // If the entry is a regular file and its extension is .dat, add its path to our list.
            if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                dat_files.push_back(entry.path().string());
            }
        }
    } else {
        // If the data directory does not exist or is not a directory, print an error message and finalize MPI.
        if (rank == 0) {
            std::cerr << "Error: Directory '" << data_dir << "' does not exist or is not a directory." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Sort the list of .dat files to ensure that all processes process the files in the exact same order.
    std::sort(dat_files.begin(), dat_files.end());

    // If no .dat files were found, report an error and exit.
    if (dat_files.empty()) {
        if (rank == 0) {
            std::cerr << "Error: No .dat files found in '" << data_dir << "'." << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Process each .dat file to generate the corresponding SDF matrix.
    for (const auto& filepath : dat_files) {
        if (rank == 0) {
            std::cout << "Processing file: " << filepath << std::endl;
        }

        std::ifstream infile(filepath);
        if (!infile) {
            if (rank == 0) {
                std::cerr << "Error opening file " << filepath << std::endl;
            }
            continue;
        }

        // Read the file line by line to extract airfoil boundary points.
        std::string line;
        // Stores raw coordinate pairs from the file.
        std::vector<Point> raw_points;
        raw_points.reserve(128);
        while (std::getline(infile, line)) {
            if (line.empty()) {
                continue;
            }
            std::istringstream iss(line);
            double x_val, z_val;
            // Attempt to extract two double values (x and z coordinates) from the line.
            if (iss >> x_val >> z_val && isCoordinateLine(x_val, z_val)) {
                raw_points.push_back({x_val, z_val});
            }
        }

        // If no valid points were parsed from the file, report an error (rank 0) and skip.
        if (raw_points.empty()) {
            if (rank == 0) {
                std::cerr << "No boundary points parsed for " << filepath << std::endl;
            }
            continue;
        }

        datFile airfoil;
        airfoil.boundary = buildBoundary(raw_points);

        if (airfoil.boundary.size() < 3) {
            if (rank == 0) {
                std::cerr << "Not enough boundary points to build SDF for " << filepath << std::endl;
            }
            continue;
        }

        // Create an SDFGenerator object with the reconstructed airfoil boundary.
        SDFGenerator generator(airfoil.boundary);
        std::vector<double> sdf = generator.generateTensorMPI();

        // The root process (rank 0) is responsible for saving the output file.
        // Other processes simply perform their part of the SDF computation and return.
        if (rank == 0) {
            // Determine output filepath, replacing the .dat extension with _matrix.txt.
            std::string out_filepath = filepath;
            if (out_filepath.size() >= 4 && out_filepath.substr(out_filepath.size() - 4) == ".dat") {
                out_filepath = out_filepath.substr(0, out_filepath.size() - 4) + "_matrix.txt";
            } else {
                // If the file somehow doesn't end with .dat, just append "_matrix.txt".
                out_filepath += "_matrix.txt";
            }

            // Use the generator to write the computed SDF values to the output file in matrix format.
            generator.mappingToMatrix(sdf, out_filepath);
            std::cout << "SDF generated with " << sdf.size() << " samples." << std::endl;
            std::cout << "SDF matrix written to " << out_filepath << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}