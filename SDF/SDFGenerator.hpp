/**
 * @file SDFGenerator.hpp
 * @brief Header file for the SDFGenerator class.
 * Defines the SDFGenerator class, responsible for computing Signed Distance Functions (SDF) for a given airfoil boundary.
 */

#ifndef SDF_GENERATOR_HPP
#define SDF_GENERATOR_HPP

#include <mpi.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

/**
 * @brief Represents a point in 2D space with x and z coordinates.
 */
struct Point
{
    double x, z;
};

/**
 * @brief Represents the boundary definition of an airfoil.
 * It contains a vector of points defining the airfoil's shape.
 */
struct datFile
{
    std::vector<Point> boundary;
};

/**
 * @brief Generates Signed Distance Function (SDF) for a given airfoil boundary.
 */
class SDFGenerator {
private:
    datFile airfoil;
    
    // Grid parameters
    const int GRID_SIZE = 150;
    const double X_MIN = -0.2;
    const double X_MAX = 1.1;
    const double Z_MIN = -0.4;
    const double Z_MAX = 0.4;

    double cord_size;

    /**
     * @brief Computes the chord size of the airfoil.
     * @param airfoil The airfoil boundary points.
     */
    void computeCord(const datFile& airfoil);

    /**
     * @brief Normalizes the coordinates of the airfoil points.
     * @param airfoil The airfoil boundary points.
     */
    void normalizePoint(datFile& airfoil);

    /**
     * @brief Computes the cross product of three points.
     * This is used to determine if the point p is to the left or right of the segment ab.
     * @param p The point for which to compute the cross product.
     * @param a First point defining the vector.
     * @param b Second point defining the vector.
     * @return The cross product value.
     */
    double crossProduct(Point p, Point a, Point b) const;

    /**
     * @brief Computes the Euclidean distance from a point to a line segment.
     * @param p The point from which to compute the distance.
     * @param a First point defining the line segment.
     * @param b Second point defining the line segment.
     * @return The distance from the point to the line segment.
     */
    double computeDistance(Point p, Point a, Point b) const;  
    
    /**
     * @brief Computes the minimum distance from a point to the airfoil boundary.
     * @param p The point for which to compute the minimum distance.
     * @return The minimum distance from the point to the airfoil boundary.
     */
    double getMinDistance(Point p) const;
    
    /**
     * @brief Determines if a point is inside the airfoil boundary using the winding number algorithm.
     * @param p The point to check for being inside the airfoil.
     * @return True if the point is inside the airfoil, false otherwise.
     */
    bool windingAlgorithm(Point p) const;

public:
    /**
     * @brief Constructor for SDFGenerator.
     * @param boundary A vector of points defining the airfoil boundary.
     */
    SDFGenerator(const std::vector<Point>& boundary);

    /**
     * @brief Writes the computed SDF values to a matrix and saves it to a file.
     * @param sdf A flattened vector containing the SDF values for each point in the SDF grid.
     * @param filename The name of the file to write the SDF matrix to.
     */
    void mappingToMatrix(const std::vector<double>& sdf, const std::string& filename);

    /**
     * @brief Generates the SDF values for the airfoil boundary using MPI for parallel computation.
     * Each process computes a portion of the SDF grid, and the results are gathered together.
     * @return A flattened vector representing the complete SDF grid (GRID_SIZE * GRID_SIZE).
     */
    std::vector<double> generateTensorMPI();
};

#endif // SDF_GENERATOR_HPP