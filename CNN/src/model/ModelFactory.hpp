#ifndef MODELFACTORY_HPP
#define MODELFACTORY_HPP

#include "model/CNNModel.hpp"
#include "tuning/TrialConfig.hpp"
#include <cstdint>
#include <memory>
#include <mpi.h>

class ModelFactory {
public:
    std::unique_ptr<CNNModel> build(const TrialConfig& config,
                                    size_t sdf_height,
                                    size_t sdf_width,
                                    size_t scalar_features,
                                    uint64_t seed,
                                    MPI_Comm communicator = MPI_COMM_WORLD) const;
};

#endif // MODELFACTORY_HPP
