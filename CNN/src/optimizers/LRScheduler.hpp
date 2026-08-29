#ifndef LRSCHEDULER_HPP
#define LRSCHEDULER_HPP

#include <cstddef>
#include <memory>
#include <string>

/**
 * @class LRScheduler
 * @brief Abstract base class for learning rate scheduling policies.
 */
class LRScheduler {
public:
    virtual ~LRScheduler() = default;

    /**
     * @brief Computes the scheduled learning rate for a given epoch.
     * @param epoch Current epoch (0-indexed).
     * @param total_epochs Total number of training epochs.
     * @return Effective learning rate for the epoch.
     */
    virtual float get_rate(size_t epoch, size_t total_epochs) const = 0;

    /**
     * @brief Returns a human-readable name of the scheduler.
     */
    virtual std::string name() const = 0;
};

/**
 * @class ConstantLR
 * @brief Fixed learning rate policy.
 */
class ConstantLR : public LRScheduler {
public:
    explicit ConstantLR(float learning_rate);

    float get_rate(size_t epoch, size_t total_epochs) const override;
    std::string name() const override;

    float learning_rate() const { return _learning_rate; }

private:
    float _learning_rate;
};

/**
 * @class StepLR
 * @brief Decays the learning rate by gamma every step_size epochs.
 */
class StepLR : public LRScheduler {
public:
    StepLR(float initial_lr, size_t step_size, float gamma = 0.1f);

    float get_rate(size_t epoch, size_t total_epochs) const override;
    std::string name() const override;

    float initial_lr() const { return _initial_lr; }
    size_t step_size() const { return _step_size; }
    float gamma() const { return _gamma; }

private:
    float _initial_lr;
    size_t _step_size;
    float _gamma;
};

/**
 * @class CosineAnnealingLR
 * @brief Cosine annealing decay from max_lr to min_lr across total_epochs.
 */
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(float min_lr, float max_lr);

    float get_rate(size_t epoch, size_t total_epochs) const override;
    std::string name() const override;

    float min_lr() const { return _min_lr; }
    float max_lr() const { return _max_lr; }

private:
    float _min_lr;
    float _max_lr;
};

/**
 * @class WarmupCosineLR
 * @brief Linear warmup followed by cosine annealing decay.
 */
class WarmupCosineLR : public LRScheduler {
public:
    WarmupCosineLR(float start_lr, float peak_lr, float min_lr, size_t warmup_epochs);

    float get_rate(size_t epoch, size_t total_epochs) const override;
    std::string name() const override;

    float start_lr() const { return _start_lr; }
    float peak_lr() const { return _peak_lr; }
    float min_lr() const { return _min_lr; }
    size_t warmup_epochs() const { return _warmup_epochs; }

private:
    float _start_lr;
    float _peak_lr;
    float _min_lr;
    size_t _warmup_epochs;
};

#endif // LRSCHEDULER_HPP
