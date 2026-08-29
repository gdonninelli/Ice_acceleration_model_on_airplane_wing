#include "LRScheduler.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

// Constant Learning Rate Scheduler

ConstantLR::ConstantLR(float learning_rate) : _learning_rate(learning_rate) {
    if (!std::isfinite(_learning_rate) || _learning_rate <= 0.0f) {
        throw std::invalid_argument("ConstantLR learning_rate must be finite and positive.");
    }
}

float ConstantLR::get_rate(size_t /*epoch*/, size_t /*total_epochs*/) const {
    return _learning_rate;
}

std::string ConstantLR::name() const {
    std::ostringstream ss;
    ss << "constant(lr=" << _learning_rate << ")";
    return ss.str();
}

// Step Learning Rate Scheduler

StepLR::StepLR(float initial_lr, size_t step_size, float gamma)
    : _initial_lr(initial_lr), _step_size(step_size), _gamma(gamma) {
    if (!std::isfinite(_initial_lr) || _initial_lr <= 0.0f) {
        throw std::invalid_argument("StepLR initial_lr must be finite and positive.");
    }
    if (_step_size == 0) {
        throw std::invalid_argument("StepLR step_size must be greater than zero.");
    }
    if (!std::isfinite(_gamma) || _gamma <= 0.0f || _gamma > 1.0f) {
        throw std::invalid_argument("StepLR gamma must be in the range (0, 1].");
    }
}

float StepLR::get_rate(size_t epoch, size_t /*total_epochs*/) const {
    const size_t steps = epoch / _step_size;
    return _initial_lr * std::pow(_gamma, static_cast<float>(steps));
}

std::string StepLR::name() const {
    std::ostringstream ss;
    ss << "step(lr0=" << _initial_lr << ",step=" << _step_size
       << ",gamma=" << _gamma << ")";
    return ss.str();
}

// Cosine Annealing Learning Rate Scheduler

CosineAnnealingLR::CosineAnnealingLR(float min_lr, float max_lr)
    : _min_lr(min_lr), _max_lr(max_lr) {
    if (!std::isfinite(_min_lr) || _min_lr <= 0.0f) {
        throw std::invalid_argument("CosineAnnealingLR min_lr must be finite and positive.");
    }
    if (!std::isfinite(_max_lr) || _max_lr < _min_lr) {
        throw std::invalid_argument("CosineAnnealingLR max_lr must be finite and >= min_lr.");
    }
}

float CosineAnnealingLR::get_rate(size_t epoch, size_t total_epochs) const {
    if (total_epochs <= 1) {
        return _max_lr;
    }
    const float fraction = std::clamp(
        static_cast<float>(epoch) / static_cast<float>(total_epochs - 1),
        0.0f, 1.0f);
    return _min_lr + 0.5f * (_max_lr - _min_lr) *
                         (1.0f + std::cos(std::numbers::pi_v<float> * fraction));
}

std::string CosineAnnealingLR::name() const {
    std::ostringstream ss;
    ss << "cosine(min=" << _min_lr << ",max=" << _max_lr << ")";
    return ss.str();
}

// Warmup Cosine Learning Rate Scheduler

WarmupCosineLR::WarmupCosineLR(float start_lr,
                               float peak_lr,
                               float min_lr,
                               size_t warmup_epochs)
    : _start_lr(start_lr),
      _peak_lr(peak_lr),
      _min_lr(min_lr),
      _warmup_epochs(warmup_epochs) {
    if (!std::isfinite(_start_lr) || _start_lr <= 0.0f) {
        throw std::invalid_argument("WarmupCosineLR start_lr must be finite and positive.");
    }
    if (!std::isfinite(_peak_lr) || _peak_lr < _start_lr) {
        throw std::invalid_argument("WarmupCosineLR peak_lr must be finite and >= start_lr.");
    }
    if (!std::isfinite(_min_lr) || _min_lr <= 0.0f || _min_lr > _peak_lr) {
        throw std::invalid_argument("WarmupCosineLR min_lr must be finite and <= peak_lr.");
    }
}

float WarmupCosineLR::get_rate(size_t epoch, size_t total_epochs) const {
    if (_warmup_epochs > 0 && epoch < _warmup_epochs) {
        const float alpha = static_cast<float>(epoch) / static_cast<float>(_warmup_epochs);
        return _start_lr + alpha * (_peak_lr - _start_lr);
    }

    const size_t decay_epochs = (total_epochs > _warmup_epochs)
                                    ? (total_epochs - _warmup_epochs)
                                    : 1;
    const size_t current_decay_epoch = (epoch >= _warmup_epochs) ? (epoch - _warmup_epochs) : 0;
    const float fraction = (decay_epochs <= 1)
                               ? 1.0f
                               : std::clamp(static_cast<float>(current_decay_epoch) /
                                                static_cast<float>(decay_epochs - 1),
                                            0.0f, 1.0f);

    return _min_lr + 0.5f * (_peak_lr - _min_lr) *
                         (1.0f + std::cos(std::numbers::pi_v<float> * fraction));
}

std::string WarmupCosineLR::name() const {
    std::ostringstream ss;
    ss << "warmup_cosine(start=" << _start_lr << ",peak=" << _peak_lr
       << ",min=" << _min_lr << ",warmup=" << _warmup_epochs << ")";
    return ss.str();
}
