#ifndef SEARCHSPACE_HPP
#define SEARCHSPACE_HPP

#include "TrialConfig.hpp"
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class CandidateSource {
public:
    virtual ~CandidateSource() = default;
    virtual std::vector<TrialConfig> candidates() const = 0;
};

template <typename T>
struct NamedChoice {
    std::string label;
    T value;
};

class ParameterGrid : public CandidateSource {
public:
    explicit ParameterGrid(TrialConfig base) : _base(std::move(base)) {}

    template <typename T, typename Setter>
    void add_choice(std::string name,
                    std::vector<NamedChoice<T>> choices,
                    Setter setter) {
        if (name.empty()) {
            throw std::invalid_argument("Search axis name must not be empty.");
        }
        if (choices.empty()) {
            throw std::invalid_argument("Search axis must contain at least one choice.");
        }
        if (_axis_names.contains(name)) {
            throw std::invalid_argument("Duplicate search axis: " + name);
        }

        Axis axis;
        axis.name = name;
        axis.options.reserve(choices.size());
        std::set<std::string> labels;
        for (auto& choice : choices) {
            if (choice.label.empty()) {
                throw std::invalid_argument("Search choice label must not be empty.");
            }
            if (!labels.insert(choice.label).second) {
                throw std::invalid_argument(
                    "Duplicate choice label in search axis '" + name + "'.");
            }
            Option option;
            option.label = std::move(choice.label);
            option.apply = [value = std::move(choice.value), setter](TrialConfig& config) {
                std::invoke(setter, config, value);
            };
            axis.options.push_back(std::move(option));
        }
        _axis_names.insert(std::move(name));
        _axes.push_back(std::move(axis));
    }

    std::vector<TrialConfig> candidates() const override {
        std::vector<TrialConfig> expanded{_base};
        for (const auto& axis : _axes) {
            std::vector<TrialConfig> next;
            next.reserve(expanded.size() * axis.options.size());
            for (const auto& candidate : expanded) {
                for (const auto& option : axis.options) {
                    TrialConfig configured = candidate;
                    option.apply(configured);
                    configured.selected_parameters[axis.name] = option.label;
                    next.push_back(std::move(configured));
                }
            }
            expanded = std::move(next);
        }

        for (auto& candidate : expanded) {
            std::string name = _base.name;
            bool first = true;
            for (const auto& [parameter, value] : candidate.selected_parameters) {
                name += first ? " [" : ", ";
                name += parameter + "=" + value;
                first = false;
            }
            if (!first) {
                name += "]";
            }
            candidate.name = std::move(name);
        }
        return expanded;
    }

private:
    struct Option {
        std::string label;
        std::function<void(TrialConfig&)> apply;
    };

    struct Axis {
        std::string name;
        std::vector<Option> options;
    };

    TrialConfig _base;
    std::vector<Axis> _axes;
    std::set<std::string> _axis_names;
};

#endif // SEARCHSPACE_HPP
