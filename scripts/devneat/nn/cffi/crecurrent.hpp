#ifndef CRECURRENT_HPP
#define CRECURRENT_HPP

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <vector>
#include <tuple>
#include <array>
#include <functional>

class RecurrentNetwork {
public:
    RecurrentNetwork() = default;
    RecurrentNetwork(int num_inputs, int num_outputs, const boost::python::list& nodes, const boost::python::list& conns, const boost::python::list& activation_defs);
    boost::python::list activate(const boost::python::list& inputs);
    void reset();
    RecurrentNetwork clone() const noexcept;
private:
    boost::python::list m_outputs;
    int num_inputs;
    int num_outputs;
    int active;
    std::vector<std::tuple<double, int>> nodes;
    std::vector<std::tuple<int, int, double>> conns;
    std::vector<std::function<float(float)>> activations;
    std::array<std::vector<double>, 2> values;
};

#endif